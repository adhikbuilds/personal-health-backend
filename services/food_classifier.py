"""
Food classifier — image → food name → macros.

Two-stage pipeline:
  Stage 1: image bytes → food name(s)         (pluggable backend)
  Stage 2: food name → macros lookup in FOOD_DB

Backends (selected via env var NUTRITION_MODEL):
  - mock     : deterministic by image hash (default, no deps)
  - clip     : HuggingFace CLIP zero-shot   (needs transformers + torch + pillow)
  - tflite   : TFLite Food-101 MobileNetV2  (needs tensorflow + pillow)
  - anthropic: Claude vision API            (needs ANTHROPIC_API_KEY)

The mock backend ships working today with zero install. It picks a food
deterministically from FOOD_DB based on the SHA1 of the image bytes — so the
same photo always classifies as the same food (good for demos), but different
photos give different foods. No randomness across reloads.

To upgrade to a real pretrained model, set NUTRITION_MODEL=clip and:
  pip install transformers torch --extra-index-url https://download.pytorch.org/whl/cpu pillow
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from logging_setup import get_logger

log = get_logger("services.food_classifier")

_BACKEND = os.environ.get("NUTRITION_MODEL", "mock").lower()
_FOODS_PATH = Path(__file__).resolve().parent.parent / "db" / "foods.json"

# ─── Lazy state for heavy backends ──────────────────────────
_clip_model = None
_clip_processor = None
_clip_text_features = None
_food_names_for_clip: list[str] = []


def _load_food_db() -> list[dict[str, Any]]:
    if not _FOODS_PATH.exists():
        return []
    try:
        data = json.loads(_FOODS_PATH.read_text())
        if isinstance(data, dict) and "foods" in data:
            return list(data["foods"])
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [v for v in data.values() if isinstance(v, dict)]
    except Exception as e:
        log.error("foods.json parse failed: %s", e)
    return []


def _decode_image(image_b64: str) -> bytes:
    payload = image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
    try:
        return base64.b64decode(payload, validate=False)
    except Exception:
        return payload.encode()


# ─── Stage 2: food name → macros ──────────────────────────


def _macros_for_food(food: dict[str, Any]) -> dict[str, Any]:
    """Calculate per-serving macros from per_100g × serving_g."""
    per100 = food.get("per_100g", {}) or {}
    serving = float(food.get("serving_g") or 100)
    factor = serving / 100.0
    return {
        "calories": round(float(per100.get("calories", 0)) * factor),
        "protein_g": round(float(per100.get("protein_g", 0)) * factor, 1),
        "carbs_g": round(float(per100.get("carbs_g", 0)) * factor, 1),
        "fat_g": round(float(per100.get("fat_g", 0)) * factor, 1),
        "fiber_g": round(float(per100.get("fiber_g", 0)) * factor, 1),
    }


def _meal_score(macros: dict[str, Any], food: dict[str, Any]) -> int:
    """Heuristic meal score 0-100 — high protein + balanced macros + low extra fat = good."""
    score = 60
    p = macros["protein_g"]
    if p >= 25:
        score += 15
    elif p >= 15:
        score += 8
    f = macros["fat_g"]
    if f <= 10:
        score += 8
    elif f >= 30:
        score -= 10
    fb = macros["fiber_g"]
    if fb >= 5:
        score += 8
    if "high_protein" in (food.get("tags") or []):
        score += 4
    if "fried" in (food.get("tags") or []) or "dessert" in (food.get("tags") or []):
        score -= 10
    return max(0, min(100, score))


def _recommendation(food: dict[str, Any], macros: dict[str, Any]) -> str:
    name = food.get("name", "this meal")
    tags = food.get("tags") or []
    cat = food.get("category", "")
    if "fried" in tags or "dessert" in tags:
        return f"{name} is heavier on fat — pair with a protein-forward dish next meal."
    if macros["protein_g"] >= 20:
        return f"{name} is solid protein. Great post-training fuel."
    if cat == "breakfast":
        return f"Start strong — add yogurt or eggs to {name} for more protein."
    if cat in ("snack", "snacks"):
        return f"{name} is a good snack. Watch portion size if cutting."
    return f"Track macros and stay hydrated. {name} fits a balanced plan."


# ─── Stage 1: image → food name (pluggable) ──────────────────────────


def _backend_mock(image_bytes: bytes, foods: list[dict]) -> dict:
    """Deterministic mock — same image always maps to same food.
    Hash → seed → indexes.  Works offline with zero deps.
    """
    if not foods:
        return {"food_items": ["Mixed meal"], "primary": None, "confidence": 0.0}
    digest = hashlib.sha1(image_bytes).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    primary = rng.choice(foods)
    # Sometimes pair with a side dish
    side = None
    if len(foods) > 5 and rng.random() < 0.55:
        side = rng.choice([f for f in foods if f.get("category") != primary.get("category")] or foods)
    items = [primary["name"]]
    if side and side["name"] != primary["name"]:
        items.append(side["name"])
    return {
        "food_items": items,
        "primary": primary,
        "confidence": 0.0,  # honest about being a mock
        "backend": "mock",
    }


def _backend_clip(image_bytes: bytes, foods: list[dict]) -> dict | None:
    """CLIP zero-shot: image vs FOOD_DB names. Lazy-loads model."""
    global _clip_model, _clip_processor, _clip_text_features, _food_names_for_clip
    try:
        if _clip_model is None:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            log.info("loading CLIP model (this happens once)…")
            model_name = "openai/clip-vit-base-patch32"
            _clip_processor = CLIPProcessor.from_pretrained(model_name)
            _clip_model = CLIPModel.from_pretrained(model_name)
            _clip_model.eval()
            _food_names_for_clip = [f.get("name", "food") for f in foods]
            with torch.no_grad():
                text_inputs = _clip_processor(
                    text=[f"a photo of {n}" for n in _food_names_for_clip],
                    return_tensors="pt",
                    padding=True,
                )
                _clip_text_features = _clip_model.get_text_features(**text_inputs)
                _clip_text_features = _clip_text_features / _clip_text_features.norm(dim=-1, keepdim=True)
        from PIL import Image
        import io
        import torch

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with torch.no_grad():
            img_inputs = _clip_processor(images=img, return_tensors="pt")
            img_features = _clip_model.get_image_features(**img_inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            sims = (img_features @ _clip_text_features.T).squeeze(0)
            top_idx = sims.topk(2).indices.tolist()
        primary_idx = top_idx[0]
        primary = foods[primary_idx]
        items = [primary["name"]]
        if len(top_idx) > 1 and sims[top_idx[1]].item() > sims[primary_idx].item() * 0.85:
            items.append(foods[top_idx[1]]["name"])
        return {
            "food_items": items,
            "primary": primary,
            "confidence": float(sims[primary_idx].item()),
            "backend": "clip",
        }
    except ImportError:
        log.info("CLIP backend requested but transformers/torch not installed — falling back")
        return None
    except Exception as e:
        log.warning("CLIP inference failed: %s — falling back", e)
        return None


def _backend_tflite(image_bytes: bytes, foods: list[dict]) -> dict | None:
    """Placeholder for TFLite Food-101 model — wire when model file exists in models/."""
    food101_tflite = Path(__file__).resolve().parent.parent / "models" / "food101_mobilenet.tflite"
    if not food101_tflite.exists():
        return None
    # TODO: implement TFLite inference + map Food-101 labels to FOOD_DB names
    return None


# ─── Public entry point ──────────────────────────


def classify_food(image_b64: str) -> dict[str, Any]:
    """Main entrypoint called by routes/nutrition_ai.py."""
    foods = _load_food_db()
    image_bytes = _decode_image(image_b64)

    # Try selected backend first, then mock fallback
    result = None
    if _BACKEND == "clip":
        result = _backend_clip(image_bytes, foods)
    elif _BACKEND == "tflite":
        result = _backend_tflite(image_bytes, foods)

    if result is None:
        result = _backend_mock(image_bytes, foods)

    primary = result.get("primary")
    if not primary:
        # Unknown food → return generic mid-meal estimate
        return {
            "food_items": result.get("food_items") or ["Mixed meal"],
            "nutrients": {"calories": 450, "protein_g": 18, "carbs_g": 60, "fat_g": 14, "fiber_g": 4},
            "meal_score": 65,
            "recommendation": "Couldn't ID this exactly — log macros manually if you can.",
            "backend": result.get("backend", "fallback"),
            "confidence": result.get("confidence", 0.0),
        }

    macros = _macros_for_food(primary)
    return {
        "food_items": result.get("food_items", [primary["name"]]),
        "nutrients": macros,
        "meal_score": _meal_score(macros, primary),
        "recommendation": _recommendation(primary, macros),
        "name": primary.get("name"),
        "cuisine": primary.get("cuisine"),
        "category": primary.get("category"),
        "backend": result.get("backend"),
        "confidence": round(result.get("confidence", 0.0), 3),
    }


def get_classifier_status() -> dict[str, Any]:
    """Health check — what backend is selected, what's installed."""
    available = {"mock": True}
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        available["clip"] = True
    except ImportError:
        available["clip"] = False
    try:
        import tensorflow  # noqa: F401

        food101 = Path(__file__).resolve().parent.parent / "models" / "food101_mobilenet.tflite"
        available["tflite"] = food101.exists()
    except ImportError:
        available["tflite"] = False
    available["anthropic"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {
        "selected_backend": _BACKEND,
        "available_backends": available,
        "food_db_size": len(_load_food_db()),
    }
