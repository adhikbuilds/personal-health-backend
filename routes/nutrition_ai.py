from __future__ import annotations

"""
Personal Health — AI Nutrition Analysis.

Pipeline (in order, first hit wins):
  1. Local pretrained classifier (services.food_classifier) — works offline
     against the FOOD_DB (210 Indian/global foods). Backend selectable via
     env NUTRITION_MODEL=mock|clip|tflite.
  2. Anthropic Claude vision (if ANTHROPIC_API_KEY set) — richer recommendations.
  3. Static fallback template.

The local classifier is now the DEFAULT (was Anthropic). It always returns
something useful — no API key required to demo.
"""

import json
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from logging_setup import get_logger
from services.food_classifier import classify_food, get_classifier_status

router = APIRouter(tags=["Nutrition"])
log = get_logger("routes.nutrition_ai")

try:
    from anthropic import Anthropic

    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False


class FoodAnalysisRequest(BaseModel):
    image_b64: str
    athlete_id: str = "athlete_01"


SYSTEM_PROMPT = (
    "You are a sports nutrition expert. Analyze this food photo. "
    "Return ONLY valid JSON with these exact keys: "
    '{"food_items": ["item1", "item2"], '
    '"nutrients": {"calories": 450, "protein_g": 25, "carbs_g": 55, "fat_g": 15, "fiber_g": 8}, '
    '"meal_score": 72, '
    '"recommendation": "one sentence coaching tip"} '
    "Estimate realistic values for an Indian/global diet. "
    "meal_score is 0-100 (100 = perfect athlete meal). "
    "Be specific about food items you can identify."
)


def _call_anthropic(image_b64: str) -> dict | None:
    if not ANTHROPIC_OK or not settings.anthropic_api_key:
        return None
    try:
        client = Anthropic(api_key=settings.anthropic_api_key, timeout=15.0)
        msg = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": "Analyze this meal."},
                    ],
                }
            ],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
        # Extract JSON from response
        import re

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except Exception as e:
        log.warning("nutrition AI failed", extra={"error": str(e)})
        return None


def _fallback_analysis() -> dict:
    """Template response when API is unavailable."""
    return {
        "food_items": ["Unable to identify — AI analysis unavailable"],
        "nutrients": {
            "calories": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0,
            "fiber_g": 0,
        },
        "meal_score": 0,
        "recommendation": "Set your ANTHROPIC_API_KEY to enable AI food analysis.",
    }


@router.get("/nutrition/classifier/status")
async def classifier_status():
    """Health check — which classifier backend is active + which are installable."""
    return get_classifier_status()


@router.post("/nutrition/analyze")
async def analyze_food(req: FoodAnalysisRequest):
    """Analyze a food photo and return nutrient breakdown.

    Order: local classifier (FOOD_DB) → Anthropic Claude → static fallback.
    """
    if not req.image_b64:
        raise HTTPException(400, "image_b64 required")

    start = time.time()
    source = "local"
    result = None

    # 1. Local pretrained classifier (default — works offline)
    try:
        result = classify_food(req.image_b64)
    except Exception as e:
        log.warning("local food classifier failed: %s", e)
        result = None

    # 2. Anthropic Claude vision (if API key available and local was too low confidence)
    if ANTHROPIC_OK and settings.anthropic_api_key and (result is None or result.get("confidence", 1.0) < 0.20):
        anthropic_result = _call_anthropic(req.image_b64)
        if anthropic_result:
            result = anthropic_result
            source = "anthropic"

    # 3. Static fallback
    if result is None:
        result = _fallback_analysis()
        source = "fallback"

    # Validate structure
    nutrients = result.get("nutrients", {})
    food_items = result.get("food_items", [])
    meal_score = int(result.get("meal_score", 0))
    recommendation = result.get("recommendation", "")

    latency = round((time.time() - start) * 1000)
    log.info(
        "nutrition analysis",
        extra={
            "source": source,
            "backend": result.get("backend"),
            "latency_ms": latency,
            "meal_score": meal_score,
            "items": len(food_items),
        },
    )

    return {
        "source": source,
        "backend": result.get("backend"),  # mock | clip | tflite | anthropic
        "confidence": result.get("confidence"),
        "food_items": food_items,
        "nutrients": {
            "calories": nutrients.get("calories", 0),
            "protein_g": nutrients.get("protein_g", 0),
            "carbs_g": nutrients.get("carbs_g", 0),
            "fat_g": nutrients.get("fat_g", 0),
            "fiber_g": nutrients.get("fiber_g", 0),
        },
        "meal_score": meal_score,
        "recommendation": recommendation,
        "latency_ms": latency,
    }
