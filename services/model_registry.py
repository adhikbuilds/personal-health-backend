from __future__ import annotations

"""
Personal Health — Model Registry + Versioned Inference.

This is the backbone of the data flywheel. It:

1. Loads the trained MLP classifier (pose_classifier.h5 or .tflite)
2. Normalizes input features using saved mean/std
3. Runs inference: feature vector → quality class + confidence
4. Tracks model versions so retrains can be compared and rolled back

The PoseAnalyzer calls `predict_quality()` as the PRIMARY form scorer.
If the model isn't available (no TF, no model file, bad input), it
returns None and the PoseAnalyzer falls through to rule-based scoring.

Model files:
  models/pose_classifier.h5       — Keras model (server-side)
  models/pose_classifier.tflite   — TFLite model (on-device / lightweight)
  models/norm_params.json         — {mean, std, features} for normalization
  models/training_metrics.json    — {accuracy, epochs, class_labels, ...}
  models/registry.json            — version history + active model pointer
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logging_setup import get_logger
from services import sports_catalog

log = get_logger("services.model_registry")

_MODELS_DIR = Path(__file__).parent.parent / "models"
_REGISTRY_PATH = _MODELS_DIR / "registry.json"

# Sport index mapping is sourced from the unified catalog so adding a new
# sport doesn't require editing this file. Kept as a module-level name
# for any external callers that have imported it directly.
SPORT_INDEX = sports_catalog.build_sport_index_map()

QUALITY_CLASSES = ["poor", "average", "good", "elite"]

# Quality class → form score midpoint (used to convert class prediction
# back to a 0-100 score range for backward-compatibility)
QUALITY_SCORE_MAP = {
    "poor": 35.0,
    "average": 60.0,
    "good": 80.0,
    "elite": 95.0,
}


# ─── Model Version Tracking ────────────────────────────────────────────────


@dataclass
class ModelVersion:
    version: str  # "v1", "v2", ...
    trained_at: str  # ISO timestamp
    data_source: str  # "synthetic", "real", "mixed"
    accuracy: float  # test accuracy 0-1
    epochs: int
    real_frames: int  # how many real frames were in the training set
    synthetic_frames: int
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _load_registry() -> dict:
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    return {"active_version": None, "versions": []}


def _save_registry(data: dict) -> None:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def register_model(version: ModelVersion) -> None:
    """Register a new model version after training."""
    reg = _load_registry()
    reg["versions"].append(version.to_dict())
    reg["active_version"] = version.version
    _save_registry(reg)
    log.info("model registered", extra={"version": version.version, "accuracy": version.accuracy})


def get_active_version() -> str | None:
    return _load_registry().get("active_version")


def get_registry() -> dict:
    return _load_registry()


def set_active_version(version: str) -> bool:
    """Switch the active model version (for rollback or A/B testing)."""
    reg = _load_registry()
    known = {v["version"] for v in reg.get("versions", [])}
    if version not in known:
        return False
    reg["active_version"] = version
    _save_registry(reg)
    _MODEL_CACHE.clear()  # force reload
    log.info("active model switched", extra={"version": version})
    return True


# ─── Model Loading + Inference ──────────────────────────────────────────────

# Thread-safe model cache so we only load once.
_MODEL_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def _load_norm_params() -> tuple[list[float], list[float], list[str]] | None:
    norm_path = _MODELS_DIR / "norm_params.json"
    if not norm_path.exists():
        return None
    with open(norm_path) as f:
        d = json.load(f)
    return d["mean"], d["std"], d["features"]


def _load_keras_model():
    """Try to load the .h5 model via TensorFlow/Keras."""
    try:
        import tensorflow as tf

        model_path = _MODELS_DIR / "pose_classifier.h5"
        if not model_path.exists():
            return None
        model = tf.keras.models.load_model(str(model_path), compile=False)
        log.info("keras model loaded", extra={"path": str(model_path)})
        return model
    except Exception as e:
        log.warning("keras model load failed", extra={"error": str(e)})
        return None


def _load_tflite_model():
    """Try to load the .tflite model (lighter, no TF dependency)."""
    try:
        import numpy as np

        tflite_path = _MODELS_DIR / "pose_classifier.tflite"
        if not tflite_path.exists():
            return None
        # Try tflite_runtime first, then tensorflow.lite
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        interpreter = tflite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        log.info("tflite model loaded", extra={"path": str(tflite_path)})
        return interpreter
    except Exception as e:
        log.warning("tflite model load failed", extra={"error": str(e)})
        return None


def _get_model():
    """Load model (cached). Tries keras first, then tflite."""
    with _CACHE_LOCK:
        if "model" in _MODEL_CACHE:
            return _MODEL_CACHE["model"], _MODEL_CACHE.get("type")
        if "norm" not in _MODEL_CACHE:
            _MODEL_CACHE["norm"] = _load_norm_params()

        # Try keras first (more accurate, supports GPU)
        model = _load_keras_model()
        if model is not None:
            _MODEL_CACHE["model"] = model
            _MODEL_CACHE["type"] = "keras"
            return model, "keras"

        # Fallback to tflite (lighter)
        model = _load_tflite_model()
        if model is not None:
            _MODEL_CACHE["model"] = model
            _MODEL_CACHE["type"] = "tflite"
            return model, "tflite"

        _MODEL_CACHE["model"] = None
        _MODEL_CACHE["type"] = None
        return None, None


def _normalize_features(raw: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(v - m) / s for v, m, s in zip(raw, mean, std)]


def predict_quality(
    frame_data: dict,
    sport: str = "vertical_jump",
) -> dict | None:
    """
    Run the trained MLP classifier on a single frame's feature vector.

    Args:
        frame_data: dict with keys matching norm_params features
                    (hip_angle_l, knee_angle_l, ..., limb_symmetry_idx)
        sport: sport key for the sport_idx feature

    Returns:
        {"quality": "good", "confidence": 0.87, "scores": [0.01, 0.05, 0.87, 0.07],
         "form_score": 80.0, "source": "model"}
        or None if model isn't available.
    """
    model, model_type = _get_model()
    if model is None:
        return None

    norm = _MODEL_CACHE.get("norm")
    if norm is None:
        return None

    mean, std, feature_names = norm

    # Build feature vector in the same order as training
    try:
        raw = []
        for fname in feature_names:
            if fname == "sport_idx":
                raw.append(sports_catalog.sport_index(sport) / 4.0)
            else:
                val = frame_data.get(fname, 0.0)
                raw.append(float(val) if val is not None else 0.0)

        normalized = _normalize_features(raw, mean, std)
    except Exception as e:
        log.warning("feature extraction failed", extra={"error": str(e)})
        return None

    # Run inference
    try:
        import numpy as np

        if model_type == "keras":
            input_arr = np.array([normalized], dtype=np.float32)
            probs = model.predict(input_arr, verbose=0)[0]
        elif model_type == "tflite":
            input_arr = np.array([normalized], dtype=np.float32)
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            model.set_tensor(input_details[0]["index"], input_arr)
            model.invoke()
            probs = model.get_tensor(output_details[0]["index"])[0]
        else:
            return None

        probs = [float(p) for p in probs]
        predicted_idx = int(np.argmax(probs))
        quality = QUALITY_CLASSES[predicted_idx]
        confidence = probs[predicted_idx]

        # Convert class to form score (weighted average by class probabilities)
        form_score = sum(probs[i] * QUALITY_SCORE_MAP[QUALITY_CLASSES[i]] for i in range(len(QUALITY_CLASSES)))
        form_score = round(min(100.0, max(0.0, form_score)), 1)

        return {
            "quality": quality,
            "confidence": round(confidence, 3),
            "class_probabilities": dict(zip(QUALITY_CLASSES, [round(p, 4) for p in probs])),
            "form_score": form_score,
            "source": "model",
            "model_type": model_type,
        }
    except Exception as e:
        log.warning("model inference failed", extra={"error": str(e)})
        return None


def reload_model() -> bool:
    """Force reload the model from disk (after a retrain)."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
    model, mtype = _get_model()
    return model is not None
