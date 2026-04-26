from __future__ import annotations

"""
Personal Health — AI Nutrition Analysis.

Take a photo of your food → Claude analyzes it → returns nutrient breakdown.
Falls back to a template response when API key is missing.
"""

import json
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from logging_setup import get_logger

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


@router.post("/nutrition/analyze")
async def analyze_food(req: FoodAnalysisRequest):
    """Analyze a food photo and return nutrient breakdown."""
    if not req.image_b64:
        raise HTTPException(400, "image_b64 required")

    start = time.time()
    result = _call_anthropic(req.image_b64)
    source = "anthropic"

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
            "latency_ms": latency,
            "meal_score": meal_score,
            "items": len(food_items),
        },
    )

    return {
        "source": source,
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
