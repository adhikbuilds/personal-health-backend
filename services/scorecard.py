"""
ActiveBharat — Session Scorecard Image Generator
──────────────────────────────────────────────────────────────────────────────
Generates a 1080x1080 shareable PNG scorecard summarising an athlete's
training session.  Uses only PIL/Pillow — no external fonts or assets.

Public API:
    generate_scorecard(session_summary, athlete) -> bytes   (PNG)
    generate_scorecard_response(session_summary, athlete)   (FastAPI Response)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import math
from datetime import datetime, timezone
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# ─── Theme constants ────────────────────────────────────────────────────────

W, H = 1080, 1080

BG = "#0f172a"
ACCENT = "#06b6d4"
WHITE = "#ffffff"
WHITE_DIM = "#94a3b8"
DARK_CARD = "#1e293b"
BORDER = "#334155"

GREEN = "#22c55e"
CYAN = "#06b6d4"
ORANGE = "#f97316"
RED = "#ef4444"

SPORT_LABELS: dict[str, str] = {
    "vertical_jump": "Vertical Jump",
    "snatch": "Snatch",
    "clean_and_jerk": "Clean & Jerk",
    "deadlift": "Deadlift",
    "squat": "Squat",
    "sprint": "Sprint",
}


# ─── Font helpers ───────────────────────────────────────────────────────────


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a scalable TTF font; fall back to default bitmap font."""
    # Common paths for DejaVu on Linux (Ubuntu/Debian) — works without install
    _ttf_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in _ttf_candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    # Last resort — Pillow >=10 supports size on load_default
    try:
        return ImageFont.load_default(size=size)  # type: ignore[arg-type]
    except TypeError:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Any) -> tuple[int, int]:
    """Get (width, height) of rendered text, compatible with all Pillow versions."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ─── Drawing primitives ────────────────────────────────────────────────────


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font: Any,
    fill: str = WHITE,
    x_center: int = W // 2,
) -> None:
    tw, _ = _text_size(draw, text, font)
    draw.text((x_center - tw // 2, y), text, font=font, fill=fill)


def _score_color(score: float) -> str:
    if score > 80:
        return GREEN
    if score > 65:
        return CYAN
    if score > 50:
        return ORANGE
    return RED


def _draw_score_ring(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    radius: int,
    score: float,
    font_large: Any,
    font_small: Any,
) -> None:
    """Draw a circular score gauge with coloured arc proportional to score."""
    color = _score_color(score)
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)

    # Background ring
    draw.arc(bbox, 0, 360, fill=BORDER, width=14)

    # Score arc — 0 is 3-o'clock; start at top (-90 deg)
    sweep = score / 100 * 360
    draw.arc(bbox, start=-90, end=-90 + sweep, fill=color, width=14)

    # Inner filled darker circle
    inner_r = radius - 22
    draw.ellipse(
        (cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r),
        fill=DARK_CARD,
    )

    # Score number
    score_text = str(int(round(score)))
    _draw_centered_text(draw, score_text, cy - 38, font_large, fill=color, x_center=cx)
    _draw_centered_text(draw, "Form Score", cy + 28, font_small, fill=WHITE_DIM, x_center=cx)


def _draw_stat_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    value: str,
    font_val: Any,
    font_lbl: Any,
    accent: str = ACCENT,
) -> None:
    """Draw a rounded-rect stat tile."""
    draw.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=DARK_CARD, outline=BORDER)
    _draw_centered_text(draw, value, y + 18, font_val, fill=accent, x_center=x + w // 2)
    _draw_centered_text(draw, label, y + h - 38, font_lbl, fill=WHITE_DIM, x_center=x + w // 2)


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def _quality_bar_text(dist: dict[str, int]) -> str:
    """Produce a compact string like 'E12 G45 A50 P13'."""
    parts = []
    for key, prefix in [("elite", "E"), ("good", "G"), ("average", "A"), ("poor", "P")]:
        count = dist.get(key, 0)
        if count:
            parts.append(f"{prefix}{count}")
    return " / ".join(parts) if parts else "-"


# ─── Quality distribution mini-bar ──────────────────────────────────────────


def _draw_quality_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    dist: dict[str, int],
    font_lbl: Any,
) -> None:
    """Draw a horizontal stacked bar showing quality distribution."""
    draw.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=DARK_CARD, outline=BORDER)

    label = "Quality Split"
    _draw_centered_text(draw, label, y + h - 38, font_lbl, fill=WHITE_DIM, x_center=x + w // 2)

    total = sum(dist.values()) or 1
    bar_x = x + 24
    bar_y = y + 20
    bar_w = w - 48
    bar_h = 22
    colors = {"elite": GREEN, "good": CYAN, "average": ORANGE, "poor": RED}
    labels = {"elite": "E", "good": "G", "average": "A", "poor": "P"}

    # Background
    draw.rounded_rectangle(
        (bar_x, bar_y, bar_x + bar_w, bar_y + bar_h),
        radius=6,
        fill=BORDER,
    )

    # Segments
    cx = bar_x
    for key in ("elite", "good", "average", "poor"):
        count = dist.get(key, 0)
        seg_w = int(bar_w * count / total)
        if seg_w < 1:
            continue
        draw.rectangle((cx, bar_y, cx + seg_w, bar_y + bar_h), fill=colors[key])
        cx += seg_w

    # Legend line beneath bar
    legend_y = bar_y + bar_h + 8
    legend_parts: list[str] = []
    for key in ("elite", "good", "average", "poor"):
        count = dist.get(key, 0)
        if count:
            legend_parts.append(f"{labels[key]}:{count}")
    legend_text = "  ".join(legend_parts) if legend_parts else "-"
    _draw_centered_text(draw, legend_text, legend_y, font_lbl, fill=WHITE_DIM, x_center=x + w // 2)


# ─── Main generator ─────────────────────────────────────────────────────────


def generate_scorecard(session_summary: dict, athlete: dict) -> bytes:
    """
    Generate a 1080x1080 PNG scorecard image.

    Returns PNG file bytes.  Never raises — returns a minimal error card
    on unexpected failure.
    """
    try:
        return _build_scorecard(session_summary, athlete)
    except Exception:
        return _build_error_card()


def _build_scorecard(summary: dict, athlete: dict) -> bytes:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ── Fonts ───────────────────────────────────────────────────────────
    font_title = _load_font(42)
    font_sport = _load_font(28)
    font_score = _load_font(64)
    font_val = _load_font(36)
    font_lbl = _load_font(20)
    font_footer = _load_font(18)
    font_small = _load_font(22)

    # ── Extract values ──────────────────────────────────────────────────
    sport_raw = summary.get("sport", athlete.get("sport", "training"))
    sport_label = SPORT_LABELS.get(sport_raw, sport_raw.replace("_", " ").title())
    form_score = float(summary.get("avg_form_score", 0))
    peak_jump = summary.get("peak_jump_height_cm")
    symmetry = summary.get("avg_symmetry")
    xp = summary.get("xp_earned", 0)
    duration = summary.get("duration_seconds", 0)
    reps = summary.get("rep_count", 0)
    quality_dist = summary.get("quality_distribution", {})
    athlete_name = athlete.get("name", "Athlete")
    tier = athlete.get("tier", "")
    bpi = athlete.get("bpi")

    # ── Top section — branding ──────────────────────────────────────────
    y = 48
    _draw_centered_text(draw, "ActiveBharat", y, font_title, fill=ACCENT)
    y += 56
    _draw_centered_text(draw, sport_label, y, font_sport, fill=WHITE_DIM)
    y += 44

    # Thin accent line
    line_margin = 200
    draw.line((line_margin, y, W - line_margin, y), fill=ACCENT, width=2)

    # ── Score ring ──────────────────────────────────────────────────────
    ring_cy = y + 130
    _draw_score_ring(draw, W // 2, ring_cy, 100, form_score, font_score, font_small)

    # ── Stats grid (2 rows x 3 cols) ───────────────────────────────────
    grid_top = ring_cy + 140
    pad = 20
    box_w = (W - pad * 4) // 3
    box_h = 90

    stats_row1 = [
        ("Peak Jump", f"{peak_jump:.1f} cm" if peak_jump is not None else "-"),
        ("Symmetry", f"{symmetry * 100:.0f}%" if symmetry is not None else "-"),
        ("XP Earned", f"+{xp}"),
    ]
    stats_row2 = [
        ("Duration", _format_duration(duration)),
        ("Reps", str(reps)),
    ]

    for i, (label, value) in enumerate(stats_row1):
        bx = pad + i * (box_w + pad)
        _draw_stat_box(draw, bx, grid_top, box_w, box_h, label, value, font_val, font_lbl)

    row2_y = grid_top + box_h + pad
    for i, (label, value) in enumerate(stats_row2):
        bx = pad + i * (box_w + pad)
        _draw_stat_box(draw, bx, row2_y, box_w, box_h, label, value, font_val, font_lbl)

    # Quality bar in remaining slot of row 2
    qb_x = pad + 2 * (box_w + pad)
    _draw_quality_bar(draw, qb_x, row2_y, box_w, box_h, quality_dist, font_lbl)

    # ── BPI badge (if present) ──────────────────────────────────────────
    badge_y = row2_y + box_h + pad + 10
    if bpi is not None:
        bpi_text = f"BPI  {bpi:,}"
        _draw_centered_text(draw, bpi_text, badge_y, font_val, fill=ACCENT)
        badge_y += 50

    # ── Bottom section — athlete info ───────────────────────────────────
    footer_y = max(badge_y + 10, H - 180)

    # Thin line
    draw.line((line_margin, footer_y, W - line_margin, footer_y), fill=BORDER, width=1)
    footer_y += 20

    _draw_centered_text(draw, athlete_name, footer_y, font_sport, fill=WHITE)
    footer_y += 36

    tier_line = tier
    if tier:
        tier_line = f"{tier} Tier"
    date_str = datetime.now(timezone.utc).strftime("%d %b %Y")
    meta_line = f"{tier_line}  |  {date_str}" if tier_line else date_str
    _draw_centered_text(draw, meta_line, footer_y, font_lbl, fill=WHITE_DIM)
    footer_y += 36

    _draw_centered_text(draw, "activebharat.in", footer_y, font_footer, fill=ACCENT)

    # ── Encode ──────────────────────────────────────────────────────────
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _build_error_card() -> bytes:
    """Minimal fallback card returned when generation fails."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(28)
    _draw_centered_text(draw, "ActiveBharat", H // 2 - 40, font, fill=ACCENT)
    _draw_centered_text(draw, "Scorecard unavailable", H // 2 + 10, font, fill=WHITE_DIM)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─── FastAPI helper ─────────────────────────────────────────────────────────


def generate_scorecard_response(session_summary: dict, athlete: dict):
    """
    Return a FastAPI ``Response`` containing the scorecard PNG.

    Usage in a route::

        @app.get("/scorecard/{session_id}")
        async def scorecard(session_id: str):
            summary = ...
            athlete = ...
            return generate_scorecard_response(summary, athlete)
    """
    from fastapi.responses import Response  # deferred to avoid import at module level

    png_bytes = generate_scorecard(session_summary, athlete)
    session_id = session_summary.get("session_id", "scorecard")
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="scorecard_{session_id}.png"',
            "Cache-Control": "public, max-age=86400",
        },
    )
