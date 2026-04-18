#!/usr/bin/env python3
"""
Personal Health — Automated Retrain Orchestrator.

Single-command cycle that:
  1. Checks if enough real session data exists to justify a retrain
  2. Exports real frames (with quality gate)
  3. Mixes with synthetic data
  4. Trains a new model
  5. Evaluates accuracy
  6. Compares to the current model
  7. Deploys if the new model is better (or within tolerance)

Usage:
  python pipeline/retrain.py                    # full auto
  python pipeline/retrain.py --dry-run          # preview only
  python pipeline/retrain.py --force            # retrain even if not enough data
  python pipeline/retrain.py --min-frames 200   # lower the threshold

This script is designed to be run as a cron job or triggered manually
after a huddle or batch of sessions.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PIPELINE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = PIPELINE_DIR / "dataset"
DB_DIR = BASE_DIR / "db"

# Thresholds
DEFAULT_MIN_FRAMES = 500
ACCURACY_TOLERANCE = 0.02  # deploy new model if within 2% of current


def _log(msg: str) -> None:
    print(f"[RETRAIN] {msg}")


def _count_real_frames() -> int:
    """Count total completed-session frames in the database."""
    sessions_file = DB_DIR / "sessions.json"
    if not sessions_file.exists():
        return 0
    with open(sessions_file) as f:
        sessions = json.load(f)
    total = 0
    for s in sessions.values():
        if s.get("status") != "completed":
            continue
        frames = s.get("frames", []) or []
        total += len(frames)
    return total


def _current_model_accuracy() -> float:
    """Read the current model's test accuracy."""
    metrics_path = MODELS_DIR / "training_metrics.json"
    if not metrics_path.exists():
        return 0.0
    with open(metrics_path) as f:
        d = json.load(f)
    return float(d.get("test_accuracy", 0))


def _export_real_data(min_frames_per_session: int = 10) -> Path | None:
    """Export real session data using the existing export pipeline + quality gate."""
    _log("Exporting real session frames with quality gate...")

    sessions_file = DB_DIR / "sessions.json"
    if not sessions_file.exists():
        _log("No sessions.json found")
        return None

    with open(sessions_file) as f:
        sessions = json.load(f)

    # Import quality gate
    sys.path.insert(0, str(BASE_DIR))
    from services.frame_quality import filter_training_frames

    import csv

    output_path = DATASET_DIR / "real_export_auto.csv"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    FEATURE_FIELDS = [
        "hip_angle_l",
        "hip_angle_r",
        "knee_angle_l",
        "knee_angle_r",
        "shoulder_angle_l",
        "shoulder_angle_r",
        "elbow_angle_l",
        "elbow_angle_r",
        "ankle_dorsiflexion_l",
        "ankle_dorsiflexion_r",
        "trunk_lean",
        "spine_deviation",
        "shoulder_hip_sep",
        "head_forward_pos",
        "com_height_norm",
        "estimated_jump_height",
        "limb_symmetry_idx",
        "form_score",
    ]

    fieldnames = (
        ["session_id", "athlete_id", "frame_num", "sport"]
        + FEATURE_FIELDS
        + [
            "phase_label",
            "quality_label",
            "feedback_tag",
            "source",
            "_quality_score",
        ]
    )

    total_written = 0
    total_rejected = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for sid, session in sessions.items():
            if session.get("status") != "completed":
                continue
            frames = session.get("frames", []) or []
            if len(frames) < min_frames_per_session:
                continue

            sport = session.get("sport", "unknown")
            accepted, stats = filter_training_frames(frames, sport)
            total_rejected += stats["rejected"]

            # Infer quality label from avg form score
            avg_score = float(
                session.get("avg_form_score", 0) or session.get("summary", {}).get("avg_form_score", 0) or 0
            )
            if avg_score >= 85:
                quality_label = "elite"
            elif avg_score >= 70:
                quality_label = "good"
            elif avg_score >= 50:
                quality_label = "average"
            else:
                quality_label = "poor"

            for frame in accepted:
                row = {
                    "session_id": sid,
                    "athlete_id": session.get("athlete_id", ""),
                    "frame_num": frame.get("frame_num", 0),
                    "sport": sport,
                    "phase_label": frame.get("phase", "setup"),
                    "quality_label": quality_label,
                    "feedback_tag": frame.get("primary_feedback", ""),
                    "source": "real",
                    "_quality_score": frame.get("_quality_score", 0),
                }
                for field in FEATURE_FIELDS:
                    row[field] = frame.get(field, 0.0)
                writer.writerow(row)
                total_written += 1

    _log(f"Exported {total_written} quality-gated frames ({total_rejected} rejected)")
    return output_path if total_written > 0 else None


def _mix_data(real_path: Path) -> Path | None:
    """Mix real export with synthetic data."""
    _log("Mixing real + synthetic data...")

    synthetic_path = DATASET_DIR / "training_data.csv"
    if not synthetic_path.exists():
        _log("Synthetic training_data.csv not found — run generate_dataset.py first")
        return None

    output_path = DATASET_DIR / "mixed_retrain.csv"

    sys.path.insert(0, str(PIPELINE_DIR))
    from mix_datasets import mix

    mix(
        real_paths=[str(real_path)],
        synthetic_path=str(synthetic_path),
        output_path=str(output_path),
        real_weight=4,
        max_synthetic=2000,
    )
    return output_path if output_path.exists() else None


def _train(dataset_path: Path) -> dict | None:
    """Train a new model on the mixed dataset."""
    _log("Training new model...")

    # Backup current model
    backup_dir = MODELS_DIR / "backup"
    backup_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for f in ["pose_classifier.h5", "pose_classifier.tflite", "norm_params.json", "training_metrics.json"]:
        src = MODELS_DIR / f
        if src.exists():
            shutil.copy2(src, backup_dir / f"{ts}_{f}")

    sys.path.insert(0, str(PIPELINE_DIR))

    # Override dataset path in model_trainer
    import pipeline.model_trainer as trainer

    trainer.DATASET_PATH = dataset_path

    X, y, rows = trainer.load_data()
    if X is None:
        _log("No data loaded — training aborted")
        return None

    result = trainer.train_model(X, y)
    if result is None:
        _log("Training failed")
        return None

    # Read new metrics
    metrics_path = MODELS_DIR / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def _register_version(metrics: dict, real_frames: int) -> None:
    """Register the new model version in the registry."""
    sys.path.insert(0, str(BASE_DIR))
    from services.model_registry import ModelVersion, register_model

    # Determine next version number
    from services.model_registry import get_registry

    reg = get_registry()
    versions = reg.get("versions", [])
    next_num = len(versions) + 1
    version_id = f"v{next_num}"

    register_model(
        ModelVersion(
            version=version_id,
            trained_at=datetime.now(timezone.utc).isoformat(),
            data_source="mixed",
            accuracy=metrics.get("test_accuracy", 0),
            epochs=metrics.get("epochs_trained", 0),
            real_frames=real_frames,
            synthetic_frames=2000,
            notes=f"Auto-retrain with {real_frames} real frames",
        )
    )


def retrain(
    min_frames: int = DEFAULT_MIN_FRAMES,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Full retrain cycle. Returns a status dict.
    """
    _log("=" * 60)
    _log("RETRAIN CYCLE START")
    _log("=" * 60)

    # Step 1: Check data availability
    real_count = _count_real_frames()
    current_acc = _current_model_accuracy()
    _log(f"Real frames available: {real_count}")
    _log(f"Current model accuracy: {current_acc:.1%}")
    _log(f"Retrain threshold: {min_frames} frames")

    if real_count < min_frames and not force:
        _log(f"Not enough data ({real_count} < {min_frames}). Use --force to override.")
        return {
            "status": "skipped",
            "reason": f"insufficient data ({real_count} < {min_frames})",
            "real_frames": real_count,
            "current_accuracy": current_acc,
        }

    if dry_run:
        _log("[DRY RUN] Would export, mix, train, and evaluate.")
        return {
            "status": "dry_run",
            "real_frames": real_count,
            "current_accuracy": current_acc,
        }

    # Step 2: Export real data with quality gate
    real_path = _export_real_data()
    if not real_path:
        return {"status": "failed", "reason": "no quality-gated frames exported"}

    # Step 3: Mix with synthetic
    mixed_path = _mix_data(real_path)
    if not mixed_path:
        return {"status": "failed", "reason": "mixing failed"}

    # Step 4: Train
    metrics = _train(mixed_path)
    if not metrics:
        return {"status": "failed", "reason": "training failed"}

    new_acc = metrics.get("test_accuracy", 0)
    _log(f"New model accuracy: {new_acc:.1%}")
    _log(f"Current model accuracy: {current_acc:.1%}")

    # Step 5: Compare and deploy
    if new_acc >= current_acc - ACCURACY_TOLERANCE:
        _log(f"New model accepted (acc {new_acc:.1%} >= {current_acc:.1%} - {ACCURACY_TOLERANCE:.0%})")
        _register_version(metrics, real_count)

        # Reload model in the running server
        try:
            from services.model_registry import reload_model

            reload_model()
            _log("Model hot-reloaded in running server")
        except Exception:
            _log("Could not hot-reload — restart server to use new model")

        return {
            "status": "deployed",
            "previous_accuracy": current_acc,
            "new_accuracy": new_acc,
            "real_frames": real_count,
            "improvement": round(new_acc - current_acc, 4),
        }
    else:
        _log(f"New model rejected (acc {new_acc:.1%} < {current_acc:.1%} - {ACCURACY_TOLERANCE:.0%})")
        # Restore backup would go here — for now the backup is in models/backup/
        return {
            "status": "rejected",
            "reason": f"accuracy regression ({new_acc:.1%} < {current_acc:.1%})",
            "previous_accuracy": current_acc,
            "new_accuracy": new_acc,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated model retrain cycle")
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--force", action="store_true", help="Retrain even with insufficient data")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't train")
    args = parser.parse_args()

    result = retrain(min_frames=args.min_frames, force=args.force, dry_run=args.dry_run)
    print(f"\n[RETRAIN] Result: {json.dumps(result, indent=2)}")
