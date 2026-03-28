"""
Personal Health — Real Session Data Exporter
=============================================
Exports completed session frames to a labeled CSV for model retraining.

Usage:
  # After a huddle where athletes had "good" overall form:
  python export_real_data.py --quality good --output dataset/real_batch1.csv

  # Export only sessions from the last 7 days:
  python export_real_data.py --quality average --days 7 --output dataset/real_average.csv

  # Dry run — shows what would be exported without writing:
  python export_real_data.py --quality elite --dry-run

Arguments:
  --quality     Session-level label to apply: elite / good / average / poor
  --output      Output CSV path (default: dataset/real_export.csv)
  --min-frames  Minimum frames per session to include (default: 30)
  --days        Only sessions from last N days (default: all)
  --dry-run     Preview what would be exported without writing

After exporting, mix with synthetic data and retrain:
  python mix_datasets.py --real dataset/real_batch1.csv \
                         --synthetic dataset/training_data.csv \
                         --output dataset/mixed_training.csv
  python model_trainer.py --dataset dataset/mixed_training.csv
"""

import json
import csv
import argparse
from pathlib import Path
from datetime import datetime, timedelta


DB_PATH = Path(__file__).parent / "db"
DATASET_PATH = Path(__file__).parent / "dataset"

FEATURE_FIELDS = [
    "hip_angle_l", "hip_angle_r",
    "knee_angle_l", "knee_angle_r",
    "shoulder_angle_l", "shoulder_angle_r",
    "elbow_angle_l", "elbow_angle_r",
    "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
    "trunk_lean", "spine_deviation", "shoulder_hip_sep", "head_forward_pos",
    "com_height_norm", "estimated_jump_height", "limb_symmetry_idx",
    "form_score",
]


def export(quality_label: str, output_path: str, min_frames: int = 30,
           days: int = None, dry_run: bool = False):
    sessions_file = DB_PATH / "sessions.json"
    if not sessions_file.exists():
        print("[EXPORT] db/sessions.json not found. Run the backend first.")
        return []

    with open(sessions_file, encoding="utf-8") as f:
        sessions = json.load(f)

    cutoff = None
    if days:
        cutoff = datetime.utcnow() - timedelta(days=days)

    rows = []
    skipped_no_frames = 0
    skipped_not_complete = 0
    skipped_old = 0
    skipped_no_pose = 0

    for sid, session in sessions.items():
        # Only completed sessions
        if session.get("status") != "completed":
            skipped_not_complete += 1
            continue

        # Date filter
        if cutoff:
            started_at_str = session.get("started_at")
            if started_at_str:
                try:
                    started_at = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
                    started_at = started_at.replace(tzinfo=None)
                    if started_at < cutoff:
                        skipped_old += 1
                        continue
                except Exception:
                    pass

        frames = session.get("frames", [])
        if len(frames) < min_frames:
            skipped_no_frames += 1
            continue

        sport = session.get("sport", "unknown")
        athlete_id = session.get("athlete_id", "unknown")

        for frame in frames:
            # Skip frames with no pose detected
            if not frame.get("pose_detected", True):
                skipped_no_pose += 1
                continue

            # Skip frames with zero angles (bad capture)
            if frame.get("knee_angle_l", 0) == 0 and frame.get("hip_angle_l", 0) == 0:
                skipped_no_pose += 1
                continue

            row = {
                "session_id":   sid,
                "athlete_id":   athlete_id,
                "frame_num":    frame.get("frame_num", 0),
                "sport":        sport,
            }
            for field in FEATURE_FIELDS:
                row[field] = frame.get(field, 0.0)

            row["phase_label"]   = frame.get("phase", "setup")
            row["quality_label"] = quality_label
            row["feedback_tag"]  = frame.get("primary_feedback", "")
            row["source"]        = "real"

            rows.append(row)

    # Stats
    print(f"\n[EXPORT] Scan complete:")
    print(f"  Total sessions:          {len(sessions)}")
    print(f"  Skipped (not complete):  {skipped_not_complete}")
    print(f"  Skipped (too old):       {skipped_old}")
    print(f"  Skipped (< {min_frames} frames):   {skipped_no_frames}")
    print(f"  Frames with no pose:     {skipped_no_pose}")
    print(f"  Eligible frames:         {len(rows)}")

    if not rows:
        print("\n[EXPORT] Nothing to export. Tips:")
        print("  - Make sure sessions have status='completed'")
        print("  - Run seed_sessions.py to create sample sessions")
        print("  - Try --min-frames 10 to lower the threshold")
        return rows

    # Sport distribution
    sport_counts = {}
    for row in rows:
        sport_counts[row["sport"]] = sport_counts.get(row["sport"], 0) + 1
    print(f"\n[EXPORT] Sport distribution:")
    for sport, count in sorted(sport_counts.items(), key=lambda x: -x[1]):
        print(f"  {sport:<20} {count:>5} frames")

    if dry_run:
        print(f"\n[DRY RUN] Would write {len(rows)} rows to {output_path}")
        print("[DRY RUN] No file written.")
        return rows

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[EXPORT] ✓ {len(rows)} rows → {out}")
    print(f"[EXPORT] Quality label applied: '{quality_label}'")
    print(f"\nNext steps:")
    print(f"  python mix_datasets.py --real {out} --synthetic dataset/training_data.csv \\")
    print(f"                         --output dataset/mixed_training.csv --real-weight 4")
    print(f"  python model_trainer.py --dataset dataset/mixed_training.csv")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export real session data for model training")
    parser.add_argument("--quality", required=True, choices=["elite", "good", "average", "poor"],
                        help="Form quality label to apply to all exported frames")
    parser.add_argument("--output", default="dataset/real_export.csv",
                        help="Output CSV file path")
    parser.add_argument("--min-frames", type=int, default=30,
                        help="Minimum frames per session (skip shorter sessions)")
    parser.add_argument("--days", type=int, default=None,
                        help="Only include sessions from the last N days")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be exported without writing")
    args = parser.parse_args()

    export(
        quality_label=args.quality,
        output_path=args.output,
        min_frames=args.min_frames,
        days=args.days,
        dry_run=args.dry_run,
    )
