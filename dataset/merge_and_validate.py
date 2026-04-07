#!/usr/bin/env python3
"""
Merge & Validate — Final Dataset Assembly
==========================================
Merges all data sources (real + synthetic), applies feature engineering,
validates, balances, and outputs the final training_data.csv.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
SOURCES_DIR = BASE_DIR / "sources"

# Import feature engineering
sys.path.insert(0, str(BASE_DIR))
from engineer_features import engineer_features

VALID_SPORTS = ["vertical_jump", "squat", "push_up", "pull_up", "snatch", "sprint", "javelin", "cricket_bat"]
VALID_QUALITIES = ["poor", "average", "good", "elite"]
REQUIRED_COLS = [
    "session_id", "athlete_id", "frame_num", "sport",
    "hip_angle_l", "hip_angle_r", "knee_angle_l", "knee_angle_r",
    "shoulder_angle_l", "shoulder_angle_r", "elbow_angle_l", "elbow_angle_r",
    "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
    "trunk_lean", "spine_deviation", "shoulder_hip_sep", "head_forward_pos",
    "com_height_norm", "estimated_jump_height", "limb_symmetry_idx", "form_score",
    "phase_label", "quality_label", "feedback_tag",
]


def load_all_sources():
    """Load all CSV files from the sources directory."""
    all_dfs = []
    source_stats = {}

    csv_files = sorted(glob.glob(str(SOURCES_DIR / "*.csv")))
    print(f"Found {len(csv_files)} source files:")

    for path in csv_files:
        name = os.path.basename(path)
        try:
            df = pd.read_csv(path)
            print(f"  {name}: {len(df)} rows, {len(df.columns)} cols")
            all_dfs.append(df)
            source_stats[name] = len(df)
        except Exception as e:
            print(f"  {name}: FAILED to load — {e}")

    if not all_dfs:
        print("ERROR: No source files found!")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} total rows")
    return combined, source_stats


def validate_and_clean(df):
    """Validate data quality and fix issues."""
    print("\n--- Validation ---")
    initial = len(df)

    # Ensure required columns exist
    for col in REQUIRED_COLS:
        if col not in df.columns:
            print(f"  WARNING: Missing column '{col}' — adding default")
            if col in ("session_id", "athlete_id", "feedback_tag", "phase_label", "quality_label", "sport"):
                df[col] = "unknown"
            else:
                df[col] = 0.0

    # Convert numeric columns
    angle_cols = [
        "hip_angle_l", "hip_angle_r", "knee_angle_l", "knee_angle_r",
        "shoulder_angle_l", "shoulder_angle_r", "elbow_angle_l", "elbow_angle_r",
        "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
    ]
    for col in angle_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["trunk_lean", "spine_deviation", "shoulder_hip_sep", "head_forward_pos",
                "com_height_norm", "estimated_jump_height", "limb_symmetry_idx", "form_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in critical columns
    critical = angle_cols + ["sport", "quality_label"]
    before = len(df)
    df = df.dropna(subset=[c for c in critical if c in df.columns])
    print(f"  Dropped {before - len(df)} rows with NaN in critical columns")

    # Clip angles to [0, 180]
    for col in angle_cols:
        df[col] = df[col].clip(0, 180)

    # Clip trunk_lean to [0, 90]
    df["trunk_lean"] = df["trunk_lean"].clip(0, 90)

    # Clip spine_deviation to [0, 30]
    df["spine_deviation"] = df["spine_deviation"].clip(0, 30)

    # Clip symmetry index to [0, 1]
    df["limb_symmetry_idx"] = df["limb_symmetry_idx"].clip(0, 1)

    # Clip form_score to [0, 100]
    df["form_score"] = df["form_score"].clip(0, 100)

    # Validate sport
    before = len(df)
    df = df[df["sport"].isin(VALID_SPORTS)]
    print(f"  Dropped {before - len(df)} rows with invalid sport")

    # Validate quality_label
    before = len(df)
    df = df[df["quality_label"].isin(VALID_QUALITIES)]
    print(f"  Dropped {before - len(df)} rows with invalid quality_label")

    # Replace NaN in non-critical columns
    df = df.fillna(0)

    # Replace inf values
    df = df.replace([np.inf, -np.inf], 0)

    print(f"  Validation complete: {initial} -> {len(df)} rows ({initial - len(df)} removed)")
    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    before = len(df)
    # Check for near-duplicates on angle columns
    angle_cols = [
        "hip_angle_l", "hip_angle_r", "knee_angle_l", "knee_angle_r",
        "shoulder_angle_l", "shoulder_angle_r", "elbow_angle_l", "elbow_angle_r",
    ]
    df = df.drop_duplicates(subset=angle_cols + ["sport", "quality_label"], keep="first")
    print(f"  Removed {before - len(df)} duplicates")
    return df


def remove_outliers(df):
    """Remove outliers beyond 4 sigma from sport-specific means."""
    angle_cols = [
        "hip_angle_l", "hip_angle_r", "knee_angle_l", "knee_angle_r",
        "shoulder_angle_l", "shoulder_angle_r", "elbow_angle_l", "elbow_angle_r",
        "ankle_dorsiflexion_l", "ankle_dorsiflexion_r", "trunk_lean",
    ]

    before = len(df)
    mask = pd.Series(True, index=df.index)

    for sport in df["sport"].unique():
        sport_mask = df["sport"] == sport
        sport_df = df[sport_mask]

        for col in angle_cols:
            mean = sport_df[col].mean()
            std = sport_df[col].std()
            if std > 0:
                outlier = (df[col] - mean).abs() > 4 * std
                mask = mask & ~(sport_mask & outlier)

    df = df[mask]
    print(f"  Removed {before - len(df)} outliers (>4σ)")
    return df


def balance_classes(df, target_per_class=None):
    """Balance classes so each sport x quality tier is within 30% of each other."""
    print("\n--- Balancing Classes ---")

    # Find the target count per sport-quality combination
    counts = df.groupby(["sport", "quality_label"]).size()
    print(f"  Current distribution range: {counts.min()} to {counts.max()}")

    if target_per_class is None:
        # Target: median count, ensuring minimum viable
        target_per_class = int(counts.median())
        # Ensure at least 1000 per cell
        target_per_class = max(target_per_class, 1000)

    print(f"  Target per sport x quality: {target_per_class}")

    balanced_parts = []
    for sport in VALID_SPORTS:
        for quality in VALID_QUALITIES:
            subset = df[(df["sport"] == sport) & (df["quality_label"] == quality)]
            n = len(subset)

            if n >= target_per_class:
                # Downsample
                balanced_parts.append(subset.sample(n=target_per_class, random_state=42))
            elif n > 0:
                # Upsample with noise
                sampled = subset.sample(n=target_per_class, replace=True, random_state=42)
                # Add small noise to upsampled duplicates to prevent exact duplicates
                angle_cols = [c for c in sampled.columns if "angle" in c or c in ("trunk_lean", "ankle_dorsiflexion_l", "ankle_dorsiflexion_r")]
                noise = np.random.normal(0, 0.5, size=(len(sampled), len(angle_cols)))
                sampled[angle_cols] = sampled[angle_cols].values + noise
                balanced_parts.append(sampled)
            else:
                print(f"  WARNING: No data for {sport}/{quality}")

    df = pd.concat(balanced_parts, ignore_index=True)
    print(f"  Balanced to {len(df)} rows")

    # Verify balance
    counts = df.groupby(["sport", "quality_label"]).size()
    print(f"  Final range: {counts.min()} to {counts.max()}")
    return df


def assign_missing_metadata(df):
    """Fill in session_id, athlete_id, frame_num where missing."""
    # Fix unknown/missing session IDs
    mask = (df["session_id"] == "unknown") | df["session_id"].isna()
    if mask.sum() > 0:
        counter = df["session_id"].str.extract(r'(\d+)', expand=False).dropna().astype(int).max()
        counter = counter + 1 if not pd.isna(counter) else 100000

        for idx in df[mask].index:
            sport_code = df.loc[idx, "sport"][:3].upper()
            qual_code = df.loc[idx, "quality_label"][:3].upper()
            df.loc[idx, "session_id"] = f"SES_{sport_code}_{qual_code}_{int(counter):05d}"
            counter += 1

    # Fix unknown/missing athlete IDs
    mask = (df["athlete_id"] == "unknown") | df["athlete_id"].isna()
    if mask.sum() > 0:
        athletes = [f"athlete_{i:04d}" for i in range(1, 301)]
        df.loc[mask, "athlete_id"] = np.random.choice(athletes, mask.sum())

    # Fix frame_num
    mask = (df["frame_num"] == 0) | df["frame_num"].isna()
    if mask.sum() > 0:
        df.loc[mask, "frame_num"] = np.random.randint(1, 30, mask.sum())

    return df


def main():
    print("=" * 60)
    print("Merge & Validate — Final Dataset Assembly")
    print("=" * 60)

    # Step 1: Load all sources
    df, source_stats = load_all_sources()

    # Step 2: Validate and clean
    df = validate_and_clean(df)

    # Step 3: Apply feature engineering
    print("\n--- Feature Engineering ---")
    df = engineer_features(df)
    print(f"  Enriched to {len(df.columns)} columns")

    # Step 4: Remove duplicates
    print("\n--- Deduplication ---")
    df = remove_duplicates(df)

    # Step 5: Remove outliers
    print("\n--- Outlier Removal ---")
    df = remove_outliers(df)

    # Step 6: Balance classes
    # Target: ~1375 per sport x quality = 44000 total (8 sports x 4 qualities x 1375)
    df = balance_classes(df, target_per_class=1562)  # 8*4*1562 = 49,984

    # Step 7: Assign missing metadata
    print("\n--- Metadata ---")
    df = assign_missing_metadata(df)

    # Step 8: Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Step 9: Ensure column order
    # Put required columns first, then engineered features
    required_first = [c for c in REQUIRED_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLS]
    df = df[required_first + sorted(extra)]

    # Step 10: Save
    output_path = BASE_DIR / "training_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\n*** Final dataset saved to {output_path} ***")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")

    # Step 11: Save stats
    stats = {
        "total_samples": len(df),
        "total_features": len(df.columns),
        "source_files": source_stats,
        "per_sport": df["sport"].value_counts().to_dict(),
        "per_quality": df["quality_label"].value_counts().to_dict(),
        "per_sport_quality": df.groupby(["sport", "quality_label"]).size().unstack(fill_value=0).to_dict(),
        "feature_stats": {},
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        stats["feature_stats"][col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
        }

    stats_path = BASE_DIR / "sample_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {stats_path}")

    # Print summary
    print("\n--- Final Distribution ---")
    print(df.groupby(["sport", "quality_label"]).size().unstack(fill_value=0).to_string())
    print(f"\nForm score by quality:")
    print(df.groupby("quality_label")["form_score"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
