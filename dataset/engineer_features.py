#!/usr/bin/env python3
"""
Feature Engineering for Sports Biomechanics Dataset
====================================================
Computes derived features from base joint angle columns.
Importable as a module or runnable standalone.

Base features expected:
  hip_angle_l, hip_angle_r, knee_angle_l, knee_angle_r,
  shoulder_angle_l, shoulder_angle_r, elbow_angle_l, elbow_angle_r,
  ankle_dorsiflexion_l, ankle_dorsiflexion_r,
  trunk_lean, spine_deviation, shoulder_hip_sep, head_forward_pos,
  com_height_norm, estimated_jump_height
"""

import pandas as pd
import numpy as np
import sys
import os


def compute_bilateral_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average of left/right joint angles."""
    df = df.copy()
    df["hip_angle_avg"] = (df["hip_angle_l"] + df["hip_angle_r"]) / 2
    df["knee_angle_avg"] = (df["knee_angle_l"] + df["knee_angle_r"]) / 2
    df["shoulder_angle_avg"] = (df["shoulder_angle_l"] + df["shoulder_angle_r"]) / 2
    df["elbow_angle_avg"] = (df["elbow_angle_l"] + df["elbow_angle_r"]) / 2
    df["ankle_avg"] = (df["ankle_dorsiflexion_l"] + df["ankle_dorsiflexion_r"]) / 2
    return df


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute biomechanically meaningful joint angle ratios."""
    df = df.copy()
    # Hip-knee ratio: indicator of hip-knee coordination
    knee_avg = (df["knee_angle_l"] + df["knee_angle_r"]) / 2
    hip_avg = (df["hip_angle_l"] + df["hip_angle_r"]) / 2
    df["hip_knee_ratio"] = hip_avg / knee_avg.clip(lower=1.0)

    # Upper-lower body ratio: shoulder+elbow avg / hip+knee avg
    upper_avg = ((df["shoulder_angle_l"] + df["shoulder_angle_r"]) / 2 +
                 (df["elbow_angle_l"] + df["elbow_angle_r"]) / 2) / 2
    lower_avg = (hip_avg + knee_avg) / 2
    df["upper_lower_ratio"] = upper_avg / lower_avg.clip(lower=1.0)

    # Ankle-knee ratio: relevant for squat depth assessment
    ankle_avg = (df["ankle_dorsiflexion_l"] + df["ankle_dorsiflexion_r"]) / 2
    df["ankle_knee_ratio"] = ankle_avg / knee_avg.clip(lower=1.0)

    return df


def compute_symmetry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute bilateral symmetry indices per joint pair and aggregate."""
    df = df.copy()
    joint_pairs = [
        ("hip_angle_l", "hip_angle_r", "hip_symmetry"),
        ("knee_angle_l", "knee_angle_r", "knee_symmetry"),
        ("shoulder_angle_l", "shoulder_angle_r", "shoulder_symmetry"),
        ("elbow_angle_l", "elbow_angle_r", "elbow_symmetry"),
        ("ankle_dorsiflexion_l", "ankle_dorsiflexion_r", "ankle_symmetry"),
    ]

    sym_cols = []
    for left, right, name in joint_pairs:
        # LSI = min(L,R) / max(L,R) — standard Limb Symmetry Index
        max_val = np.maximum(df[left], df[right]).clip(lower=1.0)
        min_val = np.minimum(df[left], df[right])
        df[name] = min_val / max_val
        sym_cols.append(name)

    # Aggregate symmetry (mean of all joint LSIs)
    df["aggregate_symmetry"] = df[sym_cols].mean(axis=1)

    # Bilateral deviation: max absolute L-R difference across all pairs
    deviations = []
    for left, right, _ in joint_pairs:
        deviations.append(np.abs(df[left] - df[right]))
    df["bilateral_deviation"] = np.column_stack(deviations).max(axis=1)

    # Max single-joint asymmetry (worst asymmetry)
    df["worst_asymmetry"] = 1.0 - df[sym_cols].min(axis=1)

    return df


def compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite biomechanical scores."""
    df = df.copy()

    # Extension index: mean of all joint angles normalized to [0,1]
    angle_cols = ["hip_angle_l", "hip_angle_r", "knee_angle_l", "knee_angle_r",
                  "shoulder_angle_l", "shoulder_angle_r", "elbow_angle_l", "elbow_angle_r",
                  "ankle_dorsiflexion_l", "ankle_dorsiflexion_r"]
    df["extension_index"] = df[angle_cols].mean(axis=1) / 180.0

    # Trunk alignment score: composite of trunk lean and spine deviation
    # Lower is better; scale to 0-1 where 1 = perfect alignment
    df["trunk_alignment_score"] = 1.0 - np.clip(
        (df["trunk_lean"] + df["spine_deviation"] * 2) / 60.0, 0, 1
    )

    # Flexion depth score: how deep the athlete goes (for squat, jump, etc.)
    # Based on knee and hip angles — lower angles = deeper
    knee_avg = (df["knee_angle_l"] + df["knee_angle_r"]) / 2
    hip_avg = (df["hip_angle_l"] + df["hip_angle_r"]) / 2
    df["flexion_depth"] = 1.0 - np.clip((knee_avg + hip_avg) / 360.0, 0, 1)

    # Upper body engagement: for push_up, pull_up, snatch, javelin
    shoulder_avg = (df["shoulder_angle_l"] + df["shoulder_angle_r"]) / 2
    elbow_avg = (df["elbow_angle_l"] + df["elbow_angle_r"]) / 2
    df["upper_body_engagement"] = (shoulder_avg + elbow_avg) / 360.0

    # Postural stability index: combination of trunk lean, spine deviation, head position
    df["postural_stability"] = 1.0 - np.clip(
        (np.abs(df["trunk_lean"]) / 45.0 +
         np.abs(df["spine_deviation"]) / 15.0 +
         np.abs(df["head_forward_pos"]) / 10.0) / 3.0,
        0, 1
    )

    return df


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sport-specific interaction features."""
    df = df.copy()

    knee_avg = (df["knee_angle_l"] + df["knee_angle_r"]) / 2
    hip_avg = (df["hip_angle_l"] + df["hip_angle_r"]) / 2
    shoulder_avg = (df["shoulder_angle_l"] + df["shoulder_angle_r"]) / 2
    elbow_avg = (df["elbow_angle_l"] + df["elbow_angle_r"]) / 2

    # Kinetic chain index: product of key joint angles normalized
    # Captures the coordination chain from ankle to shoulder
    df["kinetic_chain_idx"] = (
        knee_avg * hip_avg * df["trunk_lean"].clip(lower=0.1)
    ) / (180 * 180 * 45)

    # Lower body power proxy: deeper flexion + higher COM = more power potential
    df["power_proxy"] = df["com_height_norm"] * (1.0 - knee_avg / 180.0)

    # Arm-trunk coordination: for throwing sports
    df["arm_trunk_coord"] = np.abs(shoulder_avg - df["trunk_lean"]) / 180.0

    # Hip-shoulder separation normalized (for rotational sports)
    df["hip_shoulder_sep_norm"] = df["shoulder_hip_sep"] / 90.0

    # Total body flexion: sum of all major joint flexions
    df["total_flexion"] = (knee_avg + hip_avg + shoulder_avg + elbow_avg) / 4.0

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point: apply all feature engineering to a DataFrame.
    Input must have the base joint angle columns.
    Returns DataFrame with all original + derived columns.
    """
    # Ensure numeric types
    numeric_cols = [
        "hip_angle_l", "hip_angle_r", "knee_angle_l", "knee_angle_r",
        "shoulder_angle_l", "shoulder_angle_r", "elbow_angle_l", "elbow_angle_r",
        "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
        "trunk_lean", "spine_deviation", "shoulder_hip_sep", "head_forward_pos",
        "com_height_norm", "estimated_jump_height"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = compute_bilateral_averages(df)
    df = compute_ratios(df)
    df = compute_symmetry_features(df)
    df = compute_composite_scores(df)
    df = compute_interaction_features(df)

    return df


if __name__ == "__main__":
    # Standalone usage: python engineer_features.py input.csv output.csv
    if len(sys.argv) < 2:
        # Default: process the training data
        input_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
        output_path = os.path.join(os.path.dirname(__file__), "training_data_enriched.csv")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".csv", "_enriched.csv")

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    df = engineer_features(df)
    print(f"  Engineered to {len(df.columns)} columns")

    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    # Print sample of new features
    new_cols = [c for c in df.columns if c not in pd.read_csv(input_path, nrows=0).columns]
    print(f"\nNew features ({len(new_cols)}):")
    for col in new_cols:
        print(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, "
              f"min={df[col].min():.3f}, max={df[col].max():.3f}")
