"""
Personal Health — Dataset Mixer
================================
Combines real session exports with synthetic training data for model retraining.
Applies weighting so real data dominates while synthetic fills gaps.

Usage:
  python mix_datasets.py \
    --real dataset/real_batch1.csv \
    --synthetic dataset/training_data.csv \
    --output dataset/mixed_training.csv \
    --real-weight 4

--real-weight 4 means each real frame is duplicated 4x in the final dataset,
giving it 4x the gradient influence vs synthetic data during training.

If you have multiple real batches:
  python mix_datasets.py \
    --real dataset/real_batch1.csv dataset/real_batch2.csv \
    --synthetic dataset/training_data.csv \
    --output dataset/mixed_training.csv
"""

import argparse
import csv
import random
from pathlib import Path

SEED = 42
random.seed(SEED)


def load_csv(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mix(real_paths: list, synthetic_path: str, output_path: str, real_weight: int = 4, max_synthetic: int = 2000):

    # Load real data
    real_rows = []
    for rp in real_paths:
        batch = load_csv(rp)
        real_rows.extend(batch)
        print(f"[MIX] Loaded {len(batch)} real rows from {rp}")

    # Load synthetic data (cap to avoid overwhelming real data)
    synthetic_rows = load_csv(synthetic_path)
    print(f"[MIX] Loaded {len(synthetic_rows)} synthetic rows from {synthetic_path}")

    if len(synthetic_rows) > max_synthetic:
        synthetic_rows = random.sample(synthetic_rows, max_synthetic)
        print(f"[MIX] Synthetic capped to {max_synthetic} rows")

    # Upweight real data by duplication
    weighted_real = real_rows * real_weight
    print(f"[MIX] Real rows after {real_weight}x weighting: {len(weighted_real)}")

    # Combine and shuffle
    combined = weighted_real + synthetic_rows
    random.shuffle(combined)
    print(f"[MIX] Total mixed dataset: {len(combined)} rows")

    # Quality distribution
    q_counts = {}
    for row in combined:
        q = row.get("quality_label", "unknown")
        q_counts[q] = q_counts.get(q, 0) + 1
    print("\n[MIX] Quality distribution in mixed dataset:")
    for q, count in sorted(q_counts.items()):
        pct = count / len(combined) * 100
        print(f"  {q:<10} {count:>5} ({pct:.1f}%)")

    # Write
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if combined:
        fieldnames = list(combined[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined)
        print(f"\n[MIX] ✓ Mixed dataset written to {out}")
        print(f"[MIX] Next: python model_trainer.py --dataset {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", nargs="+", required=True, help="Real data CSV file(s)")
    parser.add_argument("--synthetic", required=True, help="Synthetic data CSV")
    parser.add_argument("--output", required=True, help="Output mixed CSV")
    parser.add_argument("--real-weight", type=int, default=4, help="How many times to duplicate real rows (default 4)")
    parser.add_argument(
        "--max-synthetic", type=int, default=2000, help="Cap on synthetic rows to include (default 2000)"
    )
    args = parser.parse_args()

    mix(args.real, args.synthetic, args.output, args.real_weight, args.max_synthetic)
