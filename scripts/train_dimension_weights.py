"""Train dimension-level weights using fund_score / growth_score / tech_score.

This is "Plan A": train on the same signals that the screener produces,
so the learned weights can be plugged directly into combine_scores().

Usage:
    poetry run python scripts/train_dimension_weights.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

INPUT_PATH = Path("training_data.parquet")
OUTPUT_PATH = Path("weights/dimension_weights.json")

DIM_COLS = ["fund_score", "growth_score", "tech_score"]


def main():
    if not INPUT_PATH.exists():
        print(f"Training data not found: {INPUT_PATH}")
        print("Run: poetry run python scripts/collect_training_data.py")
        return

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

    # Drop rows with missing dimension scores or forward returns
    train_df = df[DIM_COLS + ["forward_return"]].dropna()
    print(f"Usable records: {len(train_df)}")

    X = train_df[DIM_COLS].values
    y = train_df["forward_return"].values

    # Ridge regression (L2 regularization to avoid extreme weights)
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))

    # Raw coefficients (can be negative)
    raw_weights = dict(zip(DIM_COLS, model.coef_))

    # Clip to non-negative and normalize (so weights sum to 1)
    clipped = {k: max(v, 0.0) for k, v in raw_weights.items()}
    total = sum(clipped.values())
    normalized = {k: round(v / total, 4) if total > 0 else 0.0 for k, v in clipped.items()}

    # Also compute correlation of each dimension with returns
    correlations = {}
    for col in DIM_COLS:
        correlations[col] = float(train_df[col].corr(train_df["forward_return"]))

    result = {
        "rmse": rmse,
        "raw_coefficients": {k: round(v, 6) for k, v in raw_weights.items()},
        "normalized_weights": normalized,
        "correlations": {k: round(v, 6) for k, v in correlations.items()},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"\nRMSE: {rmse:.6f}")
    print("\nRaw coefficients (can be negative):")
    for k, v in raw_weights.items():
        print(f"  {k}: {v:.6f}")
    print("\nNormalized weights (non-negative, sum to 1):")
    for k, v in normalized.items():
        print(f"  {k}: {v:.4f} ({v*100:.2f}%)")
    print("\nCorrelation with forward_return:")
    for k, v in correlations.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
