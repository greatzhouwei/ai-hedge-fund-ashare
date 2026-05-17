"""Train LightGBM + Ridge to optimize screener weights.

Usage:
    poetry run python scripts/train_weights.py

Outputs:
    - weights/rolling_weights.csv   monthly rolling weights
    - weights/final_weights.json    recommended final weights
    - weights/feature_importance.csv  LightGBM importances
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

INPUT_PATH = Path("training_data.parquet")
OUTPUT_DIR = Path("weights")

# All raw sub-item features (no pre-computed dimension scores)
FEATURE_COLS = [
    # Fundamentals
    "fund_roe",
    "fund_net_margin",
    "fund_op_margin",
    "fund_curr_ratio",
    "fund_de_ratio",
    "fund_fcf_ps",
    "fund_eps",
    "fund_pe",
    "fund_pb",
    "fund_ps",
    # Growth
    "growth_revenue_growth",
    "growth_revenue_trend",
    "growth_eps_growth",
    "growth_eps_trend",
    "growth_fcf_growth",
    "growth_fcf_trend",
    "growth_bv_growth",
    "growth_peg",
    "growth_ps",
    "growth_gross_margin",
    "growth_gm_trend",
    "growth_operating_margin",
    "growth_om_trend",
    "growth_net_margin",
    "growth_nm_trend",
    # Technical
    "tech_adx",
    "tech_tf_bullish",
    "tech_rsi14",
    "tech_price_vs_bb",
    "tech_mr_bullish",
    "tech_mom_3m",
    "tech_mom_6m",
    "tech_mom_bullish",
    "tech_vol_regime",
    "tech_vol_bullish",
]

DIMENSION_MAP = {
    "fund_roe": "fundamentals",
    "fund_net_margin": "fundamentals",
    "fund_op_margin": "fundamentals",
    "fund_curr_ratio": "fundamentals",
    "fund_de_ratio": "fundamentals",
    "fund_fcf_ps": "fundamentals",
    "fund_eps": "fundamentals",
    "fund_pe": "fundamentals",
    "fund_pb": "fundamentals",
    "fund_ps": "fundamentals",
    "growth_revenue_growth": "growth",
    "growth_revenue_trend": "growth",
    "growth_eps_growth": "growth",
    "growth_eps_trend": "growth",
    "growth_fcf_growth": "growth",
    "growth_fcf_trend": "growth",
    "growth_bv_growth": "growth",
    "growth_peg": "growth",
    "growth_ps": "growth",
    "growth_gross_margin": "growth",
    "growth_gm_trend": "growth",
    "growth_operating_margin": "growth",
    "growth_om_trend": "growth",
    "growth_net_margin": "growth",
    "growth_nm_trend": "growth",
    "tech_adx": "technical",
    "tech_tf_bullish": "technical",
    "tech_rsi14": "technical",
    "tech_price_vs_bb": "technical",
    "tech_mr_bullish": "technical",
    "tech_mom_3m": "technical",
    "tech_mom_6m": "technical",
    "tech_mom_bullish": "technical",
    "tech_vol_regime": "technical",
    "tech_vol_bullish": "technical",
}


def _ensure_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        print("lightgbm not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
        import lightgbm as lgb
        return lgb


def fill_na(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill missing values with median per feature."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    return df


def train_lightgbm(X: np.ndarray, y: np.ndarray, lgb) -> tuple:
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    return model, rmse


def _to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64, np.int8, np.int16)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    return obj


def solve_ridge_weights(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Ridge regression with L2 regularization; weights clipped to non-negative and normalized."""
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    w = ridge.coef_
    w = np.maximum(w, 0)
    total = w.sum()
    if total > 0:
        w = w / total
    return dict(zip(feature_names, w))


def solve_nnls_weights(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Non-negative least squares; weights normalized to sum to 1."""
    w, _ = nnls(X, y)
    total = w.sum()
    if total > 0:
        w = w / total
    return dict(zip(feature_names, w))


def aggregate_dimension_weights(feature_weights: dict[str, float]) -> dict[str, float]:
    """Roll up feature weights into 3 dimension weights."""
    dim_sums = {"fundamentals": 0.0, "growth": 0.0, "technical": 0.0}
    for feat, w in feature_weights.items():
        dim = DIMENSION_MAP.get(feat, "other")
        dim_sums[dim] = dim_sums.get(dim, 0.0) + w
    total = sum(dim_sums.values())
    return {k: round(v / total, 4) if total > 0 else 0.0 for k, v in dim_sums.items()}


def rolling_train(df: pd.DataFrame, window_months: int = 18) -> pd.DataFrame:
    """Rolling-window train: each month uses prior N months to fit weights."""
    lgb = _ensure_lightgbm()
    dates = sorted(df["trade_date"].unique())
    rows = []

    for i in range(window_months, len(dates)):
        train_dates = dates[i - window_months : i]
        test_date = dates[i]

        train_df = df[df["trade_date"].isin(train_dates)]
        train_df = fill_na(train_df, FEATURE_COLS)

        X = train_df[FEATURE_COLS].values
        y = train_df["forward_return"].values

        if len(X) < 100:
            continue

        # Standardize features for linear models (tree model uses raw X)
        scaler = StandardScaler()
        scaler.fit(X)
        X_std = scaler.transform(X)

        # LightGBM (raw features)
        model, rmse = train_lightgbm(X, y, lgb)
        importances = model.feature_importances_

        # Ridge / NNLS on standardized features
        ridge_w = solve_ridge_weights(X_std, y, FEATURE_COLS)
        nnls_w = solve_nnls_weights(X_std, y, FEATURE_COLS)

        dim_ridge = aggregate_dimension_weights(ridge_w)
        dim_nnls = aggregate_dimension_weights(nnls_w)

        row = {"test_date": test_date, "rmse": rmse}
        for f in FEATURE_COLS:
            row[f"ridge_{f}"] = ridge_w.get(f, 0)
            row[f"nnls_{f}"] = nnls_w.get(f, 0)
            row[f"imp_{f}"] = dict(zip(FEATURE_COLS, importances)).get(f, 0)
        for dim in ["fundamentals", "growth", "technical"]:
            row[f"dim_ridge_{dim}"] = dim_ridge[dim]
            row[f"dim_nnls_{dim}"] = dim_nnls[dim]

        rows.append(row)

    return pd.DataFrame(rows)


def final_train(df: pd.DataFrame) -> dict:
    """Train on full history and output recommended weights."""
    lgb = _ensure_lightgbm()
    df = fill_na(df, FEATURE_COLS)
    X = df[FEATURE_COLS].values
    y = df["forward_return"].values

    # Standardize for linear models
    scaler = StandardScaler()
    scaler.fit(X)
    X_std = scaler.transform(X)

    model, rmse = train_lightgbm(X, y, lgb)
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))

    ridge_w = solve_ridge_weights(X_std, y, FEATURE_COLS)
    nnls_w = solve_nnls_weights(X_std, y, FEATURE_COLS)

    dim_ridge = aggregate_dimension_weights(ridge_w)
    dim_nnls = aggregate_dimension_weights(nnls_w)

    return {
        "rmse": rmse,
        "feature_ridge": {k: round(v, 6) for k, v in ridge_w.items()},
        "feature_nnls": {k: round(v, 6) for k, v in nnls_w.items()},
        "dimension_ridge": dim_ridge,
        "dimension_nnls": dim_nnls,
        "feature_importance": {k: round(v, 6) for k, v in importances.items()},
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }


def main():
    if not INPUT_PATH.exists():
        print(f"Training data not found: {INPUT_PATH}")
        print("Run: poetry run python scripts/collect_training_data.py")
        return

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} records from {INPUT_PATH}")
    print(f"Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"Unique dates: {df['trade_date'].nunique()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Rolling window
    print("\nRunning rolling-window training (window=18 months)...")
    rolling_df = rolling_train(df, window_months=18)
    rolling_path = OUTPUT_DIR / "rolling_weights.csv"
    rolling_df.to_csv(rolling_path, index=False)
    print(f"Saved rolling weights to {rolling_path}")

    if not rolling_df.empty:
        print("\nRolling dimension weights (Ridge, last 6 months average):")
        recent = rolling_df.tail(6)
        for dim in ["fundamentals", "growth", "technical"]:
            avg = recent[f"dim_ridge_{dim}"].mean()
            print(f"  {dim}: {avg:.4f}")

    # Final train on full history
    print("\nTraining on full history...")
    final = final_train(df)
    final_path = OUTPUT_DIR / "final_weights.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(_to_python(final), f, ensure_ascii=False, indent=2)
    print(f"Saved final weights to {final_path}")

    print("\n=== Recommended Dimension Weights (Ridge) ===")
    for dim, w in final["dimension_ridge"].items():
        print(f"  {dim}: {w:.4f}")

    print("\n=== Recommended Dimension Weights (NNLS) ===")
    for dim, w in final["dimension_nnls"].items():
        print(f"  {dim}: {w:.4f}")

    print("\n=== Top 10 Feature Importances (LightGBM) ===")
    imp = final["feature_importance"]
    for feat, w in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat}: {w:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
