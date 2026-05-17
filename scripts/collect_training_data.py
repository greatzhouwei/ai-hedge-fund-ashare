"""Collect historical screener scores and forward returns for model training.

Usage:
    poetry run python scripts/collect_training_data.py

Outputs training_data.parquet with one row per stock per month.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

from src.backtesting.jq_adapter import JQDataAdapter
from src.backtesting.jq_screener import (
    WEIGHTS,
    combine_scores,
    get_candidate_stocks,
    score_fundamentals,
    score_growth,
    score_technical,
)

DB_PATH = Path("src/data/tushare_data.db")
OUTPUT_PATH = Path("training_data.parquet")
PROGRESS_PATH = Path("db/collect_training_progress.jsonl")


def get_monthly_trade_dates(conn: duckdb.DuckDBPyConnection, start_date: str, end_date: str) -> list[str]:
    start_num = start_date.replace("-", "")
    end_num = end_date.replace("-", "")
    df = conn.execute(
        """
        SELECT MIN(cal_date) as first_date
        FROM trade_cal
        WHERE is_open = 1 AND cal_date BETWEEN ? AND ?
        GROUP BY SUBSTR(cal_date, 1, 6)
        ORDER BY first_date
        """,
        [start_num, end_num],
    ).fetchdf()
    return [pd.to_datetime(d, format="%Y%m%d").strftime("%Y-%m-%d") for d in df["first_date"]]


def get_prices_on_date(conn: duckdb.DuckDBPyConnection, tickers: list[str], date: str) -> dict[str, float]:
    date_num = date.replace("-", "")
    if not tickers:
        return {}
    placeholders = ",".join(["?"] * len(tickers))
    df = conn.execute(
        f"""
        SELECT ts_code, close
        FROM daily
        WHERE ts_code IN ({placeholders}) AND trade_date = ?
        """,
        tickers + [date_num],
    ).fetchdf()
    return {r["ts_code"]: float(r["close"]) for _, r in df.iterrows()}


def extract_features(fund_detail: dict, growth_detail: dict, tech_detail: dict) -> dict:
    """Flatten nested details into a single feature dict."""
    feats = {}

    # Fundamentals
    prof = fund_detail.get("profitability", {})
    health = fund_detail.get("health", {})
    val = fund_detail.get("valuation", {})
    feats["fund_roe"] = prof.get("roe")
    feats["fund_net_margin"] = prof.get("net_margin")
    feats["fund_op_margin"] = prof.get("op_margin")
    feats["fund_curr_ratio"] = health.get("curr_ratio")
    feats["fund_de_ratio"] = health.get("de_ratio")
    feats["fund_fcf_ps"] = health.get("fcf_ps")
    feats["fund_eps"] = health.get("eps")
    feats["fund_pe"] = val.get("pe")
    feats["fund_pb"] = val.get("pb")
    feats["fund_ps"] = val.get("ps")
    feats["fund_score"] = fund_detail.get("overall")

    # Growth
    feats["growth_revenue_growth"] = growth_detail.get("revenue_growth")
    feats["growth_revenue_trend"] = growth_detail.get("revenue_trend")
    feats["growth_eps_growth"] = growth_detail.get("eps_growth")
    feats["growth_eps_trend"] = growth_detail.get("eps_trend")
    feats["growth_fcf_growth"] = growth_detail.get("fcf_growth")
    feats["growth_fcf_trend"] = growth_detail.get("fcf_trend")
    feats["growth_bv_growth"] = growth_detail.get("bv_growth")
    feats["growth_peg"] = growth_detail.get("peg")
    feats["growth_ps"] = growth_detail.get("ps")
    feats["growth_gross_margin"] = growth_detail.get("gross_margin")
    feats["growth_gm_trend"] = growth_detail.get("gm_trend")
    feats["growth_operating_margin"] = growth_detail.get("operating_margin")
    feats["growth_om_trend"] = growth_detail.get("om_trend")
    feats["growth_net_margin"] = growth_detail.get("net_margin")
    feats["growth_nm_trend"] = growth_detail.get("nm_trend")
    feats["growth_growth_score"] = growth_detail.get("growth_score")
    feats["growth_val_score"] = growth_detail.get("val_score")
    feats["growth_margin_score"] = growth_detail.get("margin_score")
    feats["growth_score"] = growth_detail.get("weighted_score")

    # Technical
    feats["tech_adx"] = tech_detail.get("adx")
    feats["tech_tf_bullish"] = tech_detail.get("tf_bullish")
    feats["tech_rsi14"] = tech_detail.get("rsi14")
    feats["tech_price_vs_bb"] = tech_detail.get("price_vs_bb")
    feats["tech_mr_bullish"] = tech_detail.get("mr_bullish")
    feats["tech_mom_3m"] = tech_detail.get("mom_3m")
    feats["tech_mom_6m"] = tech_detail.get("mom_6m")
    feats["tech_mom_bullish"] = tech_detail.get("mom_bullish")
    feats["tech_vol_regime"] = tech_detail.get("vol_regime")
    feats["tech_vol_bullish"] = tech_detail.get("vol_bullish")
    feats["tech_score"] = tech_detail.get("overall")

    return feats


def _load_done_dates(progress_path: Path) -> set[str]:
    done = set()
    if not progress_path.exists():
        return done
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "date" in rec:
                    done.add(rec["date"])
            except Exception:
                pass
    return done


def main():
    adapter = JQDataAdapter(str(DB_PATH))
    conn = adapter.conn

    dates = get_monthly_trade_dates(conn, "2022-01-01", "2024-12-31")
    print(f"Monthly trade dates: {len(dates)}")

    done_dates = _load_done_dates(PROGRESS_PATH)
    print(f"Already done: {len(done_dates)}")

    # Load existing data if any
    records: list[dict] = []
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        records = existing.to_dict("records")
        print(f"Loaded {len(records)} existing records from {OUTPUT_PATH}")

    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    progress_f = open(PROGRESS_PATH, "a", encoding="utf-8") if PROGRESS_PATH.exists() else open(PROGRESS_PATH, "w", encoding="utf-8")

    for i, date in enumerate(dates[:-1]):
        if date in done_dates:
            continue
        next_date = dates[i + 1]
        next_date = dates[i + 1]
        t0 = datetime.now()

        candidates = get_candidate_stocks(adapter, date)
        if not candidates:
            print(f"[{i + 1}/{len(dates) - 1}] {date}: no candidates")
            continue

        fund_scores, fund_details = score_fundamentals(adapter, candidates, date)
        growth_scores, growth_details = score_growth(adapter, candidates, date)
        tech_scores, tech_details = score_technical(adapter, candidates, date)

        combined = combine_scores(fund_scores, growth_scores, tech_scores, WEIGHTS)

        # Get buy/sell prices
        buy_prices = get_prices_on_date(conn, candidates, date)
        sell_prices = get_prices_on_date(conn, candidates, next_date)

        month_records = 0
        for t in combined:
            if t not in buy_prices or t not in sell_prices:
                continue
            if t not in fund_details or t not in growth_details or t not in tech_details:
                continue

            buy_p = buy_prices[t]
            sell_p = sell_prices[t]
            if buy_p <= 0:
                continue

            forward_ret = (sell_p - buy_p) / buy_p

            feats = extract_features(fund_details[t], growth_details[t], tech_details[t])
            feats["trade_date"] = date
            feats["ticker"] = t
            feats["forward_return"] = forward_ret
            feats["buy_price"] = buy_p
            feats["sell_price"] = sell_p
            records.append(feats)
            month_records += 1

        elapsed = (datetime.now() - t0).total_seconds()
        print(
            f"[{i + 1}/{len(dates) - 1}] {date} -> {next_date} "
            f"candidates={len(candidates)} records={month_records} {elapsed:.1f}s"
        )

        progress_f.write(
            json.dumps(
                {
                    "date": date,
                    "next_date": next_date,
                    "candidates": len(candidates),
                    "records": month_records,
                    "elapsed": elapsed,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        progress_f.flush()

    progress_f.close()

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"\nSaved {len(df)} records to {OUTPUT_PATH}")
    else:
        print("\nNo records collected.")


if __name__ == "__main__":
    main()
