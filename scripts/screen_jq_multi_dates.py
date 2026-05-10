"""Multi-date JQ-style 3-dimension screener.

Generates signal JSONL files for backtrader rebalance backtest.

Usage:
    poetry run python scripts/screen_jq_multi_dates.py \
        --start-date 2023-01-01 --end-date 2024-12-31 \
        --rebalance-freq month --top-n 20
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
os.environ.setdefault(
    "TUSHARE_DB_PATH", str(Path("db/tushare_data.db").resolve())
)

from src.backtesting.jq_adapter import JQDataAdapter
from src.backtesting.jq_screener import run_screener


def generate_rebalance_dates(start: str, end: str, freq: str) -> list[str]:
    import duckdb

    db_path = os.environ.get("TUSHARE_DB_PATH", "db/tushare_data.db")
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute(
        """
        SELECT cal_date FROM trade_cal
        WHERE is_open = 1 AND cal_date BETWEEN ? AND ?
        ORDER BY cal_date
        """,
        [start.replace("-", ""), end.replace("-", "")],
    ).fetchdf()
    conn.close()

    if df.empty:
        return []

    df["cal_date"] = pd.to_datetime(df["cal_date"], format="%Y%m%d")

    if freq == "month":
        groups = df.groupby([df["cal_date"].dt.year, df["cal_date"].dt.month])
        return [g["cal_date"].iloc[-1].strftime("%Y-%m-%d") for _, g in groups]
    elif freq == "quarter":
        groups = df.groupby([df["cal_date"].dt.year, df["cal_date"].dt.quarter])
        return [g["cal_date"].iloc[-1].strftime("%Y-%m-%d") for _, g in groups]
    return []


def screen_date(adapter: JQDataAdapter, date_str: str, top_n: int) -> None:
    result_file = Path("batch_screener_results/jq") / f"backtest_jq_{date_str.replace('-', '')}.jsonl"
    if result_file.exists():
        print(f"  [{date_str}] Already exists, skipping.")
        return

    print(f"  [{date_str}] Screening ...")
    try:
        results, _ = run_screener(adapter, date_str, top_n=top_n)
    except (ValueError, KeyError, duckdb.Error) as e:
        print(f"  [{date_str}] FAILED: {e}")
        return

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        for ticker, score in results:
            rec = {"ticker": ticker, "score": round(score, 4)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  [{date_str}] Done -> {len(results)} stocks.")


def main():
    parser = argparse.ArgumentParser(description="JQ multi-date screener")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--rebalance-freq", default="month", choices=["month", "quarter"]
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--db-path", default="db/tushare_data.db")
    args = parser.parse_args()

    dates = generate_rebalance_dates(
        args.start_date, args.end_date, args.rebalance_freq
    )

    print(f"JQ Screener: {args.start_date} ~ {args.end_date}")
    print(f"Frequency: {args.rebalance_freq}")
    print(f"Dates: {len(dates)}")
    print(f"Top-N: {args.top_n}")
    print()

    adapter = JQDataAdapter(args.db_path)
    try:
        for date_str in dates:
            screen_date(adapter, date_str, args.top_n)
    finally:
        adapter.close()

    print("\nAll dates complete.")


if __name__ == "__main__":
    main()
