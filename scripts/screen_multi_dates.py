"""Multi-date stock screener using 3agents.

Generates signals for multiple rebalancing dates by reusing
run_backtest_3agents logic.

Usage:
    poetry run python scripts/screen_multi_dates.py \
        --start-date 2023-01-01 \
        --end-date 2024-12-31 \
        --rebalance-freq month \
        --workers 4
"""

import argparse
import json
import multiprocessing as mp
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from run_backtest_3agents import (
    OUTPUT_DIR,
    load_done_tickers,
    load_tickers_from_fina,
    worker_init,
    worker_task,
)


def generate_rebalance_dates(start: str, end: str, freq: str) -> list[str]:
    """Generate last trade date of each month/quarter using trade_cal."""
    import duckdb
    import os

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
        return [g.iloc[-1].strftime("%Y-%m-%d") for _, g in groups]
    elif freq == "quarter":
        groups = df.groupby([df["cal_date"].dt.year, df["cal_date"].dt.quarter])
        return [g.iloc[-1].strftime("%Y-%m-%d") for _, g in groups]
    return []


def screen_date(date_str: str, tickers: list[str], workers: int) -> None:
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    start_dt = target_dt - pd.Timedelta(days=365)
    start_date = start_dt.strftime("%Y-%m-%d")

    result_file = OUTPUT_DIR / f"backtest_3agents_{date_str.replace('-', '')}.jsonl"

    done = load_done_tickers(result_file)
    pending = [t for t in tickers if t not in done]

    if not pending:
        print(f"  [{date_str}] Already done ({len(done)}).")
        return

    print(f"  [{date_str}] Analyzing {len(pending)}/{len(tickers)} stocks ...")

    with open(result_file, "a", encoding="utf-8") as f:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=worker_init) as pool:
            task_args = [(t, date_str, start_date) for t in pending]
            for result in pool.imap_unordered(worker_task, task_args):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    print(f"  [{date_str}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Multi-date stock screener")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--rebalance-freq", default="month", choices=["month", "quarter"]
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tickers = load_tickers_from_fina()
    dates = generate_rebalance_dates(
        args.start_date, args.end_date, args.rebalance_freq
    )

    print(f"Multi-date Screener: {args.start_date} ~ {args.end_date}")
    print(f"Frequency: {args.rebalance_freq}")
    print(f"Dates: {len(dates)}")
    print(f"Tickers: {len(tickers)}")
    print()

    for date_str in dates:
        screen_date(date_str, tickers, args.workers)

    print("\nAll dates complete.")


if __name__ == "__main__":
    main()
