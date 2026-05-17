"""Retry failed fina_indicator for specific tickers."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import duckdb
import pandas as pd
import tushare as ts

DB_PATH = Path("src/data/tushare_data.db")
PROGRESS_PATH = Path("db/sync_fina_indicator_progress.jsonl")
SLEEP_SEC = 0.5

RETRY_TICKERS = [
    "300324.SZ", "300933.SZ", "600739.SH", "600827.SH",
]


def get_tushare_pro():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("TUSHARE_TOKEN not set")
    return ts.pro_api(token)


def _sync_ticker_once(pro, ts_code: str, conn: duckdb.DuckDBPyConnection) -> dict:
    t0 = time.time()
    all_dfs: list[pd.DataFrame] = []

    try:
        df = pro.fina_indicator(ts_code=ts_code)
        if df is None or df.empty:
            return {"ts_code": ts_code, "status": "empty", "rows": 0, "elapsed": round(time.time() - t0, 1), "synced_at": datetime.now().isoformat()}

        df = df.loc[:, ~df.columns.duplicated()]
        all_dfs.append(df)

        while len(df) == 100:
            min_end_date = str(df["end_date"].min())
            time.sleep(0.3)
            df = pro.fina_indicator(ts_code=ts_code, end_date=min_end_date)
            if df is None or df.empty:
                break
            df = df.loc[:, ~df.columns.duplicated()]

            existing_dates = set(pd.concat(all_dfs)["end_date"].unique())
            new_rows = df[~df["end_date"].isin(existing_dates)]
            if new_rows.empty:
                break
            all_dfs.append(new_rows)

            if len(df) < 100:
                break

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_code", "end_date"])

        conn.register("tmp_fina", combined)
        conn.execute("""
            INSERT OR REPLACE INTO fina_indicator
            SELECT * FROM tmp_fina
        """)
        conn.unregister("tmp_fina")

        return {
            "ts_code": ts_code,
            "status": "ok",
            "rows": len(combined),
            "batches": len(all_dfs),
            "elapsed": round(time.time() - t0, 1),
            "synced_at": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "ts_code": ts_code,
            "status": "error",
            "error": str(e),
            "elapsed": round(time.time() - t0, 1),
            "synced_at": datetime.now().isoformat(),
        }


def main():
    print("=" * 60)
    print("Retry failed fina_indicator -> DuckDB")
    print(f"Tickers: {RETRY_TICKERS}")
    print("=" * 60)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    pro = get_tushare_pro()
    conn = duckdb.connect(str(DB_PATH))

    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    progress_f = open(PROGRESS_PATH, "a", encoding="utf-8")

    try:
        for i, ts_code in enumerate(RETRY_TICKERS, 1):
            result = _sync_ticker_once(pro, ts_code, conn)
            progress_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress_f.flush()

            status = result["status"]
            rows = result.get("rows", 0)
            elapsed = result["elapsed"]
            print(f"[{i}/{len(RETRY_TICKERS)}] {ts_code}  {status:6s}  rows={rows:3d}  {elapsed:.1f}s")

            if status == "error":
                print(f"  [WARN] {ts_code} error: {result.get('error', '')}")

            time.sleep(SLEEP_SEC)
    finally:
        progress_f.close()
        conn.close()

    print("\nRetry complete.")


if __name__ == "__main__":
    main()
