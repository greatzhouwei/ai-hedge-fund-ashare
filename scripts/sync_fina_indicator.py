"""Sync Tushare fina_indicator data into DuckDB.

Usage:
    poetry run python scripts/sync_fina_indicator.py

Resumable via progress log in db/sync_fina_indicator_progress.jsonl.
Rate-limit aware (~200 req/min on free tier -> 0.3s sleep).
"""

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

DB_PATH = Path("db/tushare_data.db")
PROGRESS_PATH = Path("db/sync_fina_indicator_progress.jsonl")
BATCH_SIZE = 200  # rows per INSERT
SLEEP_SEC = 0.5   # ~80 req/min, still conservative for free tier


def get_tushare_pro():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("TUSHARE_TOKEN not set")
    return ts.pro_api(token)


def get_stock_codes(db_path: Path) -> list[str]:
    conn = duckdb.connect(str(db_path), read_only=True)
    codes = conn.execute("SELECT ts_code FROM stock_basic ORDER BY ts_code").fetchdf()[
        "ts_code"
    ].tolist()
    conn.close()
    return codes


def load_done_tickers(progress_path: Path) -> set[str]:
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
                if rec.get("status") == "ok":
                    done.add(rec["ts_code"])
            except Exception:
                pass
    return done


def ensure_table(conn: duckdb.DuckDBPyConnection, sample_df: pd.DataFrame):
    """Create fina_indicator table if not exists, inferring schema from sample_df."""
    type_map = {
        "object": "VARCHAR",
        "int64": "BIGINT",
        "float64": "DOUBLE",
        "bool": "BOOLEAN",
    }
    cols = []
    for col, dtype in sample_df.dtypes.items():
        safe_col = f'"{col}"'
        sql_type = type_map.get(str(dtype), "VARCHAR")
        cols.append(f"{safe_col} {sql_type}")

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS fina_indicator (
        {', '.join(cols)}
    )
    """
    conn.execute(create_sql)

    # Add index for common queries
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_fina_ts_end ON fina_indicator(ts_code, end_date)"
    )


def _is_ip_limit_error(error_msg: str) -> bool:
    """Detect Tushare IP-access limit errors."""
    return "IP" in error_msg and ("限" in error_msg or "limit" in error_msg.lower())


def _sync_ticker_once(pro, ts_code: str, conn: duckdb.DuckDBPyConnection) -> dict:
    """Download fina_indicator for one ticker (single attempt)."""
    t0 = time.time()
    all_dfs: list[pd.DataFrame] = []

    try:
        # First fetch: most recent records (up to 100)
        df = pro.fina_indicator(ts_code=ts_code)
        if df is None or df.empty:
            return {"ts_code": ts_code, "status": "empty", "rows": 0, "elapsed": round(time.time() - t0, 1), "synced_at": datetime.now().isoformat()}

        df = df.loc[:, ~df.columns.duplicated()]
        all_dfs.append(df)

        # Paginate when we hit the 100-row cap
        while len(df) == 100:
            min_end_date = str(df["end_date"].min())
            time.sleep(0.3)  # brief rate-limit breath between pages
            df = pro.fina_indicator(ts_code=ts_code, end_date=min_end_date)
            if df is None or df.empty:
                break
            df = df.loc[:, ~df.columns.duplicated()]

            # Drop rows already fetched (end_date == min_end_date will overlap)
            existing_dates = set(pd.concat(all_dfs)["end_date"].unique())
            new_rows = df[~df["end_date"].isin(existing_dates)]
            if new_rows.empty:
                break
            all_dfs.append(new_rows)

            # Fewer than 100 means we reached the end of history
            if len(df) < 100:
                break

        # Combine and deduplicate by (ts_code, end_date)
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_code", "end_date"])

        # Insert into DuckDB
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


def sync_ticker(pro, ts_code: str, conn: duckdb.DuckDBPyConnection, max_retries: int = 3) -> dict:
    """Download fina_indicator with retry and IP-limit backoff."""
    last_result = None
    for attempt in range(max_retries + 1):
        last_result = _sync_ticker_once(pro, ts_code, conn)
        if last_result["status"] in ("ok", "empty"):
            return last_result
        error_msg = last_result.get("error", "")
        if _is_ip_limit_error(error_msg):
            if attempt < max_retries:
                wait = 90 + attempt * 30  # 90s, 120s, 150s
                print(f"  [IP_LIMIT] {ts_code} attempt {attempt + 1}/{max_retries + 1} failed, waiting {wait}s...")
                time.sleep(wait)
                continue
        # Non-IP error or exhausted retries
        return last_result
    return last_result


def main():
    print("=" * 60)
    print("Sync Tushare fina_indicator -> DuckDB")
    print("=" * 60)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    pro = get_tushare_pro()
    codes = get_stock_codes(DB_PATH)
    print(f"Total stocks in stock_basic: {len(codes)}")

    done = load_done_tickers(PROGRESS_PATH)
    print(f"Already synced: {len(done)}")

    pending = [c for c in codes if c not in done]
    print(f"Pending: {len(pending)}")

    if not pending:
        print("All done!")
        return

    # Connect to DuckDB (writable)
    conn = duckdb.connect(str(DB_PATH))

    # Create table from first pending ticker if table doesn't exist yet
    existing_tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    if "fina_indicator" not in existing_tables:
        print("Creating fina_indicator table from first sample...")
        sample_df = pro.fina_indicator(ts_code=pending[0])
        if sample_df is not None and not sample_df.empty:
            sample_df = sample_df.loc[:, ~sample_df.columns.duplicated()]
            ensure_table(conn, sample_df)
            print(f"Table created with {len(sample_df.columns)} columns")
        else:
            raise RuntimeError("Failed to fetch sample fina_indicator")

    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    progress_f = open(PROGRESS_PATH, "a", encoding="utf-8")

    try:
        for i, ts_code in enumerate(pending, 1):
            result = sync_ticker(pro, ts_code, conn)
            progress_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress_f.flush()

            status = result["status"]
            rows = result.get("rows", 0)
            elapsed = result["elapsed"]

            if i % 10 == 0 or i <= 3 or i == len(pending):
                print(f"[{i:5d}/{len(pending)}] {ts_code}  {status:6s}  rows={rows:3d}  {elapsed:.1f}s  ({datetime.now().strftime('%H:%M:%S')})")

            if status == "error":
                print(f"  [WARN] {ts_code} error: {result.get('error', '')}")

            # Rate limit friendly
            time.sleep(SLEEP_SEC)
    finally:
        progress_f.close()
        conn.close()

    print("\nSync complete.")


if __name__ == "__main__":
    main()
