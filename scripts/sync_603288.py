"""补充下载海天味业(603288.SH)缺失的 fina_indicator 和 balancesheet 数据."""

import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import duckdb
import pandas as pd
import tushare as ts

DB_PATH = Path("src/data/tushare_data.db")
TS_CODE = "603288.SH"
SLEEP_SEC = 0.5


def get_tushare_pro():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("TUSHARE_TOKEN not set")
    return ts.pro_api(token, timeout=60)


def ensure_table(conn: duckdb.DuckDBPyConnection, table_name: str, sample_df: pd.DataFrame):
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
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join(cols)}
    )
    """
    conn.execute(create_sql)


def sync_table(pro, ts_code: str, conn: duckdb.DuckDBPyConnection, table_name: str, api_name: str) -> dict:
    """Download all historical data for one ticker from one Tushare API."""
    t0 = time.time()
    all_dfs: list[pd.DataFrame] = []

    print(f"  -> 开始下载 {table_name} for {ts_code}...")

    try:
        api = getattr(pro, api_name)
        df = api(ts_code=ts_code)
        if df is None or df.empty:
            return {"table": table_name, "ts_code": ts_code, "status": "empty", "rows": 0,
                    "elapsed": round(time.time() - t0, 1), "synced_at": datetime.now().isoformat()}

        df = df.loc[:, ~df.columns.duplicated()]
        all_dfs.append(df)
        print(f"     第1批: {len(df)} 条, end_date 范围 {df['end_date'].min()} ~ {df['end_date'].max()}")

        while len(df) == 100:
            min_end_date = str(df["end_date"].min())
            time.sleep(0.3)
            df = api(ts_code=ts_code, end_date=min_end_date)
            if df is None or df.empty:
                break
            df = df.loc[:, ~df.columns.duplicated()]

            existing_dates = set(pd.concat(all_dfs)["end_date"].unique())
            new_rows = df[~df["end_date"].isin(existing_dates)]
            if new_rows.empty:
                print(f"     分页结束: 无新数据")
                break
            all_dfs.append(new_rows)
            print(f"     第{len(all_dfs)}批: {len(new_rows)} 条新数据, end_date 到 {df['end_date'].min()}")

            if len(df) < 100:
                break

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_code", "end_date"])

        # Ensure table exists
        existing_tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if table_name not in existing_tables:
            print(f"     创建表 {table_name}...")
            ensure_table(conn, table_name, combined.head(0))

        # Delete existing data for this stock, then insert fresh
        conn.execute(f"DELETE FROM {table_name} WHERE ts_code = '{ts_code}'")
        conn.register("tmp_sync", combined)
        conn.execute(f"""
            INSERT INTO {table_name}
            SELECT * FROM tmp_sync
        """)
        conn.unregister("tmp_sync")

        return {
            "table": table_name,
            "ts_code": ts_code,
            "status": "ok",
            "rows": len(combined),
            "batches": len(all_dfs),
            "elapsed": round(time.time() - t0, 1),
            "synced_at": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "table": table_name,
            "ts_code": ts_code,
            "status": "error",
            "error": str(e),
            "elapsed": round(time.time() - t0, 1),
            "synced_at": datetime.now().isoformat(),
        }


def main():
    print("=" * 60)
    print(f"补充下载 {TS_CODE} (海天味业) 缺失数据")
    print("=" * 60)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    pro = get_tushare_pro()
    conn = duckdb.connect(str(DB_PATH))

    # 1. fina_indicator
    print("\n[1/2] 下载 fina_indicator...")
    result1 = sync_table(pro, TS_CODE, conn, "fina_indicator", "fina_indicator")
    print(f"     结果: {result1['status']}, {result1.get('rows', 0)} 条, {result1['elapsed']:.1f}s")
    if result1["status"] == "error":
        print(f"     错误: {result1.get('error', '')}")
    time.sleep(SLEEP_SEC)

    # 2. balancesheet
    print("\n[2/2] 下载 balancesheet...")
    result2 = sync_table(pro, TS_CODE, conn, "balancesheet", "balancesheet")
    print(f"     结果: {result2['status']}, {result2.get('rows', 0)} 条, {result2['elapsed']:.1f}s")
    if result2["status"] == "error":
        print(f"     错误: {result2.get('error', '')}")

    conn.close()

    print("\n" + "=" * 60)
    print("下载完成")
    print("=" * 60)
    print(f"  fina_indicator : {result1.get('rows', 0)} 条 ({result1['status']})")
    print(f"  balancesheet   : {result2.get('rows', 0)} 条 ({result2['status']})")


if __name__ == "__main__":
    main()
