"""同步海天味业(603288.SH) 2026-05-06 ~ 2026-05-08 的 daily_basic, daily, trade_cal 数据."""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import duckdb
import pandas as pd
import tushare as ts

DB_PATH = Path("src/data/tushare_data.db")
TS_CODE = "603288.SH"
START_DATE = "20260506"
END_DATE = "20260508"


def get_tushare_pro():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("TUSHARE_TOKEN not set")
    return ts.pro_api(token, timeout=60)


def main():
    print("=" * 60)
    print(f"同步 {TS_CODE} {START_DATE} ~ {END_DATE} 数据")
    print("=" * 60)

    pro = get_tushare_pro()
    conn = duckdb.connect(str(DB_PATH))

    # 1. trade_cal
    print("\n[1/3] 同步 trade_cal...")
    df_cal = pro.trade_cal(exchange="SSE", start_date="20260501", end_date="20260515")
    if not df_cal.empty:
        conn.execute("DELETE FROM trade_cal WHERE cal_date >= '20260501' AND cal_date <= '20260515'")
        conn.register("tmp_cal", df_cal)
        conn.execute("""
            INSERT INTO trade_cal
            SELECT exchange, cal_date, is_open, pretrade_date
            FROM tmp_cal
        """)
        conn.unregister("tmp_cal")
        print(f"     插入 {len(df_cal)} 条 trade_cal 记录")

    # 2. daily_basic (603288.SH only)
    print(f"\n[2/3] 同步 daily_basic {TS_CODE}...")
    df_db = pro.daily_basic(ts_code=TS_CODE, start_date=START_DATE, end_date=END_DATE)
    if not df_db.empty:
        # Ensure table columns match
        existing_cols = [row[0] for row in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'daily_basic'"
        ).fetchall()]
        if not existing_cols:
            # Fallback: describe the table
            sample = conn.execute("SELECT * FROM daily_basic LIMIT 0").fetchdf()
            existing_cols = list(sample.columns)

        # Keep only columns that exist in the table
        df_db = df_db[[c for c in df_db.columns if c in existing_cols]]

        conn.execute(f"DELETE FROM daily_basic WHERE ts_code = '{TS_CODE}' AND trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'")
        conn.register("tmp_db", df_db)
        conn.execute("""
            INSERT INTO daily_basic
            SELECT * FROM tmp_db
        """)
        conn.unregister("tmp_db")
        print(f"     插入 {len(df_db)} 条 daily_basic 记录")
        for _, r in df_db.iterrows():
            print(f"       {r['trade_date']}: PE={r.get('pe_ttm')}, PB={r.get('pb')}, PS={r.get('ps_ttm')}")

    # 3. daily (603288.SH only)
    print(f"\n[3/3] 同步 daily {TS_CODE}...")
    df_d = pro.daily(ts_code=TS_CODE, start_date=START_DATE, end_date=END_DATE)
    if not df_d.empty:
        conn.execute(f"DELETE FROM daily WHERE ts_code = '{TS_CODE}' AND trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'")
        conn.register("tmp_d", df_d)
        conn.execute("""
            INSERT INTO daily
            SELECT * FROM tmp_d
        """)
        conn.unregister("tmp_d")
        print(f"     插入 {len(df_d)} 条 daily 记录")
        for _, r in df_d.iterrows():
            print(f"       {r['trade_date']}: open={r['open']}, close={r['close']}, vol={r['vol']}")

    conn.close()
    print("\n同步完成。")


if __name__ == "__main__":
    main()
