"""同步 adj_factor 复权因子到本地 DuckDB.

用法:
    python scripts/sync_adj_factor.py 603288.SH          # 单只股票
    python scripts/sync_adj_factor.py --all               # 全市场（约2小时）
    python scripts/sync_adj_factor.py --date 20260508     # 按日期全市场
"""

import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import duckdb
import pandas as pd
import tushare as ts

DB_PATH = Path("src/data/tushare_data.db")


def get_tushare_pro():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("TUSHARE_TOKEN not set")
    return ts.pro_api(token, timeout=60)


def ensure_table(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS adj_factor (
            ts_code VARCHAR NOT NULL,
            trade_date VARCHAR NOT NULL,
            adj_factor DOUBLE NOT NULL,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)


def sync_by_stock(pro, conn, ts_code: str):
    """同步单只股票全部历史 adj_factor."""
    print(f"同步 {ts_code} ...")
    df = pro.adj_factor(ts_code=ts_code)
    if df.empty:
        print(f"  无数据")
        return 0
    # Keep only columns we need
    df = df[["ts_code", "trade_date", "adj_factor"]]
    # Upsert via DELETE + INSERT (DuckDB < 0.10 may not support ON CONFLICT)
    conn.execute(f"""
        DELETE FROM adj_factor WHERE ts_code = '{ts_code}'
    """)
    conn.register("tmp_adj", df)
    conn.execute("INSERT INTO adj_factor SELECT * FROM tmp_adj")
    conn.unregister("tmp_adj")
    print(f"  插入 {len(df)} 条")
    return len(df)


def sync_by_date(pro, conn, trade_date: str):
    """同步某一天全市场 adj_factor."""
    df = pro.adj_factor(trade_date=trade_date)
    if df.empty:
        return 0
    df = df[["ts_code", "trade_date", "adj_factor"]]
    conn.execute(f"""
        DELETE FROM adj_factor WHERE trade_date = '{trade_date}'
    """)
    conn.register("tmp_adj", df)
    conn.execute("INSERT INTO adj_factor SELECT * FROM tmp_adj")
    conn.unregister("tmp_adj")
    return len(df)


def sync_all(pro, conn):
    """全量同步：按日期逐日拉取全市场数据."""
    # Get all distinct trade_dates from daily table
    dates = conn.execute("""
        SELECT DISTINCT trade_date FROM daily ORDER BY trade_date ASC
    """).fetchdf()["trade_date"].tolist()
    total = len(dates)
    print(f"需同步 {total} 个交易日的全市场 adj_factor ...")
    inserted = 0
    for i, d in enumerate(dates):
        n = sync_by_date(pro, conn, d)
        inserted += n
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] {d} 已同步, 累计 {inserted} 条")
    print(f"\n全量同步完成, 共 {inserted} 条")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stock", nargs="?", help="单只股票代码, 如 603288.SH")
    parser.add_argument("--all", action="store_true", help="同步全市场")
    parser.add_argument("--date", help="同步某一天, 如 20260508")
    args = parser.parse_args()

    pro = get_tushare_pro()
    conn = duckdb.connect(str(DB_PATH))
    ensure_table(conn)

    if args.stock:
        sync_by_stock(pro, conn, args.stock)
    elif args.date:
        n = sync_by_date(pro, conn, args.date)
        print(f"同步 {args.date} 完成, {n} 条")
    elif args.all:
        sync_all(pro, conn)
    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
