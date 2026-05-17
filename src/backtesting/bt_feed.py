"""DuckDB daily price data adapter for backtrader."""

import os
from datetime import datetime
from pathlib import Path

import backtrader as bt
import duckdb
import pandas as pd


def load_ticker_data(ts_code: str, db_path: str | None = None) -> pd.DataFrame:
    """Load daily OHLCV for a single ticker from DuckDB."""
    if db_path is None:
        db_path = os.environ.get("TUSHARE_DB_PATH", "src/data/tushare_data.db")

    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute(
        """
        SELECT trade_date, open, high, low, close, vol as volume
        FROM daily
        WHERE ts_code = ?
        ORDER BY trade_date
        """,
        [ts_code],
    ).fetchdf()
    conn.close()

    if df.empty:
        raise ValueError(f"No daily data found for {ts_code}")

    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df.set_index("trade_date", inplace=True)
    return df


def create_feed(
    ts_code: str,
    db_path: str | None = None,
    fromdate: datetime | None = None,
    todate: datetime | None = None,
) -> bt.feeds.PandasData:
    """Create a backtrader PandasData feed for a single ticker."""
    df = load_ticker_data(ts_code, db_path)

    kwargs: dict = {"dataname": df, "name": ts_code}
    if fromdate is not None:
        kwargs["fromdate"] = fromdate
    if todate is not None:
        kwargs["todate"] = todate

    return bt.feeds.PandasData(**kwargs)