"""DuckDB local store for Tushare data.

Provides fast local queries as a fallback before calling Tushare API.
All connections are read-only (safe for multi-process access on Windows).
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).parent / "tushare_data.db"


class DuckDBStore:
    """Read-only wrapper around local DuckDB with Tushare data."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or os.environ.get("TUSHARE_DB_PATH", DEFAULT_DB_PATH))
        self._conn: duckdb.DuckDBPyConnection | None = None

    def _connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path), read_only=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------
    def get_daily(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Query daily price (open/high/low/close/vol/amount).

        Dates should be in 'YYYY-MM-DD' or 'YYYYMMDD' format.
        Returns None if table or ticker missing.
        """
        conn = self._connect()
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "daily" not in tables:
            return None

        # Normalise dates to YYYYMMDD
        s = _norm_date(start_date) if start_date else None
        e = _norm_date(end_date) if end_date else None

        sql = 'SELECT * FROM daily WHERE ts_code = ?'
        params: list = [ticker]
        if s:
            sql += ' AND trade_date >= ?'
            params.append(s)
        if e:
            sql += ' AND trade_date <= ?'
            params.append(e)
        sql += ' ORDER BY trade_date'

        df = conn.execute(sql, params).fetchdf()
        if df.empty:
            return None
        return df

    def get_daily_basic(
        self,
        ticker: str | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Query daily_basic (market cap, PE, PB, etc.).

        If trade_date is omitted, returns the latest available row for the ticker.
        """
        conn = self._connect()
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "daily_basic" not in tables:
            return None

        td = _norm_date(trade_date) if trade_date else None

        if ticker and td:
            sql = 'SELECT * FROM daily_basic WHERE ts_code = ? AND trade_date = ?'
            df = conn.execute(sql, [ticker, td]).fetchdf()
        elif ticker:
            sql = 'SELECT * FROM daily_basic WHERE ts_code = ? ORDER BY trade_date DESC LIMIT 1'
            df = conn.execute(sql, [ticker]).fetchdf()
        elif td:
            sql = 'SELECT * FROM daily_basic WHERE trade_date = ?'
            df = conn.execute(sql, [td]).fetchdf()
        else:
            return None

        if df.empty:
            return None
        return df

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------
    def get_fina_indicator(
        self,
        ticker: str,
        end_date: str | None = None,
        limit: int = 10,
    ) -> pd.DataFrame | None:
        """Query fina_indicator (ROE, margins, ratios, etc.)."""
        conn = self._connect()
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "fina_indicator" not in tables:
            return None

        e = _norm_date(end_date) if end_date else None

        sql = 'SELECT * FROM fina_indicator WHERE ts_code = ?'
        params: list = [ticker]
        if e:
            sql += ' AND end_date <= ?'
            params.append(e)
        sql += ' ORDER BY end_date DESC LIMIT ?'
        params.append(limit)

        df = conn.execute(sql, params).fetchdf()
        if df.empty:
            return None
        return df

    def get_balancesheet(
        self,
        ticker: str,
        end_date: str | None = None,
        limit: int = 10,
        fields: str | None = None,
    ) -> pd.DataFrame | None:
        """Query balancesheet."""
        conn = self._connect()
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "balancesheet" not in tables:
            return None

        e = _norm_date(end_date) if end_date else None
        cols = fields if fields else "*"

        sql = f'SELECT {cols} FROM balancesheet WHERE ts_code = ?'
        params: list = [ticker]
        if e:
            sql += ' AND end_date <= ?'
            params.append(e)
        sql += ' ORDER BY end_date DESC LIMIT ?'
        params.append(limit)

        df = conn.execute(sql, params).fetchdf()
        if df.empty:
            return None
        return df

    def get_income(
        self,
        ticker: str,
        end_date: str | None = None,
        limit: int = 10,
        fields: str | None = None,
    ) -> pd.DataFrame | None:
        """Query income statement."""
        conn = self._connect()
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "income" not in tables:
            return None

        e = _norm_date(end_date) if end_date else None
        cols = fields if fields else "*"

        sql = f'SELECT {cols} FROM income WHERE ts_code = ?'
        params: list = [ticker]
        if e:
            sql += ' AND end_date <= ?'
            params.append(e)
        sql += ' ORDER BY end_date DESC LIMIT ?'
        params.append(limit)

        df = conn.execute(sql, params).fetchdf()
        if df.empty:
            return None
        return df

    def get_cashflow(
        self,
        ticker: str,
        end_date: str | None = None,
        limit: int = 10,
    ) -> pd.DataFrame | None:
        """Query cashflow statement."""
        conn = self._connect()
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "cashflow" not in tables:
            return None

        e = _norm_date(end_date) if end_date else None

        sql = 'SELECT * FROM cashflow WHERE ts_code = ?'
        params: list = [ticker]
        if e:
            sql += ' AND end_date <= ?'
            params.append(e)
        sql += ' ORDER BY end_date DESC LIMIT ?'
        params.append(limit)

        df = conn.execute(sql, params).fetchdf()
        if df.empty:
            return None
        return df

    # ------------------------------------------------------------------
    # Calendar / misc
    # ------------------------------------------------------------------
    def get_trade_cal(
        self,
        start_date: str,
        end_date: str,
        is_open: str = "1",
    ) -> pd.DataFrame | None:
        """Query trade calendar."""
        conn = self._connect()
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "trade_cal" not in tables:
            return None

        s = _norm_date(start_date)
        e = _norm_date(end_date)

        sql = 'SELECT * FROM trade_cal WHERE cal_date >= ? AND cal_date <= ? AND is_open = ? ORDER BY cal_date DESC'
        df = conn.execute(sql, [s, e, is_open]).fetchdf()
        if df.empty:
            return None
        return df

    def get_stock_basic(
        self,
        ticker: str | None = None,
    ) -> pd.DataFrame | None:
        """Query stock basic info."""
        conn = self._connect()
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "stock_basic" not in tables:
            return None

        if ticker:
            df = conn.execute('SELECT * FROM stock_basic WHERE ts_code = ?', [ticker]).fetchdf()
        else:
            df = conn.execute('SELECT * FROM stock_basic').fetchdf()
        if df.empty:
            return None
        return df


# Singleton instance
_duckdb_store: DuckDBStore | None = None


def get_duckdb_store() -> DuckDBStore:
    """Get the global read-only DuckDB store instance."""
    global _duckdb_store
    if _duckdb_store is None:
        _duckdb_store = DuckDBStore()
    return _duckdb_store


def _norm_date(date_str: str) -> str:
    """Normalise 'YYYY-MM-DD' or 'YYYYMMDD' to 'YYYYMMDD'."""
    s = date_str.replace("-", "").strip()
    if len(s) == 8:
        return s
    raise ValueError(f"Invalid date format: {date_str}")
