"""JoinQuant-style data adapter using local DuckDB (Tushare data)."""

import os
from pathlib import Path

import duckdb
import pandas as pd


class JQDataAdapter:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.environ.get("TUSHARE_DB_PATH", "db/tushare_data.db")
        self.db_path = str(Path(db_path).resolve())
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path, read_only=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
    def to_yyyymmdd(date_str: str) -> str:
        return date_str.replace("-", "")

    def get_all_securities(self, date: str) -> pd.DataFrame:
        date_num = self.to_yyyymmdd(date)
        query = """
        SELECT ts_code, name, list_date
        FROM stock_basic
        WHERE list_status = 'L'
          AND (delist_date IS NULL OR delist_date = '')
          AND list_date <= ?
        """
        return self.conn.execute(query, [date_num]).fetchdf()

    _ALLOWED_EXTRA_FILTERS = {"report_type = 1"}

    def _batch_query(
        self,
        tickers: list[str],
        table: str,
        columns: list[str],
        end_date: str,
        limit: int,
        extra_filter: str = "",
        dedup_date_col: str = "ann_date",
    ) -> dict[str, pd.DataFrame]:
        if extra_filter and extra_filter not in self._ALLOWED_EXTRA_FILTERS:
            raise ValueError(f"Invalid extra_filter: {extra_filter!r}")
        if dedup_date_col not in {"ann_date", "f_ann_date"}:
            raise ValueError(f"Invalid dedup_date_col: {dedup_date_col!r}")
        if not tickers:
            return {}
        end_num = self.to_yyyymmdd(end_date)
        cols_str = ", ".join(columns)
        batch_size = 500
        all_dfs: list[pd.DataFrame] = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            placeholders = ",".join(["?" for _ in batch])
            filter_sql = f"AND {extra_filter}" if extra_filter else ""
            # Two-layer dedup: (1) keep latest revision per (ts_code, end_date)
            # by ordering on dedup_date_col DESC, then (2) take the most recent
            # `limit` end_dates per ts_code.
            query = f"""
            SELECT ts_code, {cols_str}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
                FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY ts_code, end_date
                        ORDER BY {dedup_date_col} DESC
                    ) as rn_dup
                    FROM {table}
                    WHERE ts_code IN ({placeholders})
                      AND end_date <= ?
                      AND {dedup_date_col} <= ?
                      {filter_sql}
                ) deduped
                WHERE rn_dup = 1
            ) sub
            WHERE rn <= ?
            ORDER BY ts_code, end_date DESC
            """
            params = batch + [end_num, end_num, limit]
            df = self.conn.execute(query, params).fetchdf()
            if not df.empty:
                all_dfs.append(df)
        if not all_dfs:
            return {}
        combined = pd.concat(all_dfs, ignore_index=True)
        result: dict[str, pd.DataFrame] = {}
        for ticker, group in combined.groupby("ts_code"):
            result[ticker] = group.drop(columns=["ts_code"]).reset_index(drop=True)
        return result

    def get_income_history(
        self, tickers: list[str], end_date: str, limit: int = 8
    ) -> dict[str, pd.DataFrame]:
        return self._batch_query(
            tickers,
            "income",
            [
                "end_date",
                "total_revenue",
                "revenue",
                "n_income_attr_p",
                "n_income",
                "operate_profit",
                "basic_eps",
                "oper_cost",
            ],
            end_date,
            limit,
            extra_filter="report_type = 1",
            dedup_date_col="f_ann_date",
        )

    def get_cashflow_history(
        self, tickers: list[str], end_date: str, limit: int = 8
    ) -> dict[str, pd.DataFrame]:
        return self._batch_query(
            tickers,
            "cashflow",
            ["end_date", "n_cashflow_act", "c_pay_acq_const_fiolta"],
            end_date,
            limit,
            extra_filter="report_type = 1",
            dedup_date_col="f_ann_date",
        )

    def get_balance_history(
        self, tickers: list[str], end_date: str, limit: int = 5
    ) -> dict[str, pd.DataFrame]:
        return self._batch_query(
            tickers,
            "balancesheet",
            [
                "end_date",
                "total_hldr_eqy_exc_min_int",
                "total_cur_assets",
                "total_cur_liab",
                "total_liab",
            ],
            end_date,
            limit,
            extra_filter="report_type = 1",
            dedup_date_col="f_ann_date",
        )

    def get_fina_indicator_history(
        self, tickers: list[str], end_date: str, limit: int = 24
    ) -> dict[str, pd.DataFrame]:
        return self._batch_query(
            tickers,
            "fina_indicator",
            [
                "end_date",
                "roe",
                "tr_yoy",
                "or_yoy",
                "netprofit_yoy",
                "dt_netprofit_yoy",
                "netprofit_margin",
                "grossprofit_margin",
                "profit_to_gr",
                "current_ratio",
                "debt_to_eqt",
                "eps",
                "basic_eps_yoy",
                "ocf_yoy",
                "equity_yoy",
                "bps_yoy",
                "fcff_ps",
            ],
            end_date,
            limit,
        )

    def get_valuation(self, tickers: list[str], date: str) -> pd.DataFrame:
        date_num = self.to_yyyymmdd(date)
        batch_size = 500
        all_dfs: list[pd.DataFrame] = []

        # Fallback to most recent trading day if exact date has no data
        effective_date = date_num
        check = self.conn.execute(
            "SELECT trade_date FROM daily_basic WHERE trade_date = ? LIMIT 1",
            [date_num],
        ).fetchone()
        if check is None:
            fallback = self.conn.execute(
                "SELECT MAX(trade_date) FROM daily_basic WHERE trade_date <= ?",
                [date_num],
            ).fetchone()
            if fallback and fallback[0]:
                effective_date = fallback[0]

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            placeholders = ",".join(["?" for _ in batch])
            query = f"""
            SELECT ts_code, pe, pe_ttm, pb, ps, ps_ttm, total_share
            FROM daily_basic
            WHERE ts_code IN ({placeholders})
              AND trade_date = ?
            """
            df = self.conn.execute(query, batch + [effective_date]).fetchdf()
            if not df.empty:
                all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def _apply_qfq(self, df: pd.DataFrame, end_date_num: str) -> pd.DataFrame:
        """用 adj_factor 将不复权价格转为前复权价格."""
        if df.empty:
            return df

        tickers = df["ts_code"].unique().tolist()
        batch_size = 500
        all_adj: list[pd.DataFrame] = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            placeholders = ",".join(["?" for _ in batch])
            query = f"""
            SELECT ts_code, trade_date, adj_factor
            FROM adj_factor
            WHERE ts_code IN ({placeholders})
              AND trade_date <= ?
            """
            adj_df = self.conn.execute(query, batch + [end_date_num]).fetchdf()
            if not adj_df.empty:
                all_adj.append(adj_df)

        if not all_adj:
            return df

        adj_combined = pd.concat(all_adj, ignore_index=True)

        # 每只股票在 end_date 的基准 adj_factor
        base_adj = (
            adj_combined.groupby("ts_code")["trade_date"]
            .max()
            .reset_index()
            .merge(adj_combined, on=["ts_code", "trade_date"], how="left")
            .rename(columns={"adj_factor": "base_factor"})
        )

        df = df.merge(
            adj_combined,
            left_on=["ts_code", "trade_date"],
            right_on=["ts_code", "trade_date"],
            how="left",
        )
        df = df.merge(
            base_adj[["ts_code", "base_factor"]], on="ts_code", how="left"
        )

        mask = (
            df["adj_factor"].notna()
            & df["base_factor"].notna()
            & (df["base_factor"] != 0)
        )
        ratio = df.loc[mask, "adj_factor"] / df.loc[mask, "base_factor"]
        for col in ["open", "high", "low", "close"]:
            df.loc[mask, col] = df.loc[mask, col] * ratio

        return df.drop(columns=["adj_factor", "base_factor"], errors="ignore")

    def get_prices(
        self, tickers: list[str], end_date: str, count: int = 130
    ) -> pd.DataFrame:
        date_num = self.to_yyyymmdd(end_date)
        batch_size = 500
        all_dfs: list[pd.DataFrame] = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            placeholders = ",".join(["?" for _ in batch])
            query = f"""
            SELECT ts_code, trade_date, open, high, low, close, vol as volume
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
                FROM daily
                WHERE ts_code IN ({placeholders})
                  AND trade_date <= ?
            ) sub
            WHERE rn <= ?
            ORDER BY ts_code, trade_date ASC
            """
            df = self.conn.execute(query, batch + [date_num, count]).fetchdf()
            if not df.empty:
                all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame()
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = self._apply_qfq(combined, date_num)
        combined["trade_date"] = pd.to_datetime(
            combined["trade_date"], format="%Y%m%d"
        )
        return combined

    def get_industry(self, tickers: list[str], date: str) -> dict[str, str]:
        date_num = self.to_yyyymmdd(date)
        batch_size = 500
        result: dict[str, str] = {}
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            placeholders = ",".join(["?" for _ in batch])
            query = f"""
            SELECT ts_code, industry
            FROM stock_basic
            WHERE ts_code IN ({placeholders})
            """
            df = self.conn.execute(query, batch).fetchdf()
            for _, row in df.iterrows():
                result[row["ts_code"]] = row["industry"] or "未知"
        return result
