"""JoinQuant-style 3-dimension stock screener using Tushare/DuckDB data."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.jq_adapter import JQDataAdapter
from src.backtesting.jq_indicators import adx, bbands, ema, rsi

WEIGHTS = {
    "fundamentals": 0.35,
    "growth": 0.40,
    "technical": 0.25,
}

COUNT_TECH = 130
BATCH_SIZE_FINA = 500
BATCH_SIZE_PRICE = 500


def _to_yyyymmdd(date_str: str) -> str:
    return date_str.replace("-", "")


def get_candidate_stocks(adapter: JQDataAdapter, date: str) -> list[str]:
    """Filter universe: exclude 科创/创业/北交/ST/停牌/新股."""
    prev_date = _prev_trade_date(adapter, date)
    df = adapter.get_all_securities(prev_date)
    if df.empty:
        return []

    df["code"] = df["ts_code"].str.split(".").str[0]
    df["name"] = df["name"].fillna("")

    prev_date_obj = datetime.strptime(prev_date, "%Y-%m-%d")

    # 停牌过滤：查询前一交易日有交易记录的股票
    prev_date_num = prev_date.replace("-", "")
    traded = set(
        r[0] for r in adapter.conn.execute(
            "SELECT DISTINCT ts_code FROM daily WHERE trade_date = ?", [prev_date_num]
        ).fetchall()
    )

    # ST过滤：用 namechange 历史名称（回测时点），而非 stock_basic 当前名称
    hist_names = adapter.conn.execute(
        """
        SELECT ts_code, name AS hist_name FROM (
            SELECT ts_code, name,
                   ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY start_date DESC) AS rn
            FROM namechange
            WHERE start_date <= ?
        ) WHERE rn = 1
        """,
        [prev_date_num],
    ).fetchdf()
    if not hist_names.empty:
        df = df.merge(hist_names, on="ts_code", how="left")
        df["check_name"] = df["hist_name"].fillna(df["name"])
    else:
        df["check_name"] = df["name"]

    # 新股过滤：与聚宽逻辑一致 (previous_date - start_date) < 365天
    list_date_obj = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    days_since_list = (prev_date_obj - list_date_obj).dt.days

    mask = (
        ~df["code"].str.startswith("68")  # 科创
        & ~df["code"].str.startswith("30")  # 创业板
        & ~df["code"].str.startswith("8")
        & ~df["code"].str.startswith("4")
        & ~df["code"].str.startswith("92")
        & ~df["code"].str.startswith("93")
        & ~df["code"].str.startswith("94")
        & ~df["check_name"].str.contains(r"ST|\*ST|退", case=False, na=False)
        & (days_since_list >= 365)
        & df["ts_code"].isin(traded)  # 排除停牌
    )
    return df.loc[mask, "ts_code"].tolist()


# ==================== 1. 基本面打分 ====================

def score_fundamentals(
    adapter: JQDataAdapter, tickers: list[str], date: str
) -> tuple[dict[str, float], dict[str, dict]]:
    # Raw data for manual TTM calculations
    income = adapter.get_income_history(tickers, date, limit=24)
    cashflow = adapter.get_cashflow_history(tickers, date, limit=24)
    balance = adapter.get_balance_history(tickers, date, limit=5)
    # Indicator for ROE and margins (need 5 periods to derive 4 quarterly ROEs)
    fina = adapter.get_fina_indicator_history(tickers, date, limit=5)
    prev_date = _prev_trade_date(adapter, date)
    val = adapter.get_valuation(tickers, prev_date)
    val_dict = {r["ts_code"]: r for _, r in val.iterrows()} if not val.empty else {}

    scores: dict[str, float] = {}
    details: dict[str, dict] = {}
    base_scores: dict[str, float] = {}
    sustainability_inputs: dict[str, tuple[float | None, float | None, float | None]] = {}

    # Query listing years for pre-IPO filtering (matching JQ)
    placeholders = ",".join(["?"] * len(tickers))
    list_dates_df = adapter.conn.execute(
        f"SELECT ts_code, list_date FROM stock_basic WHERE ts_code IN ({placeholders})",
        tickers,
    ).fetchdf()
    list_year_map: dict[str, int | None] = {}
    for _, r in list_dates_df.iterrows():
        ld = r["list_date"]
        if pd.notna(ld) and str(ld):
            list_year_map[r["ts_code"]] = int(str(ld)[:4])

    for t in tickers:
        list_year = list_year_map.get(t)
        df_inc = _filter_pre_ipo(income.get(t), list_year)
        df_cf = _filter_pre_ipo(cashflow.get(t), list_year)
        df_bal = _filter_pre_ipo(balance.get(t), list_year)
        df_f = fina.get(t)
        row_v = val_dict.get(t)

        # --- Profitability: fina_indicator with raw-data fallback ---
        row_f = df_f.iloc[0] if df_f is not None and not df_f.empty else None
        roe = _roe_ttm(df_f)
        net_margin = None
        op_margin = None

        if df_inc is not None and not df_inc.empty:
            rev_ttm_vals = _ttm_series(df_inc, "revenue")
            # 优先手动 TTM 计算利润率
            if rev_ttm_vals and abs(rev_ttm_vals[0]) > 1e-9:
                np_ttm_vals = _ttm_series(df_inc, "n_income")
                if np_ttm_vals:
                    net_margin = np_ttm_vals[0] / rev_ttm_vals[0]
                op_ttm_vals = _ttm_series(df_inc, "operate_profit")
                if op_ttm_vals:
                    op_margin = op_ttm_vals[0] / rev_ttm_vals[0]
        # fallback to fina_indicator cumulative margins
        if net_margin is None and row_f is not None:
            net_margin = _pct(row_f, "netprofit_margin")
        if op_margin is None and row_f is not None:
            op_margin = _pct(row_f, "profit_to_gr")

        if roe is None and df_inc is not None and not df_inc.empty and df_bal is not None and not df_bal.empty:
            np_ttm_vals = _ttm_series(df_inc, "n_income_attr_p")
            eq = df_bal.iloc[0].get("total_hldr_eqy_exc_min_int")
            if np_ttm_vals and pd.notna(eq) and eq != 0:
                roe = np_ttm_vals[0] / float(eq)

        # --- Health ratios from balance sheet ---
        cr = _latest_balance_ratio(df_bal, "total_cur_assets", "total_cur_liab")
        de = _latest_balance_ratio(df_bal, "total_liab", "total_hldr_eqy_exc_min_int")

        # --- EPS (latest single-quarter, matching JQ indicator.eps which is per-quarter) ---
        eps = None
        if df_inc is not None and not df_inc.empty:
            eps_sq = _to_single_quarter(df_inc, "basic_eps")
            if not eps_sq.empty:
                v = eps_sq.iloc[-1]
                if pd.notna(v):
                    eps = float(v)

        # --- FCF per share (TTM FCF / total shares) ---
        fcf_ps = None
        if df_cf is not None and not df_cf.empty:
            fcf_ttm_vals = _fcf_ttm_series(df_cf)
            if fcf_ttm_vals and row_v is not None:
                total_share = row_v.get("total_share")
                if total_share and total_share > 0:
                    fcf_ps = fcf_ttm_vals[0] / (total_share * 10000.0)

        # --- Valuation ---
        pe = _f(row_v, "pe_ttm") if row_v is not None else None
        pb = _f(row_v, "pb") if row_v is not None else None
        ps = _f(row_v, "ps_ttm") if row_v is not None else None

        score = 0.0

        # profitability (sub-module cap 0.40)
        prof_score = 0.0
        prof_count = 0
        if roe is not None and roe > 0.15:
            prof_score += 0.20
            prof_count += 1
        if net_margin is not None and net_margin > 0.20:
            prof_score += 0.10
            prof_count += 1
        if op_margin is not None and op_margin > 0.15:
            prof_score += 0.10
            prof_count += 1
        prof_score = min(prof_score, 0.40)
        score += prof_score
        prof_sig = "bullish" if prof_count >= 2 else "bearish" if prof_count == 0 else "neutral"

        # health
        health_score = 0
        if cr is not None and cr > 1.5:
            score += 0.10
            health_score += 1
        if de is not None and de < 0.5:
            score += 0.10
            health_score += 1
        if fcf_ps is not None and eps is not None and eps != 0 and fcf_ps > eps * 0.8:
            score += 0.10
            health_score += 1
        health_sig = "bullish" if health_score >= 2 else "bearish" if health_score == 0 else "neutral"

        # valuation (low = bullish)
        val_score = 0
        if pe is not None and 0 < pe < 25:
            score += 0.10
            val_score += 1
        if pb is not None and pb < 3:
            score += 0.10
            val_score += 1
        if ps is not None and ps < 5:
            score += 0.10
            val_score += 1
        val_sig = "bullish" if val_score >= 2 else "bearish" if val_score == 0 else "neutral"

        base_score = min(score, 1.0)
        base_scores[t] = base_score
        details[t] = {
            "profitability": {
                "roe": roe, "net_margin": net_margin, "op_margin": op_margin, "signal": prof_sig
            },
            "health": {
                "curr_ratio": cr, "de_ratio": de, "fcf_ps": fcf_ps, "eps": eps,
                "signal": health_sig
            },
            "valuation": {"pe": pe, "pb": pb, "ps": ps, "signal": val_sig},
            "overall": base_score,
        }

        # --- 可持续性指标采集 ---
        ocf_ttm_vals = _ttm_series(df_cf, "n_cashflow_act") if df_cf is not None else []
        np_ttm_vals = _ttm_series(df_inc, "n_income_attr_p") if df_inc is not None else []
        ttm_ocf = ocf_ttm_vals[0] if ocf_ttm_vals else None
        ttm_np = np_ttm_vals[0] if np_ttm_vals else None
        cf_ratio = ttm_ocf / ttm_np if ttm_np is not None and ttm_np > 0 and ttm_ocf is not None else None

        # 毛利率CV（聚宽使用单季毛利率最近4期，ddof=1）
        margin_cv = None
        if df_inc is not None and not df_inc.empty and "oper_cost" in df_inc.columns:
            rev_sq = _to_single_quarter(df_inc, "total_revenue")
            cost_sq = _to_single_quarter(df_inc, "oper_cost")
            if len(rev_sq) >= 4 and len(cost_sq) >= 4:
                gm_hist = []
                for i in range(len(rev_sq)):
                    r = float(rev_sq.iloc[i])
                    c = float(cost_sq.iloc[i]) if i < len(cost_sq) else None
                    if c is not None and abs(r) > 1e-9:
                        gm_hist.append((r - c) / r)
                clean_gm = [v for v in gm_hist if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if len(clean_gm) >= 4:
                    recent = clean_gm[-4:]
                    mean_gm = np.mean(recent)
                    if abs(mean_gm) > 1e-9:
                        margin_cv = np.std(recent, ddof=1) / abs(mean_gm)

        rev_growth = _ttm_yoy(df_inc, "total_revenue") if df_inc is not None else None
        sustainability_inputs[t] = (cf_ratio, margin_cv, rev_growth)

    # ==================== 可持续性因子 ====================
    all_cf_ratios = [v[0] for v in sustainability_inputs.values() if v[0] is not None]

    # 计算所有原始可持续性分
    sustainability_raws: dict[str, dict] = {}
    for t in tickers:
        cf_ratio, margin_cv, rev_growth = sustainability_inputs[t]

        # 1. 现金流质量（全市场百分位制）
        if cf_ratio is None:
            cf_score = 0.50
        else:
            cf_p30 = np.percentile(all_cf_ratios, 30)
            cf_p50 = np.percentile(all_cf_ratios, 50)
            cf_p70 = np.percentile(all_cf_ratios, 70)
            if cf_ratio >= cf_p70:
                cf_score = 1.00
            elif cf_ratio >= cf_p50:
                cf_score = 0.85
            elif cf_ratio >= cf_p30:
                cf_score = 0.70
            else:
                cf_score = 0.50

        # 2. 毛利率稳定性（CV阈值制）
        if margin_cv is None:
            cv_score = 0.80
        elif margin_cv < 0.15:
            cv_score = 1.00
        elif margin_cv < 0.30:
            cv_score = 0.80
        else:
            cv_score = 0.60

        # 3. 增长合理性（增长率区间制）
        if rev_growth is None:
            rg_score = 0.85
        elif rev_growth < 0:
            rg_score = 0.70
        elif rev_growth <= 0.30:
            rg_score = 1.00
        elif rev_growth <= 0.50:
            rg_score = 0.85
        elif rev_growth <= 1.00:
            rg_score = 0.70
        else:
            rg_score = 0.50

        raw = cf_score * cv_score * rg_score
        sustainability_raws[t] = {
            "cf_ratio": cf_ratio,
            "margin_cv": margin_cv,
            "rev_growth": rev_growth,
            "cf_score": cf_score,
            "cv_score": cv_score,
            "rg_score": rg_score,
            "raw": raw,
        }

    # 可持续性因子已禁用（factor 固定为 1.0）
    for t in tickers:
        info = sustainability_raws[t]
        info["factor"] = 1.0
        details[t]["sustainability"] = info
        scores[t] = base_scores[t]
        details[t]["overall"] = scores[t]

    return scores, details


# ==================== TTM helpers ====================

def _to_single_quarter(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Convert cumulative quarterly values to single-quarter values."""
    if df is None or df.empty or value_col not in df.columns:
        return pd.Series(dtype=float)
    df = df.sort_values("end_date", ascending=True).reset_index(drop=True)
    df = df.drop_duplicates(subset=["end_date"], keep="first").reset_index(drop=True)
    s = df[value_col].astype(float)
    months = pd.to_datetime(df["end_date"], format="%Y%m%d").dt.month
    sq = s.copy()
    non_q1 = months != 3
    sq.loc[non_q1] = s.diff().loc[non_q1]
    sq = sq.fillna(s)
    return sq


def _ttm_series(df: pd.DataFrame, value_col: str) -> list[float]:
    """Return TTM values as list (latest first)."""
    sq = _to_single_quarter(df, value_col)
    if len(sq) < 4:
        return []
    ttm = sq.rolling(window=4, min_periods=4).sum()
    values = [float(v) for v in ttm.dropna()]
    return values[::-1]


def _ttm_yoy(df: pd.DataFrame, value_col: str) -> float | None:
    """Compute latest TTM YoY growth."""
    values = _ttm_series(df, value_col)
    if len(values) < 5:
        return None
    current = values[0]
    prior = values[4]
    if abs(prior) < 1e-9:
        return None
    return (current - prior) / abs(prior)


def _sq_yoy(df: pd.DataFrame, value_col: str) -> float | None:
    """Compute latest single-quarter YoY growth (JQ style for revenue)."""
    sq = _to_single_quarter(df, value_col)
    if len(sq) < 5:
        return None
    current = float(sq.iloc[-1])
    prior = float(sq.iloc[-5])
    if abs(prior) < 1e-9:
        return None
    return (current - prior) / abs(prior)


def _filter_pre_ipo(df: pd.DataFrame | None, list_year: int | None) -> pd.DataFrame | None:
    """Filter out financial data from listing year and before (matching JQ)."""
    if df is None or df.empty or list_year is None:
        return df
    years = df["end_date"].astype(str).str[:4].astype(int)
    return df[years > list_year].copy()


def _fcf_ttm_series(df_cf: pd.DataFrame) -> list[float]:
    """Compute TTM FCF = TTM(OCF) - TTM(Capex), latest first."""
    if df_cf is None or df_cf.empty:
        return []
    df = df_cf.sort_values("end_date", ascending=True).reset_index(drop=True)
    ocf_sq = _to_single_quarter(df, "n_cashflow_act")
    capex_sq = _to_single_quarter(df, "c_pay_acq_const_fiolta")
    fcf_sq = ocf_sq - capex_sq.fillna(0)
    if len(fcf_sq) < 4:
        return []
    ttm = fcf_sq.rolling(window=4, min_periods=4).sum()
    return [float(v) for v in ttm.dropna()][::-1]


def _ttm_yoy_series(df: pd.DataFrame, value_col: str) -> list[float]:
    """Return TTM YoY growth rates as list (latest first)."""
    values = _ttm_series(df, value_col)
    if len(values) < 5:
        return []
    yoy_list: list[float] = []
    for i in range(len(values) - 4):
        current = values[i]
        prior = values[i + 4]
        if abs(prior) < 1e-9:
            yoy_list.append(float("nan"))
        else:
            yoy_list.append((current - prior) / abs(prior))
    return yoy_list


def _ocf_ttm_yoy(df_cf: pd.DataFrame) -> float | None:
    """OCF TTM YoY (cumulative → diff → rolling-4 sum → YoY), matching JQ.

    JoinQuant labels this metric as "FCF growth" but actually uses operating
    cash flow without subtracting capex (see docs/jq_tushare_diff.md 差异 4).
    """
    if df_cf is None or df_cf.empty:
        return None
    return _ttm_yoy(df_cf, "n_cashflow_act")


def _ocf_ttm_yoy_series(df_cf: pd.DataFrame) -> list[float]:
    """OCF TTM YoY series (latest first), matching JQ FCF growth semantics."""
    if df_cf is None or df_cf.empty:
        return []
    return _ttm_yoy_series(df_cf, "n_cashflow_act")


def _cumulative_yoy(df: pd.DataFrame, value_col: str) -> float | None:
    """Compute cumulative YoY growth (same quarter prior year), matching JQ indicator.eps style."""
    if df is None or df.empty or value_col not in df.columns:
        return None
    df = df.sort_values("end_date", ascending=False).reset_index(drop=True)
    df = df.drop_duplicates(subset=["end_date"], keep="first").reset_index(drop=True)
    if len(df) < 2:
        return None
    current = df.iloc[0][value_col]
    current_date = str(df.iloc[0]["end_date"])
    prior_year = str(int(current_date[:4]) - 1) + current_date[4:]
    prior_rows = df[df["end_date"] == prior_year]
    if prior_rows.empty:
        return None
    prior = prior_rows.iloc[0][value_col]
    if pd.isna(current) or pd.isna(prior) or abs(float(prior)) < 1e-9:
        return None
    return (float(current) - float(prior)) / abs(float(prior))


def _cumulative_yoy_series(df: pd.DataFrame, value_col: str) -> list[float]:
    """Return cumulative YoY growth rates as list (latest first)."""
    if df is None or df.empty or value_col not in df.columns:
        return []
    df = df.sort_values("end_date", ascending=False).reset_index(drop=True)
    df = df.drop_duplicates(subset=["end_date"], keep="first").reset_index(drop=True)
    yoy_list: list[float] = []
    for i in range(len(df)):
        current_date = str(df.iloc[i]["end_date"])
        current_val = df.iloc[i][value_col]
        prior_year = str(int(current_date[:4]) - 1) + current_date[4:]
        prior_rows = df[df["end_date"] == prior_year]
        if prior_rows.empty:
            yoy_list.append(float("nan"))
            continue
        prior_val = prior_rows.iloc[0][value_col]
        if pd.isna(current_val) or pd.isna(prior_val) or abs(float(prior_val)) < 1e-9:
            yoy_list.append(float("nan"))
        else:
            yoy_list.append((float(current_val) - float(prior_val)) / abs(float(prior_val)))
    return yoy_list


def _latest_balance_ratio(df_bal: pd.DataFrame, num_col: str, den_col: str) -> float | None:
    """Compute ratio from latest balance sheet row."""
    if df_bal is None or df_bal.empty:
        return None
    row = df_bal.iloc[0]
    num = row.get(num_col)
    den = row.get(den_col)
    if pd.isna(num) or pd.isna(den) or den == 0:
        return None
    return float(num) / float(den)


def _bv_growth(df_bal: pd.DataFrame) -> float | None:
    """Book value growth: latest equity vs same quarter prior year."""
    if df_bal is None or df_bal.empty:
        return None
    df = df_bal.sort_values("end_date", ascending=False).reset_index(drop=True)
    df = df.drop_duplicates(subset=["end_date"], keep="first").reset_index(drop=True)
    if len(df) < 2:
        return None
    latest = df.iloc[0]
    latest_date = str(latest["end_date"])
    prior_year_date = str(int(latest_date[:4]) - 1) + latest_date[4:]
    prior_rows = df[df["end_date"] == prior_year_date]
    if prior_rows.empty:
        if len(df) < 5:
            return None
        prior = df.iloc[4]
    else:
        prior = prior_rows.iloc[0]
    latest_eq = latest.get("total_hldr_eqy_exc_min_int")
    prior_eq = prior.get("total_hldr_eqy_exc_min_int")
    if pd.isna(latest_eq) or pd.isna(prior_eq) or prior_eq == 0:
        return None
    return (float(latest_eq) - float(prior_eq)) / abs(float(prior_eq))


# ==================== 2. 成长性打分 ====================

def _trend_slope(series: list[float | None]) -> float:
    clean = [v for v in series if v is not None and not (isinstance(v, float) and np.isnan(v))]
    clean = clean[:12]
    if len(clean) < 2:
        return 0.0
    y = clean[::-1]
    x = list(range(len(y)))
    n = len(y)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(i * j for i, j in zip(x, y))
    sum_x2 = sum(i * i for i in x)
    try:
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope * 100
    except ZeroDivisionError:
        return 0.0


def score_growth(
    adapter: JQDataAdapter, tickers: list[str], date: str
) -> tuple[dict[str, float], dict[str, dict]]:
    # Raw financial data for TTM calculation
    income = adapter.get_income_history(tickers, date, limit=24)
    cashflow = adapter.get_cashflow_history(tickers, date, limit=24)
    balance = adapter.get_balance_history(tickers, date, limit=5)
    # Fallback data from fina_indicator
    fina = adapter.get_fina_indicator_history(tickers, date, limit=12)
    prev_date = _prev_trade_date(adapter, date)
    val = adapter.get_valuation(tickers, prev_date)
    val_dict = {r["ts_code"]: r for _, r in val.iterrows()} if not val.empty else {}

    scores: dict[str, float] = {}
    details: dict[str, dict] = {}

    # Query listing years for pre-IPO filtering (matching JQ)
    placeholders = ",".join(["?"] * len(tickers))
    list_dates_df = adapter.conn.execute(
        f"SELECT ts_code, list_date FROM stock_basic WHERE ts_code IN ({placeholders})",
        tickers,
    ).fetchdf()
    list_year_map: dict[str, int | None] = {}
    for _, r in list_dates_df.iterrows():
        ld = r["list_date"]
        if pd.notna(ld) and str(ld):
            list_year_map[r["ts_code"]] = int(str(ld)[:4])

    for t in tickers:
        list_year = list_year_map.get(t)
        df_inc = _filter_pre_ipo(income.get(t), list_year)
        df_cf = _filter_pre_ipo(cashflow.get(t), list_year)
        df_bal = _filter_pre_ipo(balance.get(t), list_year)
        df_f = fina.get(t)

        # --- Growth metrics ---
        rev_growth = _ttm_yoy(df_inc, "total_revenue") if df_inc is not None else None
        eps_growth = _ttm_yoy(df_inc, "basic_eps") if df_inc is not None else None
        fcf_growth = _ocf_ttm_yoy(df_cf) if df_cf is not None else None
        np_growth = _ttm_yoy(df_inc, "n_income_attr_p") if df_inc is not None else None
        bv_growth = _bv_growth(df_bal) if df_bal is not None else None

        rev_yoy = _ttm_yoy_series(df_inc, "total_revenue") if df_inc is not None else []
        eps_yoy = _ttm_yoy_series(df_inc, "basic_eps") if df_inc is not None else []
        fcf_yoy = _ocf_ttm_yoy_series(df_cf) if df_cf is not None else []

        # JQ uses up to 12-period history for trends (_trend_slope limits internally)
        rev_trend = _trend_slope(rev_yoy)
        eps_trend = _trend_slope(eps_yoy)
        fcf_trend = _trend_slope(fcf_yoy)

        # --- TTM OCF and NP for cf_quality ---
        ocf_ttm_vals = _ttm_series(df_cf, "n_cashflow_act") if df_cf is not None else []
        np_ttm_vals = _ttm_series(df_inc, "n_income_attr_p") if df_inc is not None else []
        ttm_ocf = ocf_ttm_vals[0] if ocf_ttm_vals else None
        ttm_np = np_ttm_vals[0] if np_ttm_vals else None

        # --- Margins: TTM latest + cumulative history ---
        gm, nm, om = None, None, None
        gm_hist, om_hist, nm_hist = [], [], []

        if df_inc is not None and not df_inc.empty:
            # Margins from TTM (matching JQ which uses TTM margins for trends)
            rev_ttm_vals = _ttm_series(df_inc, "revenue")
            cost_ttm_vals = _ttm_series(df_inc, "oper_cost")
            op_ttm_vals = _ttm_series(df_inc, "operate_profit")
            np_ttm_vals = _ttm_series(df_inc, "n_income")
            if rev_ttm_vals:
                rev_latest = rev_ttm_vals[0]
                for i in range(len(rev_ttm_vals)):
                    rev = rev_ttm_vals[i]
                    if abs(rev) > 1e-9:
                        if cost_ttm_vals and i < len(cost_ttm_vals):
                            gm_hist.append((rev - cost_ttm_vals[i]) / rev)
                        if op_ttm_vals and i < len(op_ttm_vals):
                            om_hist.append(op_ttm_vals[i] / rev)
                        if np_ttm_vals and i < len(np_ttm_vals):
                            nm_hist.append(np_ttm_vals[i] / rev)
                if cost_ttm_vals and abs(rev_latest) > 1e-9:
                    gm = (rev_latest - cost_ttm_vals[0]) / rev_latest
                if op_ttm_vals and abs(rev_latest) > 1e-9:
                    om = op_ttm_vals[0] / rev_latest
                if np_ttm_vals and abs(rev_latest) > 1e-9:
                    nm = np_ttm_vals[0] / rev_latest

        if gm is None and df_f is not None and not df_f.empty:
            gm = _pct_row(df_f, 0, "grossprofit_margin")
        if nm is None and df_f is not None and not df_f.empty:
            nm = _pct_row(df_f, 0, "netprofit_margin")
        if om is None and df_f is not None and not df_f.empty:
            om = _pct_row(df_f, 0, "profit_to_gr")

        if not gm_hist and df_f is not None and not df_f.empty:
            gm_hist = [_pct_row(df_f, i, "grossprofit_margin") for i in range(len(df_f))]
            om_hist = [_pct_row(df_f, i, "profit_to_gr") for i in range(len(df_f))]
            nm_hist = [_pct_row(df_f, i, "netprofit_margin") for i in range(len(df_f))]

        gm_trend = _trend_slope(gm_hist)
        om_trend = _trend_slope(om_hist)
        nm_trend = _trend_slope(nm_hist)

        # --- Valuation ---
        row_v = val_dict.get(t)
        pe = _f(row_v, "pe_ttm") if row_v is not None else None
        ps = _f(row_v, "ps_ttm") if row_v is not None else None

        # PEG = PE / (TTM 归母净利润同比增长 × 100), growth capped at 200%
        peg = None
        if pe is not None and np_growth is not None and np_growth > 0:
            np_growth_capped = min(np_growth * 100, 200.0)
            peg = pe / np_growth_capped
            if peg < 0.01:
                peg = None

        # --- Sub-scores ---
        # 1. Growth trends
        growth_score = 0.0
        if rev_growth is not None:
            if rev_growth > 0.20:
                growth_score += 0.25
            elif rev_growth > 0.10:
                growth_score += 0.15
            if rev_trend > 0:
                growth_score += 0.10
        if eps_growth is not None:
            if eps_growth > 0.20:
                growth_score += 0.25
            elif eps_growth > 0.10:
                growth_score += 0.15
            if eps_trend > 0:
                growth_score += 0.10
        if fcf_growth is not None and fcf_growth > 0.15:
            growth_score += 0.25
        if bv_growth is not None and bv_growth > 0.10:
            growth_score += 0.10
        growth_score = min(growth_score, 1.0)

        # 2. Valuation (PEG only)
        val_score = 0.0
        if peg is not None:
            if peg < 1.0:
                val_score += 0.50
            elif peg < 2.0:
                val_score += 0.25
        val_score = min(val_score, 1.0)

        # 3. Margin trends
        margin_score = 0.0
        if gm is not None:
            if gm > 0.5:
                margin_score += 0.20
            if gm_trend > 0:
                margin_score += 0.20
        if om is not None:
            if om > 0.15:
                margin_score += 0.20
            if om_trend > 0:
                margin_score += 0.20
        if nm is not None and nm_trend > 0:
            margin_score += 0.20
        margin_score = min(margin_score, 1.0)

        # Weighted composite (growth=0.50, val=0.25, margin=0.25)
        raw_score = (
            growth_score * 0.50
            + val_score * 0.25
            + margin_score * 0.25
        )
        score = raw_score
        scores[t] = score

        signal = "bullish" if score > 0.6 else "bearish" if score < 0.4 else "neutral"
        confidence = round(abs(score - 0.5) * 2 * 100)

        details[t] = {
            "signal": signal,
            "confidence": confidence,
            "weighted_score": round(score, 4),
            "raw_score": round(raw_score, 4),
            "revenue_growth": rev_growth,
            "revenue_trend": rev_trend,
            "eps_growth": eps_growth,
            "eps_trend": eps_trend,
            "fcf_growth": fcf_growth,
            "fcf_trend": fcf_trend,
            "bv_growth": bv_growth,
            "peg": peg,
            "ps": ps,
            "gross_margin": gm,
            "gm_trend": gm_trend,
            "operating_margin": om,
            "om_trend": om_trend,
            "net_margin": nm,
            "nm_trend": nm_trend,
            "growth_score": growth_score,
            "val_score": val_score,
            "margin_score": margin_score,
        }

    return scores, details


def _hurst_rs(ts):
    """重标极差法 (R/S)"""
    ts = np.array(ts)
    n = len(ts)
    lags = [2**i for i in range(1, int(np.log2(n)))]
    rs_vals = []
    for lag in lags:
        if lag >= n:
            break
        k = n // lag
        rs_sub = []
        for i in range(k):
            sub = ts[i*lag:(i+1)*lag]
            if len(sub) < 2:
                continue
            mean_sub = np.mean(sub)
            dev = sub - mean_sub
            cumdev = np.cumsum(dev)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(sub)
            if s != 0:
                rs_sub.append(r / s)
        if rs_sub:
            rs_vals.append(np.mean(rs_sub))
    if len(rs_vals) < 2:
        return np.nan
    log_lags = np.log(lags[:len(rs_vals)])
    log_rs = np.log(rs_vals)
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    return slope


# ==================== 3. 技术面打分 ====================

def score_technical(
    adapter: JQDataAdapter, tickers: list[str], date: str
) -> tuple[dict[str, float], dict[str, dict]]:
    tech_date = _prev_trade_date(adapter, date)
    df_prices = adapter.get_prices(tickers, tech_date, count=COUNT_TECH)
    if df_prices.empty:
        return {}, {}

    scores: dict[str, float] = {}
    details: dict[str, dict] = {}

    for t, sub in df_prices.groupby("ts_code"):
        if len(sub) < 60:
            continue
        sub = sub.sort_values("trade_date")
        close = sub["close"].values
        high = sub["high"].values
        low = sub["low"].values

        s_close = sub["close"].reset_index(drop=True)
        s_high = sub["high"].reset_index(drop=True)
        s_low = sub["low"].reset_index(drop=True)

        adx_latest = adx(s_high, s_low, s_close, period=14).iloc[-1]
        ema8 = ema(s_close, 8).iloc[-1]
        ema21 = ema(s_close, 21).iloc[-1]
        ema55 = ema(s_close, 55).iloc[-1]

        trend_strength = adx_latest / 100.0 if pd.notna(adx_latest) else 0.0
        ema_valid = pd.notna(ema8) and pd.notna(ema21) and pd.notna(ema55)
        ema_bullish = ema_valid and ema8 > ema21 and ema21 > ema55
        ema_bearish = ema_valid and ema8 < ema21 and ema21 < ema55

        if adx_latest > 25 and ema_bullish:
            tf_bullish = 0.6 + trend_strength * 0.4
        elif adx_latest > 25 and ema_bearish:
            tf_bullish = 0.4 - trend_strength * 0.4
        else:
            tf_bullish = 0.5

        rsi14 = rsi(s_close, 14).iloc[-1]
        upper, middle, lower = bbands(s_close, 20, 2.0)
        bb_width = upper.iloc[-1] - lower.iloc[-1]
        price_vs_bb = (
            (close[-1] - lower.iloc[-1]) / bb_width * 100.0
            if pd.notna(bb_width) and bb_width != 0
            else 50.0
        )

        if price_vs_bb < 20 or rsi14 < 30:
            mr_bullish = 1.0
        elif price_vs_bb > 80 or rsi14 > 70:
            mr_bullish = 0.0
        else:
            mr_bullish = 0.5

        mom_3m = close[-1] / close[-60] - 1.0 if len(close) >= 60 else 0.0
        mom_6m = close[-1] / close[-120] - 1.0 if len(close) >= 120 else 0.0

        if mom_3m > 0 and mom_6m > 0:
            mom_bullish = 1.0
        elif mom_3m < 0 and mom_6m < 0:
            mom_bullish = 0.0
        else:
            mom_bullish = 0.5

        returns = s_close.pct_change().dropna()
        vol_regime = np.nan
        if len(returns) >= 20:
            hist_vol = returns.iloc[-20:].std() * np.sqrt(252)
            vol_ma = (
                returns.rolling(20).std().iloc[-63:].mean() * np.sqrt(252)
                if len(returns) >= 63
                else hist_vol
            )
            vol_regime = hist_vol / vol_ma if vol_ma != 0 else 1.0
            if vol_regime < 0.8:
                vol_bullish = 1.0
            elif vol_regime > 1.2:
                vol_bullish = 0.0
            else:
                vol_bullish = 0.5
        else:
            vol_bullish = 0.5

        w = {"tf": 0.20, "mr": 0.25, "mom": 0.35, "vol": 0.20}
        bullish = (
            tf_bullish * w["tf"]
            + mr_bullish * w["mr"]
            + mom_bullish * w["mom"]
            + vol_bullish * w["vol"]
        )
        scores[t] = bullish
        details[t] = {
            "adx": adx_latest,
            "ema8": ema8,
            "ema21": ema21,
            "ema55": ema55,
            "tf_bullish": tf_bullish,
            "rsi14": rsi14,
            "price_vs_bb": price_vs_bb,
            "mr_bullish": mr_bullish,
            "mom_3m": mom_3m,
            "mom_6m": mom_6m,
            "mom_bullish": mom_bullish,
            "vol_regime": vol_regime,
            "vol_bullish": vol_bullish,
            "overall": bullish,
        }

    return scores, details


# ==================== 综合 ====================

def combine_scores(
    fund_scores: dict[str, float],
    growth_scores: dict[str, float],
    tech_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    w = weights or WEIGHTS
    all_stocks = set(fund_scores) | set(growth_scores) | set(tech_scores)
    combined: dict[str, float] = {}
    for t in all_stocks:
        f = fund_scores.get(t, 0.5)
        g = growth_scores.get(t, 0.5)
        te = tech_scores.get(t, 0.5)
        combined[t] = f * w["fundamentals"] + g * w["growth"] + te * w["technical"]
    return combined


def _prev_trade_date(adapter: JQDataAdapter, date: str) -> str:
    """Return previous trading day via Tushare trade_cal table."""
    d = date.replace("-", "")
    row = adapter.conn.execute(
        "SELECT pretrade_date FROM trade_cal WHERE exchange = 'SSE' AND cal_date = ?",
        [d],
    ).fetchone()
    if row:
        p = str(row[0])
        return f"{p[:4]}-{p[4:6]}-{p[6:]}"
    return date


def run_screener(
    adapter: JQDataAdapter,
    date: str,
    top_n: int = 10,
    max_per_industry: int = 2,
    weights: dict[str, float] | None = None,
    min_combined_score: float = 0.70,
    min_tech_score: float = 0.35,
) -> tuple[list[tuple[str, float]], dict]:
    """Run full 3-dimension screener for a single date."""
    candidates = get_candidate_stocks(adapter, date)
    if not candidates:
        return [], {}

    fund_scores, fund_details = score_fundamentals(adapter, candidates, date)
    growth_scores, growth_details = score_growth(adapter, candidates, date)
    tech_scores, tech_details = score_technical(adapter, candidates, date)

    combined = combine_scores(fund_scores, growth_scores, tech_scores, weights)

    # Hard filters disabled for JQ alignment comparison
    filtered = combined

    sorted_stocks = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    industries = adapter.get_industry([s for s, _ in sorted_stocks], date)

    target_list: list[tuple[str, float]] = []
    industry_count: dict[str, int] = {}
    for t, score in sorted_stocks:
        # min_combined_score filter disabled for JQ alignment
        ind = industries.get(t, "未知")
        if industry_count.get(ind, 0) >= max_per_industry:
            continue
        target_list.append((t, score))
        industry_count[ind] = industry_count.get(ind, 0) + 1
        if len(target_list) >= top_n:
            break

    details = {
        "fundamentals": fund_details,
        "growth": growth_details,
        "technical": tech_details,
        "combined": combined,
        "industry_count": industry_count,
    }
    return target_list, details


# ==================== 辅助函数 ====================

def _f(row: pd.Series | None, col: str) -> float | None:
    if row is None:
        return None
    v = row.get(col)
    if pd.isna(v):
        return None
    return float(v)


def _pct(row: pd.Series | None, col: str) -> float | None:
    """Read a percentage column and convert to decimal (e.g. 15.2 -> 0.152)."""
    v = _f(row, col)
    return v / 100.0 if v is not None else None


def _roe_ttm(df_f: pd.DataFrame | None) -> float | None:
    """Calculate TTM ROE from fina_indicator cumulative ROE values.

    Tushare fina_indicator.roe is cumulative (year-to-date for quarterly
    reports).  We decompose into single-quarter ROE and sum the latest 4.
    """
    if df_f is None or df_f.empty:
        return None
    df = df_f.copy()
    df["end_date"] = df["end_date"].astype(str)
    df = df.sort_values("end_date", ascending=True).reset_index(drop=True)

    quarterly_roes: list[float] = []
    for i, row in df.iterrows():
        roe = _pct(row, "roe")
        if roe is None:
            continue
        curr_year = str(row["end_date"])[:4]
        prev_roe = None
        for j in range(i - 1, -1, -1):
            if str(df.iloc[j]["end_date"])[:4] == curr_year:
                prev_roe = _pct(df.iloc[j], "roe")
                break
        sq_roe = roe - prev_roe if prev_roe is not None else roe
        quarterly_roes.append(sq_roe)

    if len(quarterly_roes) >= 4:
        return sum(quarterly_roes[-4:])
    return None


def _f_row(df: pd.DataFrame, idx: int, col: str) -> float | None:
    if df is None or idx >= len(df):
        return None
    v = df.iloc[idx].get(col)
    if pd.isna(v):
        return None
    return float(v)


def _pct_row(df: pd.DataFrame, idx: int, col: str) -> float | None:
    v = _f_row(df, idx, col)
    return v / 100.0 if v is not None else None


# ==================== CLI ====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="JQ-style 3-dimension screener")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--db-path", default="src/data/tushare_data.db")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    adapter = JQDataAdapter()
    results, details = run_screener(adapter, args.date, top_n=args.top_n)

    print(f"Date: {args.date} | Candidates: {len(results)}")
    print("=" * 60)
    for i, (t, score) in enumerate(results, 1):
        fd = details["fundamentals"].get(t, {})
        gd = details["growth"].get(t, {})
        td = details["technical"].get(t, {})
        print(
            f"{i:2d}. {t} 综合={score:.4f}  "
            f"基本面={fd.get('overall', 0):.4f}  "
            f"成长={gd.get('weighted_score', 0):.4f}  "
            f"技术={td.get('overall', 0):.4f}"
        )

    if args.output:
        out = {
            "date": args.date,
            "results": [{"ticker": t, "score": round(s, 4)} for t, s in results],
            "details": {
                k: {t: v for t, v in d.items() if t in [x[0] for x in results]}
                for k, d in details.items()
                if k != "combined"
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.output}")

    adapter.close()


if __name__ == "__main__":
    main()
