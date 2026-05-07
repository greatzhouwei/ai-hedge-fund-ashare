import datetime
import logging
import math
import os
import time
import pandas as pd

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.duckdb_store import get_duckdb_store

# 绕过系统代理，直接访问 Tushare（国内服务无需代理）
for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    if _proxy_key in os.environ:
        del os.environ[_proxy_key]
        logger.info("Removed %s for direct Tushare access", _proxy_key)
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
    NorthboundHolding,
    MarginData,
)

# Global cache instance
_cache = get_cache()

# Tushare Pro API (lazy init)
_pro_api = None


def _get_pro_api(api_key: str | None = None):
    """Get or initialize Tushare Pro API."""
    global _pro_api
    if _pro_api is None:
        try:
            import tushare as ts
        except ImportError as e:
            raise ImportError("tushare is required. Install it with: poetry install") from e
        # 清除代理环境变量，Tushare 直接访问无需代理
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
            os.environ.pop(key, None)
        os.environ["NO_PROXY"] = "api.waditu.com"
        token = api_key or os.environ.get("TUSHARE_TOKEN")
        if not token:
            raise ValueError("TUSHARE_TOKEN is not set. Please set it in your .env file.")
        _pro_api = ts.pro_api(token)
    return _pro_api


# Module-level caches and rate limiter
_trade_cal_cache: dict[str, pd.DataFrame] = {}

# Endpoints with strict rate limits (~200/min on free tier)
_RATE_LIMITED_ENDPOINTS = frozenset(
    {"fina_indicator", "balancesheet", "income", "cashflow", "hk_hold", "margin_detail", "stk_holdertrade", "dividend", "index_daily"}
)


def _call_tushare(endpoint: str, pro, max_retries: int = 2, delay: float = 0.8, **kwargs) -> pd.DataFrame | None:
    """Call a Tushare API endpoint with retry and rate-limiting sleep.

    Tushare free tier restricts several endpoints to ~200 requests/min.
    When concurrent workers hit the API, intermittent empty responses are
    returned instead of exceptions. We retry empty responses and add a
    small sleep after restrictive endpoints to stay under the limit.
    """
    import tushare as ts

    func = getattr(pro, endpoint, None)
    if func is None:
        logger.warning("Unknown Tushare endpoint: %s", endpoint)
        return None

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            df = func(**kwargs)
        except Exception as e:
            last_exception = e
            # Rate-limit error messages vary; back off and retry
            logger.warning("Tushare %s attempt %d failed: %s", endpoint, attempt + 1, e)
            if attempt < max_retries:
                time.sleep(delay * (attempt + 1))
            continue

        # Tushare sometimes returns empty DataFrame under rate pressure
        if df is not None and not df.empty:
            if endpoint in _RATE_LIMITED_ENDPOINTS:
                time.sleep(0.15)
            return df

        # Empty or None → possible rate limit, retry
        if attempt < max_retries:
            logger.warning("Tushare %s returned empty on attempt %d, retrying...", endpoint, attempt + 1)
            time.sleep(delay * (attempt + 1))

    if last_exception is not None:
        logger.warning("Tushare %s exhausted retries: %s", endpoint, last_exception)
    else:
        logger.warning("Tushare %s returned empty after %d attempts", endpoint, max_retries + 1)
    return None


def _to_tushare_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYYMMDD."""
    return date_str.replace("-", "")


def _from_tushare_date(date_str: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


# Mapping from English line-item names to Tushare (table, field)
LINE_ITEM_MAPPING = {
    # Income statement (利润表) - pro.income
    "revenue": ("income", "revenue"),
    "total_revenue": ("income", "total_revenue"),
    "operating_revenue": ("income", "revenue"),
    "operating_income": ("income", "operate_profit"),
    "operating_profit": ("income", "operate_profit"),
    "ebitda": ("income", "ebitda"),
    "net_income": ("income", "n_income"),
    "net_income_attributable": ("income", "n_income_attr_p"),
    "research_and_development": ("income", "rd_exp"),
    "interest_expense": ("income", "int_exp"),
    "income_tax_expense": ("income", "income_tax"),
    "total_operating_expenses": ("income", "total_cogs"),
    "operating_expense": ("income", "oper_exp"),
    "operating_expenses": ("income", "oper_exp"),
    "selling_general_and_administrative": ("income", "admin_exp"),
    "depreciation_and_amortization": ("cashflow", "depr_fa_coga_dpba"),  # 固定资产折旧、油气资产折耗、生产性生物资产折旧
    "earnings_per_share": ("income", "basic_eps"),
    # Balance sheet (资产负债表) - pro.balancesheet
    "total_assets": ("balancesheet", "total_assets"),
    "book_value_per_share": ("balancesheet", "bps"),
    "total_liabilities": ("balancesheet", "total_liab"),
    "shareholders_equity": ("balancesheet", "total_hldr_eqy_exc_min_int"),
    "total_equity": ("balancesheet", "total_hldr_eqy_exc_min_int"),
    "outstanding_shares": ("balancesheet", "total_share"),
    "total_debt": ("balancesheet", "total_liab"),
    "cash_and_equivalents": ("balancesheet", "money_cap"),
    "current_assets": ("balancesheet", "total_cur_assets"),
    "current_liabilities": ("balancesheet", "total_cur_liab"),
    "inventory": ("balancesheet", "inventories"),
    "accounts_receivable": ("balancesheet", "accounts_receiv"),
    "property_plant_equipment": ("balancesheet", "fix_assets"),
    "goodwill": ("balancesheet", "goodwill"),
    "long_term_debt": ("balancesheet", "lt_borr"),
    "short_term_debt": ("balancesheet", "st_borr"),
    # Cash flow (现金流量表) - pro.cashflow
    "free_cash_flow": ("cashflow", "free_cashflow"),
    "operating_cash_flow": ("cashflow", "n_cashflow_act"),
    "capital_expenditure": ("cashflow", "c_pay_acq_const_fiolta"),
    "issuance_or_purchase_of_equity_shares": ("cashflow", "c_recp_cap_contrib"),
    "cash_flow_from_operations": ("cashflow", "n_cashflow_act"),
    "cash_flow_from_investing": ("cashflow", "n_cashflow_inv_act"),
    "cash_flow_from_financing": ("cashflow", "n_cash_frd_act"),
}


def _df_to_records(df: pd.DataFrame | None) -> list[dict]:
    """Convert a DataFrame to a list of dicts, handling None/empty."""
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from local DuckDB or Tushare API."""
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    # Try local DuckDB first
    try:
        store = get_duckdb_store()
        df = store.get_daily(ticker, start_date, end_date)
        if df is not None and not df.empty:
            records = _df_to_records(df)
            prices = []
            for r in records:
                prices.append(
                    Price(
                        open=float(r.get("open", 0)),
                        close=float(r.get("close", 0)),
                        high=float(r.get("high", 0)),
                        low=float(r.get("low", 0)),
                        volume=int(r.get("vol", 0)),
                        time=_from_tushare_date(str(r.get("trade_date", ""))),
                    )
                )
            _cache.set_prices(cache_key, [p.model_dump() for p in prices])
            return prices
    except Exception:
        pass

    pro = _get_pro_api(api_key)
    df = _call_tushare(
        "daily",
        pro,
        ts_code=ticker,
        start_date=_to_tushare_date(start_date),
        end_date=_to_tushare_date(end_date),
    )
    if df is None:
        return []

    records = _df_to_records(df)
    if not records:
        return []

    prices = []
    for r in records:
        prices.append(
            Price(
                open=float(r.get("open", 0)),
                close=float(r.get("close", 0)),
                high=float(r.get("high", 0)),
                low=float(r.get("low", 0)),
                volume=int(r.get("vol", 0)),
                time=_from_tushare_date(str(r.get("trade_date", ""))),
            )
        )

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from Tushare (fina_indicator + daily_basic)."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    pro = _get_pro_api(api_key)
    ts_end = _to_tushare_date(end_date)

    # fina_indicator 提供大部分财务指标
    # 多请求 8 条作为去重冗余，避免 Tushare 返回重复数据导致去重后不足 limit
    fina_df = None
    try:
        store = get_duckdb_store()
        fina_df = store.get_fina_indicator(ticker, end_date, limit=limit + 8)
    except Exception:
        pass
    if fina_df is None:
        fina_df = _call_tushare("fina_indicator", pro, ts_code=ticker, end_date=ts_end, limit=limit + 8)
    if fina_df is not None and not fina_df.empty:
        # 过滤掉尚未披露的数据（ann_date > 查询日期），防止 look-ahead bias
        if "ann_date" in fina_df.columns:
            fina_df = fina_df[fina_df["ann_date"] <= ts_end]
        # 优先保留 update_flag=1（最新修正版），再按 end_date 去重
        if "update_flag" in fina_df.columns:
            fina_df = fina_df.sort_values("update_flag", ascending=False)
        fina_df = fina_df.drop_duplicates(subset=["end_date"]).sort_values("end_date", ascending=False)
    else:
        fina_df = pd.DataFrame()

    # daily_basic 提供估值指标（trade_date 必须是交易日）
    # 先用 trade_cal 找到最近交易日，避免非交易日返回空
    # 往前查 60 天，确保长假期间也能覆盖到真实最近交易日
    basic_df = pd.DataFrame()
    cal_start = (datetime.datetime.strptime(ts_end, "%Y%m%d") - datetime.timedelta(days=60)).strftime("%Y%m%d")
    cal_key = f"{cal_start}_{ts_end}"
    cal_df = _trade_cal_cache.get(cal_key)
    if cal_df is None:
        # Try local DuckDB first
        try:
            store = get_duckdb_store()
            cal_df = store.get_trade_cal(cal_start, ts_end, "1")
        except Exception:
            cal_df = None
        if cal_df is None:
            cal_df = _call_tushare("trade_cal", pro, start_date=cal_start, end_date=ts_end, is_open="1")
        if cal_df is not None:
            _trade_cal_cache[cal_key] = cal_df
    if cal_df is not None and not cal_df.empty:
        latest_trade_date = str(cal_df.iloc[0]["cal_date"])
        basic_df = None
        try:
            store = get_duckdb_store()
            basic_df = store.get_daily_basic(ticker, latest_trade_date)
        except Exception:
            pass
        if basic_df is None:
            basic_df = _call_tushare("daily_basic", pro, ts_code=ticker, trade_date=latest_trade_date)
    if basic_df is None:
        basic_df = pd.DataFrame()

    # 将 daily_basic 按 trade_date 转为字典方便合并
    basic_map = {}
    for r in _df_to_records(basic_df):
        basic_map[str(r.get("trade_date", ""))] = r

    # 获取 balancesheet 数据（用于计算 enterprise_value）
    bs_df = None
    try:
        store = get_duckdb_store()
        bs_df = store.get_balancesheet(
            ticker,
            end_date,
            limit=limit + 8,
            fields="ts_code,end_date,total_liab,money_cap,total_share,total_hldr_eqy_exc_min_int",
        )
    except Exception:
        pass
    if bs_df is None:
        bs_df = _call_tushare(
            "balancesheet",
            pro,
            ts_code=ticker,
            end_date=ts_end,
            limit=limit + 8,
            fields="ts_code,end_date,total_liab,money_cap,total_share,total_hldr_eqy_exc_min_int",
        )
    if bs_df is not None and not bs_df.empty:
        bs_df = bs_df.drop_duplicates(subset=["end_date"]).sort_values("end_date", ascending=False)
    else:
        bs_df = pd.DataFrame()

    bs_map = {}
    for r in _df_to_records(bs_df):
        bs_map[str(r.get("end_date", ""))] = r

    # 获取 income 数据用于手动计算 EPS YoY 和单季度营收 YoY
    #（Tushare fina_indicator 的 basic_eps_yoy / q_sales_yoy 与 income 口径存在差异）
    income_df = None
    try:
        store = get_duckdb_store()
        income_df = store.get_income(
            ticker,
            end_date,
            limit=limit + 12,
            fields="ts_code,end_date,basic_eps,total_revenue,n_income_attr_p",
        )
    except Exception:
        pass
    if income_df is None:
        income_df = _call_tushare(
            "income",
            pro,
            ts_code=ticker,
            end_date=ts_end,
            limit=limit + 12,
            fields="ts_code,end_date,basic_eps,total_revenue,n_income_attr_p",
        )
    if income_df is not None and not income_df.empty:
        income_df = income_df.drop_duplicates(subset=["end_date"]).sort_values("end_date", ascending=False)
    else:
        income_df = pd.DataFrame()

    income_map = {}
    for r in _df_to_records(income_df):
        income_map[str(r.get("end_date", ""))] = r

    # 获取 cashflow 数据用于计算 TTM FCF (OCF - CapEx)
    cashflow_df = None
    try:
        store = get_duckdb_store()
        cashflow_df = store.get_cashflow(
            ticker,
            end_date,
            limit=limit + 12,
            fields="ts_code,end_date,n_cashflow_act,c_pay_acq_const_fiolta",
        )
    except Exception:
        pass
    if cashflow_df is None:
        cashflow_df = _call_tushare(
            "cashflow",
            pro,
            ts_code=ticker,
            end_date=ts_end,
            limit=limit + 12,
            fields="ts_code,end_date,n_cashflow_act,c_pay_acq_const_fiolta",
        )
    if cashflow_df is not None and not cashflow_df.empty:
        cashflow_df = cashflow_df.drop_duplicates(subset=["end_date"]).sort_values("end_date", ascending=False)
    else:
        cashflow_df = pd.DataFrame()

    cashflow_map = {}
    for r in _df_to_records(cashflow_df):
        cashflow_map[str(r.get("end_date", ""))] = r

    records = _df_to_records(fina_df)
    if not records:
        return []

    # Pre-compute D&A estimate from the most recent record with valid ebitda + ebit.
    # This is used as a fallback when quarterly reports (Q1/Q3) omit ebitda.
    da_estimate = None
    for rec in records:
        rec_ebitda = _to_float(rec.get("ebitda"))
        rec_ebit = _to_float(rec.get("ebit"))
        if rec_ebitda is not None and rec_ebitda > 0 and rec_ebit is not None and rec_ebit > 0:
            da_estimate = rec_ebitda - rec_ebit
            break

    # 预计算 TTM 所需的单季度数据
    income_items = sorted(income_map.items(), key=lambda x: x[0], reverse=True)
    eps_quarterly = _split_to_quarterly([(k, _to_float(v.get("basic_eps"))) for k, v in income_items])
    np_quarterly = _split_to_quarterly([(k, _to_float(v.get("n_income_attr_p"))) for k, v in income_items])

    cashflow_items = sorted(cashflow_map.items(), key=lambda x: x[0], reverse=True)
    ocf_quarterly = _split_to_quarterly([(k, _to_float(v.get("n_cashflow_act"))) for k, v in cashflow_items])
    capex_quarterly = _split_to_quarterly([(k, _to_float(v.get("c_pay_acq_const_fiolta"))) for k, v in cashflow_items])

    metrics = []
    for r in records:
        end_dt = str(r.get("end_date", ""))
        # 找最接近报告期的 daily_basic
        basic = basic_map.get(end_dt, {})
        if not basic and basic_map:
            # 取最近一天的
            basic = list(basic_map.values())[0]

        # 市值：万元 -> 元
        total_mv = basic.get("total_mv")
        market_cap = float(total_mv) * 10000 if total_mv is not None else None

        # PEG = PE_TTM / 单季度净利润同比百分点（与聚宽一致）
        pe_ttm = _to_float(basic.get("pe_ttm"))
        sq_np_yoy = _calculate_quarterly_yoy(np_quarterly, end_dt)
        if pe_ttm is not None and sq_np_yoy is not None and sq_np_yoy > 0:
            peg = pe_ttm / (sq_np_yoy * 100)  # 聚宽使用百分点而非小数
        else:
            peg = None

        # 从 balancesheet 获取数据计算 enterprise_value
        bs = bs_map.get(end_dt, {})
        if not bs and bs_map:
            # 取最近一期（balancesheet 日期 <= fina_indicator 日期）
            valid_dates = [d for d in bs_map.keys() if d <= end_dt]
            if valid_dates:
                bs = bs_map[max(valid_dates)]
        total_liab = _to_float(bs.get("total_liab"))
        money_cap = _to_float(bs.get("money_cap"))
        outstanding_shares = _to_float(bs.get("total_share"))
        equity_exc_min = _to_float(bs.get("total_hldr_eqy_exc_min_int"))

        # 手动计算负债权益比 = 总负债 / 归母权益（与国际通行 D/E 一致）
        debt_to_equity = None
        if total_liab is not None and equity_exc_min is not None and equity_exc_min > 0:
            debt_to_equity = total_liab / equity_exc_min

        # 计算 enterprise_value = market_cap + total_debt - cash
        # balancesheet 数据单位为元，无需转换
        ev = None
        if market_cap is not None and total_liab is not None:
            cash = money_cap if money_cap is not None else 0
            ev = market_cap + total_liab - cash

        # 从 fina_indicator 获取 ebitda，计算 EV/EBITDA
        ebitda_val = _to_float(r.get("ebitda"))
        ebit_val = _to_float(r.get("ebit"))

        # Fallback: estimate ebitda from ebit + D&A when quarterly report omits ebitda.
        ebitda_missing = ebitda_val is None or pd.isna(ebitda_val) or ebitda_val <= 0
        if (
            ebitda_missing
            and ebit_val is not None
            and not pd.isna(ebit_val)
            and ebit_val > 0
            and da_estimate is not None
            and not pd.isna(da_estimate)
            and da_estimate > 0
        ):
            ebitda_val = ebit_val + da_estimate

        # Only compute ev_to_ebitda when the source EBITDA is present (not fallback).
        # Q1/Q3 reports often omit EBITDA; using a fallback produces a single-quarter
        # figure that is not comparable to annual/H1 cumulative multiples and badly
        # distorts the historical median.  We still store the fallback EBITDA in the
        # model so that valuation agents can use the most recent profit proxy.
        ev_to_ebitda = None
        if ev is not None and ebitda_val is not None and ebitda_val > 0:
            if not ebitda_missing:
                ev_to_ebitda = ev / ebitda_val

        # 计算 free_cash_flow_yield = FCF / market_cap
        fcff_ps = _to_float(r.get("fcff_ps"))
        fcf_yield = None
        if fcff_ps is not None and outstanding_shares is not None and market_cap is not None and market_cap > 0:
            fcf = fcff_ps * outstanding_shares
            fcf_yield = fcf / market_cap

        # EPS YoY（使用 Tushare fina_indicator 提供的累计同比，与聚宽一致）
        eps_yoy = _to_pct(r.get("basic_eps_yoy"))

        # 手动计算单季度营收 YoY（Tushare fina_indicator.q_sales_yoy 与 income 口径存在差异）
        revenue_yoy = _to_pct(r.get("q_sales_yoy"))
        if end_dt and len(end_dt) == 8:
            curr_year = int(end_dt[:4])
            curr_md = end_dt[4:]
            quarter_map = {"0331": None, "0630": "0331", "0930": "0630", "1231": "0930"}
            prev_md = quarter_map.get(curr_md)
            curr_rec = income_map.get(end_dt)
            prev_year_dt = f"{curr_year - 1}{curr_md}"
            prev_year_rec = income_map.get(prev_year_dt)
            if curr_rec and prev_year_rec:
                curr_rev = _to_float(curr_rec.get("total_revenue"))
                prev_year_rev = _to_float(prev_year_rec.get("total_revenue"))
                if curr_rev is not None and prev_year_rev is not None and prev_year_rev != 0:
                    if prev_md:
                        prev_dt = f"{curr_year}{prev_md}"
                        prev_rec = income_map.get(prev_dt)
                        prev_year_prev_dt = f"{curr_year - 1}{prev_md}"
                        prev_year_prev_rec = income_map.get(prev_year_prev_dt)
                        if prev_rec and prev_year_prev_rec:
                            prev_rev = _to_float(prev_rec.get("total_revenue"))
                            prev_year_prev_rev = _to_float(prev_year_prev_rec.get("total_revenue"))
                            if prev_rev is not None and prev_year_prev_rev is not None:
                                q_rev = curr_rev - prev_rev
                                q_rev_prev = prev_year_rev - prev_year_prev_rev
                                if q_rev_prev != 0:
                                    revenue_yoy = (q_rev - q_rev_prev) / abs(q_rev_prev)
                    else:
                        revenue_yoy = (curr_rev - prev_year_rev) / abs(prev_year_rev)

        metrics.append(
            FinancialMetrics(
                ticker=ticker,
                report_period=_from_tushare_date(end_dt) if end_dt else end_date,
                period=period,
                currency="CNY",
                market_cap=market_cap,
                enterprise_value=ev,
                price_to_earnings_ratio=pe_ttm,
                price_to_book_ratio=_to_float(basic.get("pb")),
                price_to_sales_ratio=_to_float(basic.get("ps_ttm")),
                enterprise_value_to_ebitda_ratio=ev_to_ebitda,
                ebitda=ebitda_val,
                enterprise_value_to_revenue_ratio=None,
                free_cash_flow_yield=fcf_yield,
                peg_ratio=peg,
                gross_margin=_to_pct(r.get("grossprofit_margin")),
                operating_margin=_to_pct(r.get("op_of_gr")),
                net_margin=_to_pct(r.get("profit_to_gr")),
                return_on_equity=_to_pct(r.get("roe")),
                return_on_assets=_to_pct(r.get("roa")),
                return_on_invested_capital=_to_pct(r.get("roe_yearly")),
                asset_turnover=_to_float(r.get("assets_turn")),
                inventory_turnover=_to_float(r.get("inv_turn")),
                receivables_turnover=_to_float(r.get("ar_turn")),
                days_sales_outstanding=_to_float(r.get("days_ar_turn")),
                operating_cycle=_to_float(r.get("op_cycle")),
                working_capital_turnover=None,
                current_ratio=_to_float(r.get("current_ratio")),
                quick_ratio=_to_float(r.get("quick_ratio")),
                cash_ratio=_to_float(r.get("cash_ratio")),
                operating_cash_flow_ratio=_to_pct(r.get("q_ocf_to_sales")),
                debt_to_equity=debt_to_equity,
                debt_to_assets=_to_pct(r.get("debt_to_assets")),
                interest_coverage=_to_float(r.get("int_to_talcap")),
                revenue_growth=revenue_yoy,
                earnings_growth=_to_pct(r.get("netprofit_yoy")),
                # BPS 是时点值（资产负债表），相邻期环比有意义
                book_value_growth=None,  # 在下面通过相邻期 bps 计算
                # EPS / FCF / EBITDA 是累计值，必须用同口径同比（yoy），不能用环比
                earnings_per_share_growth=eps_yoy,
                free_cash_flow_growth=_to_pct(r.get("ocf_yoy")),  # 累计 OCF 同比，与聚宽一致
                operating_income_growth=_to_pct(r.get("q_op_yoy")),
                ebitda_growth=_to_pct(r.get("netprofit_yoy")),  # 无直接 ebitda_yoy，用净利润同比近似
                payout_ratio=None,  # Tushare 无标准分红率字段
                earnings_per_share=_to_float(r.get("eps")),
                book_value_per_share=_to_float(r.get("bps")),
                free_cash_flow_per_share=fcff_ps,
            )
        )

    # 仅 book_value_growth 保留相邻期环比（BPS 是资产负债表时点值，口径一致）
    for i in range(len(metrics) - 1):
        current_bps = metrics[i].book_value_per_share
        prev_bps = metrics[i + 1].book_value_per_share
        if current_bps is not None and prev_bps is not None and prev_bps != 0:
            metrics[i].book_value_growth = (current_bps - prev_bps) / abs(prev_bps)

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
    return metrics[:limit]


def _to_float(value) -> float | None:
    """Safely convert a value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_pct(value) -> float | None:
    """Convert a percentage value (e.g., 15.5 for 15.5%) to decimal (0.155)."""
    f = _to_float(value)
    if f is None:
        return None
    return f / 100


def _prev_quarter(end_dt: str) -> str | None:
    """Get the previous quarter end date.

    e.g., 20250630 -> 20250331, 20250331 -> 20241231
    """
    if len(end_dt) != 8:
        return None
    year = int(end_dt[:4])
    md = end_dt[4:]
    mapping = {"0331": (year - 1, "1231"), "0630": (year, "0331"), "0930": (year, "0630"), "1231": (year, "0930")}
    if md not in mapping:
        return None
    y, m = mapping[md]
    return f"{y:04d}{m}"


def _split_to_quarterly(values: list[tuple[str, float | None]]) -> dict[str, float | None]:
    """Convert cumulative financial values to single-quarter values.

    A-share financial reports are cumulative (Q1=Jan-Mar, Q2=Jan-Jun, etc.).
    This function splits them into single-quarter values so that TTM can be
    computed correctly.

    Args:
        values: List of (end_date, cumulative_value) tuples, sorted by end_date descending.

    Returns:
        Dict mapping end_date to single-quarter value.
    """
    value_map = {end_dt: val for end_dt, val in values}
    result: dict[str, float | None] = {}
    for end_dt, val in values:
        if val is None:
            result[end_dt] = None
            continue
        md = end_dt[4:]
        if md == "0331":
            # Q1: single quarter = cumulative value
            result[end_dt] = val
        else:
            prev_end = _prev_quarter(end_dt)
            prev_val = value_map.get(prev_end) if prev_end else None
            if prev_val is not None:
                result[end_dt] = val - prev_val
            else:
                result[end_dt] = None
    return result


def _calculate_ttm(quarterly_map: dict[str, float | None], end_dt: str) -> float | None:
    """Calculate TTM (sum of last 4 single quarters) for a given end_date."""
    total = 0.0
    curr = end_dt
    for _ in range(4):
        qv = quarterly_map.get(curr)
        if qv is None:
            return None
        total += qv
        curr = _prev_quarter(curr)
        if curr is None:
            return None
    return total


def _calculate_ttm_yoy(quarterly_map: dict[str, float | None], end_dt: str) -> float | None:
    """Calculate TTM YoY growth for a given end_date."""
    current_ttm = _calculate_ttm(quarterly_map, end_dt)
    prior_end = end_dt
    for _ in range(4):
        prior_end = _prev_quarter(prior_end)
        if prior_end is None:
            return None
    prior_ttm = _calculate_ttm(quarterly_map, prior_end)
    if current_ttm is None or prior_ttm is None or prior_ttm == 0:
        return None
    return (current_ttm - prior_ttm) / abs(prior_ttm)


def _calculate_fcf_ttm_yoy(
    ocf_quarterly: dict[str, float | None],
    capex_quarterly: dict[str, float | None],
    end_dt: str,
) -> float | None:
    """Calculate TTM FCF YoY growth. FCF = OCF - CapEx."""
    current_ocf = _calculate_ttm(ocf_quarterly, end_dt)
    current_capex = _calculate_ttm(capex_quarterly, end_dt)
    if current_ocf is None or current_capex is None:
        return None
    current_fcf = current_ocf - current_capex

    prior_end = end_dt
    for _ in range(4):
        prior_end = _prev_quarter(prior_end)
        if prior_end is None:
            return None
    prior_ocf = _calculate_ttm(ocf_quarterly, prior_end)
    prior_capex = _calculate_ttm(capex_quarterly, prior_end)
    if prior_ocf is None or prior_capex is None:
        return None
    prior_fcf = prior_ocf - prior_capex

    if prior_fcf == 0:
        return None
    return (current_fcf - prior_fcf) / abs(prior_fcf)


def _calculate_quarterly_yoy(quarterly_map: dict[str, float | None], end_dt: str) -> float | None:
    """Calculate single-quarter YoY growth for a given end_date.

    聚宽 PEG 使用单季度净利润同比百分点（非累计 TTM 或累计同比）。
    """
    curr_q = quarterly_map.get(end_dt)
    if curr_q is None:
        return None
    if len(end_dt) != 8:
        return None
    year = int(end_dt[:4])
    md = end_dt[4:]
    prev_dt = f"{year - 1}{md}"
    prev_q = quarterly_map.get(prev_dt)
    if prev_q is not None and prev_q != 0:
        return (curr_q - prev_q) / abs(prev_q)
    return None


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from Tushare (income / balancesheet / cashflow)."""
    pro = _get_pro_api(api_key)
    ts_end = _to_tushare_date(end_date)

    # 分组
    table_groups: dict[str, list[tuple[str, str]]] = {"income": [], "balancesheet": [], "cashflow": []}
    unknown = []

    # working_capital 不是 Tushare 的直接字段，需要计算
    resolved_items = list(line_items)
    needs_working_capital = "working_capital" in resolved_items
    if needs_working_capital:
        resolved_items = [item for item in resolved_items if item != "working_capital"]
        resolved_items.extend(["current_assets", "current_liabilities"])

    # dividends_and_other_cash_distributions 使用 dividend 接口而非 cashflow
    needs_dividends = "dividends_and_other_cash_distributions" in resolved_items
    if needs_dividends:
        resolved_items = [item for item in resolved_items if item != "dividends_and_other_cash_distributions"]

    for item in resolved_items:
        mapping = LINE_ITEM_MAPPING.get(item)
        if mapping:
            table_groups[mapping[0]].append((item, mapping[1]))
        else:
            unknown.append(item)

    if unknown:
        logger.warning("Unknown line items for Tushare mapping: %s", unknown)

    # 按 report_period 合并结果
    merged: dict[str, dict] = {}

    def _fetch_and_merge(table: str, fields: list[str], tushare_fields: list[str]):
        if not tushare_fields:
            return
        try:
            fields_str = ",".join(["ts_code", "end_date"] + tushare_fields)
            if table == "income":
                df = _call_tushare("income", pro, ts_code=ticker, end_date=ts_end, limit=limit + 8, fields=fields_str)
            elif table == "balancesheet":
                df = _call_tushare("balancesheet", pro, ts_code=ticker, end_date=ts_end, limit=limit + 8, fields=fields_str)
            elif table == "cashflow":
                df = _call_tushare("cashflow", pro, ts_code=ticker, end_date=ts_end, limit=limit + 8, fields=fields_str)
            else:
                return
            if df is not None and not df.empty:
                df = df.drop_duplicates(subset=["end_date"]).sort_values("end_date", ascending=False).head(limit)
        except Exception as e:
            logger.warning("Failed to fetch %s for %s: %s", table, ticker, e)
            return

        for r in _df_to_records(df):
            end_dt = str(r.get("end_date", ""))
            key = end_dt
            if key not in merged:
                merged[key] = {
                    "ticker": ticker,
                    "report_period": _from_tushare_date(end_dt) if end_dt else end_date,
                    "period": period,
                    "currency": "CNY",
                }
            for eng_name, ts_field in fields:
                merged[key][eng_name] = r.get(ts_field)

    # 去重字段并调用
    for table in ("income", "balancesheet", "cashflow"):
        group = table_groups[table]
        if not group:
            continue
        seen = set()
        unique_fields = []
        unique_pairs = []
        for eng, tsf in group:
            if tsf not in seen:
                seen.add(tsf)
                unique_fields.append((eng, tsf))
                unique_pairs.append(tsf)
        _fetch_and_merge(table, unique_fields, unique_pairs)

    # 计算 working_capital（流动资产 - 流动负债）
    if needs_working_capital:
        for data in merged.values():
            ca = data.get("current_assets")
            cl = data.get("current_liabilities")
            if ca is not None and cl is not None:
                data["working_capital"] = ca - cl

    # 使用 dividend 接口获取分红数据（总分红额 = cash_div * base_share * 10000）
    if needs_dividends:
        div_df = _call_tushare("dividend", pro, ts_code=ticker)
        if div_df is None:
            div_df = pd.DataFrame()

        div_by_year: dict[str, float] = {}
        for r in _df_to_records(div_df):
            year = str(r.get("end_date", ""))
            if not year:
                continue
            # 只统计已实施的分红
            if r.get("div_proc") != "实施":
                continue
            cash_div = _to_float(r.get("cash_div")) or 0.0
            base_share = _to_float(r.get("base_share"))
            if base_share is not None and base_share > 0:
                total_div = cash_div * base_share * 10000
            else:
                # 缺少股本时退回到每股分红作为存在性判断
                total_div = cash_div
            div_by_year[year] = div_by_year.get(year, 0.0) + total_div

        if merged:
            for key, data in merged.items():
                data["dividends_and_other_cash_distributions"] = div_by_year.get(key)
        else:
            # 仅请求 dividends 时，按年份生成独立条目
            for year in sorted(div_by_year.keys(), reverse=True)[:limit]:
                merged[year] = {
                    "ticker": ticker,
                    "report_period": _from_tushare_date(year) if year else end_date,
                    "period": period,
                    "currency": "CNY",
                    "dividends_and_other_cash_distributions": div_by_year[year],
                }

    results = [LineItem(**data) for data in merged.values()]
    return results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades (shareholder changes) from Tushare."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    pro = _get_pro_api(api_key)
    kwargs = {"ts_code": ticker, "end_date": _to_tushare_date(end_date)}
    if start_date:
        kwargs["start_date"] = _to_tushare_date(start_date)
    df = _call_tushare("stk_holdertrade", pro, **kwargs)
    if df is None:
        return []

    records = _df_to_records(df)
    if not records:
        return []

    trades = []
    for r in records:
        change_vol = _to_float(r.get("change_vol")) or 0.0
        in_de = r.get("in_de", "")  # IN=增持, DE=减持
        if in_de == "DE":
            change_vol = -abs(change_vol)
        elif in_de == "IN":
            change_vol = abs(change_vol)
        else:
            change_vol = 0.0

        ann_date = str(r.get("ann_date", ""))
        avg_price = _to_float(r.get("avg_price"))
        if avg_price is not None and math.isnan(avg_price):
            avg_price = None
        transaction_value = change_vol * avg_price if (change_vol and avg_price is not None) else None
        trades.append(
            InsiderTrade(
                ticker=ticker,
                issuer=r.get("holder_name") or ticker,
                name=r.get("holder_name"),
                title=r.get("holder_type"),
                is_board_director=None,
                transaction_date=_from_tushare_date(ann_date) if ann_date else None,
                transaction_shares=change_vol,
                transaction_price_per_share=avg_price,
                transaction_value=transaction_value,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=_to_float(r.get("after_share")),
                security_title=None,
                filing_date=_from_tushare_date(ann_date) if ann_date else ann_date,
            )
        )

    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in trades])
    return trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news via AkShare (Eastmoney). Falls back to empty list on failure."""
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare is not installed. Install it with: pip install akshare")
        return []

    code = ticker.split(".")[0]
    try:
        df = ak.stock_news_em(symbol=code)
    except Exception as e:
        logger.warning("Failed to fetch news for %s via akshare: %s", ticker, e)
        return []

    if df is None or df.empty:
        return []

    # Normalize column names (akshare may return Chinese or English column names)
    col_map = {}
    for col in df.columns:
        lower = str(col).lower()
        if "title" in lower or "标题" in lower:
            col_map["title"] = col
        elif "time" in lower or "date" in lower or "时间" in lower or "发布" in lower:
            col_map["date"] = col
        elif "source" in lower or "来源" in lower:
            col_map["source"] = col
        elif "url" in lower or "链接" in lower or "href" in lower:
            col_map["url"] = col
        elif "content" in lower or "摘要" in lower or "内容" in lower:
            col_map["content"] = col

    news_items = []
    for _, row in df.head(limit).iterrows():
        title = str(row.get(col_map.get("title", "title"), "")).strip()
        if not title:
            continue

        date_val = row.get(col_map.get("date", "pub_time"), "")
        if pd.isna(date_val):
            date_str = ""
        else:
            date_str = str(date_val).strip()

        source_val = row.get(col_map.get("source", "source"), "")
        source_str = str(source_val).strip() if not pd.isna(source_val) else ""

        url_val = row.get(col_map.get("url", "url"), "")
        url_str = str(url_val).strip() if not pd.isna(url_val) else ""

        # content 字段用于 LLM 分析（可选）
        content_val = row.get(col_map.get("content", "content"), "")
        content_str = str(content_val).strip() if not pd.isna(content_val) else ""

        news_items.append(
            CompanyNews(
                ticker=ticker,
                title=title,
                author=None,
                source=source_str or "东方财富",
                date=date_str,
                url=url_str,
            )
        )

    logger.info("Fetched %d news items for %s via akshare", len(news_items), ticker)
    return news_items


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from local DuckDB or Tushare (daily_basic.total_mv, 万元 -> 元)."""
    # Try local DuckDB first
    try:
        store = get_duckdb_store()
        df = store.get_daily_basic(ticker, end_date)
        if df is not None and not df.empty:
            total_mv = df.iloc[0].get("total_mv")
            if total_mv is not None:
                return float(total_mv) * 10000
    except Exception:
        pass

    pro = _get_pro_api(api_key)
    ts_end = _to_tushare_date(end_date)
    df = _call_tushare("daily_basic", pro, ts_code=ticker, trade_date=ts_end)
    if df is None:
        return None

    records = _df_to_records(df)
    if not records:
        return None

    total_mv = records[0].get("total_mv")
    if total_mv is None:
        return None
    return float(total_mv) * 10000


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)


def get_northbound_holdings(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 100,
    api_key: str = None,
) -> list[NorthboundHolding]:
    """Fetch northbound (Stock Connect) holdings from Tushare.

    Tushare hk_hold returns low-frequency data (typically month-end snapshots).
    We fetch a ~90-day window by default to capture 2-3 data points.
    """
    if start_date is None:
        from datetime import datetime, timedelta
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=90)
        ).strftime("%Y-%m-%d")

    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached_data := _cache.get_northbound_holdings(cache_key):
        return [NorthboundHolding(**item) for item in cached_data]

    pro = _get_pro_api(api_key)
    df = _call_tushare(
        "hk_hold",
        pro,
        ts_code=ticker,
        start_date=_to_tushare_date(start_date),
        end_date=_to_tushare_date(end_date),
    )
    if df is None:
        return []

    records = _df_to_records(df)
    if not records:
        return []

    # Sort by trade_date ascending so index 0 is oldest
    records = sorted(records, key=lambda r: str(r.get("trade_date", "")))
    holdings = []
    for r in records:
        holdings.append(
            NorthboundHolding(
                ticker=ticker,
                trade_date=_from_tushare_date(str(r.get("trade_date", ""))),
                vol=_to_float(r.get("vol")),
                ratio=_to_float(r.get("ratio")),
            )
        )

    _cache.set_northbound_holdings(
        cache_key, [h.model_dump() for h in holdings]
    )
    return holdings


def get_margin_data(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 100,
    api_key: str = None,
) -> list[MarginData]:
    """Fetch margin trading (financing & securities lending) data from Tushare.

    margin_detail is daily-frequency; we default to a ~30-day window.
    """
    if start_date is None:
        from datetime import datetime, timedelta
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=35)
        ).strftime("%Y-%m-%d")

    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached_data := _cache.get_margin_data(cache_key):
        return [MarginData(**item) for item in cached_data]

    pro = _get_pro_api(api_key)
    df = _call_tushare(
        "margin_detail",
        pro,
        ts_code=ticker,
        start_date=_to_tushare_date(start_date),
        end_date=_to_tushare_date(end_date),
    )
    if df is None:
        return []

    records = _df_to_records(df)
    if not records:
        return []

    records = sorted(records, key=lambda r: str(r.get("trade_date", "")))
    margin_items = []
    for r in records:
        margin_items.append(
            MarginData(
                ticker=ticker,
                trade_date=_from_tushare_date(str(r.get("trade_date", ""))),
                rzye=_to_float(r.get("rzye")),
                rqye=_to_float(r.get("rqye")),
                rzmre=_to_float(r.get("rzmre")),
                rzche=_to_float(r.get("rzche")),
                rqyl=_to_float(r.get("rqyl")),
                rzrqye=_to_float(r.get("rzrqye")),
            )
        )

    _cache.set_margin_data(cache_key, [m.model_dump() for m in margin_items])
    return margin_items


def get_china_bond_yield(
    term: float = 10.0,
    api_key: str = None,
) -> float | None:
    """Fetch China government bond yield with daily caching.

    Tushare ``yc_cb`` has a **2 requests/minute** rate limit,
    so we cache aggressively and reuse the same calendar day's
    value across all calls.  File-level cache is used so that
    spawned worker processes share the same cache.
    """
    from datetime import datetime
    import json
    from pathlib import Path

    cache_key = f"bond_yield_{term}"
    today = datetime.now().date().isoformat()

    # 1. Check file-level cache (shared across processes)
    file_cache_path = Path(".tushare_cache")
    file_cache_path.mkdir(exist_ok=True)
    cache_file = file_cache_path / f"{cache_key}.json"

    file_cached: dict = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                file_cached = json.load(f)
            if file_cached.get("cached_date") == today:
                logger.info(
                    "Using file-cached bond yield (term=%.1fyr): %.4f%% (%s)",
                    term,
                    file_cached["yield"] * 100,
                    file_cached.get("trade_date", "unknown"),
                )
                # Also warm the in-memory cache
                _cache.set_bond_yield(cache_key, file_cached)
                return file_cached["yield"]
        except Exception:
            pass

    # 2. Check in-memory cache (fast path for same process)
    cached = _cache.get_bond_yield(cache_key)
    if cached:
        cached_at = cached.get("cached_at")
        if isinstance(cached_at, str):
            cached_at = datetime.fromisoformat(cached_at)
        if cached_at and cached_at.date().isoformat() == today:
            # Persist to file so other processes can see it
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({**cached, "cached_date": today}, f)
            except Exception:
                pass
            return cached["yield"]

    # 3. Fetch from API (only one process should reach here per day)
    pro = _get_pro_api(api_key)
    df = _call_tushare("yc_cb", pro, ts_code="1001.CB", curve_term=term)
    if df is None or df.empty:
        logger.warning("yc_cb returned empty for term=%.1f", term)
        # Fallback chain: file cache -> memory cache -> hardcoded 2.30%
        if file_cached.get("yield") is not None:
            return file_cached["yield"]
        if cached and cached.get("yield") is not None:
            return cached["yield"]
        return 0.0230

    # Average both curve_type rows for the same term
    yield_val = df["yield"].mean() / 100.0  # Convert % → decimal
    trade_date = str(df["trade_date"].iloc[0])

    payload = {
        "yield": yield_val,
        "trade_date": trade_date,
        "cached_at": datetime.now().isoformat(),
        "cached_date": today,
    }
    _cache.set_bond_yield(cache_key, payload)

    # Persist to file for other processes
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        logger.warning("Failed to write bond yield cache file: %s", e)

    logger.info(
        "Fetched bond yield (term=%.1fyr): %.4f%% (%s)",
        term,
        yield_val * 100,
        trade_date,
    )
    return yield_val


def get_beta(
    ticker: str,
    end_date: str | None = None,
    market_index: str = "000300.SH",
    lookback: int = 252,
    api_key: str = None,
) -> float | None:
    """Calculate stock beta against a market index using recent price history.

    Beta = Cov(stock_returns, market_returns) / Var(market_returns)
    """
    import numpy as np

    pro = _get_pro_api(api_key)

    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y%m%d")
    else:
        end_date = _to_tushare_date(end_date)

    stock_df = _call_tushare("daily", pro, ts_code=ticker, end_date=end_date, limit=lookback + 5)
    index_df = _call_tushare("index_daily", pro, ts_code=market_index, end_date=end_date, limit=lookback + 5)
    if stock_df is None or stock_df.empty or index_df is None or index_df.empty:
        logger.warning("Empty price data for beta calc: stock=%s index=%s", ticker, market_index)
        return None

    # Sort by date and compute daily returns
    stock_df = stock_df.sort_values("trade_date")
    index_df = index_df.sort_values("trade_date")

    stock_ret = stock_df["close"].pct_change().dropna()
    index_ret = index_df["close"].pct_change().dropna()

    # Align to same length
    min_len = min(len(stock_ret), len(index_ret))
    if min_len < 30:
        logger.warning("Insufficient data for beta calc (%s): %d points", ticker, min_len)
        return None

    stock_ret = stock_ret.iloc[-min_len:].to_numpy()
    index_ret = index_ret.iloc[-min_len:].to_numpy()

    # Beta = Cov(stock, market) / Var(market)
    covariance = np.cov(stock_ret, index_ret)[0, 1]
    variance = np.var(index_ret, ddof=1)

    if variance == 0:
        logger.warning("Zero market variance for beta calc (%s)", ticker)
        return None

    beta = covariance / variance
    logger.info("Calculated beta for %s: %.4f (n=%d, index=%s)", ticker, beta, min_len, market_index)
    return float(beta)
