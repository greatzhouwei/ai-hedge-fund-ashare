import datetime
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
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
        token = api_key or os.environ.get("TUSHARE_TOKEN")
        if not token:
            raise ValueError("TUSHARE_TOKEN is not set. Please set it in your .env file.")
        _pro_api = ts.pro_api(token)
    return _pro_api


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
    "depreciation_and_amortization": ("income", "depr_fa_coga_dpba"),  # 固定资产折旧、油气资产折耗、生产性生物资产折旧
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
    """Fetch price data from Tushare."""
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    pro = _get_pro_api(api_key)
    try:
        df = pro.daily(
            ts_code=ticker,
            start_date=_to_tushare_date(start_date),
            end_date=_to_tushare_date(end_date),
        )
    except Exception as e:
        logger.warning("Failed to fetch prices for %s: %s", ticker, e)
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
    try:
        fina_df = pro.fina_indicator(ts_code=ticker, end_date=ts_end, limit=limit)
    except Exception as e:
        logger.warning("Failed to fetch fina_indicator for %s: %s", ticker, e)
        fina_df = pd.DataFrame()

    # daily_basic 提供估值指标
    try:
        basic_df = pro.daily_basic(ts_code=ticker, trade_date=ts_end)
    except Exception as e:
        logger.warning("Failed to fetch daily_basic for %s: %s", ticker, e)
        basic_df = pd.DataFrame()

    # 将 daily_basic 按 trade_date 转为字典方便合并
    basic_map = {}
    for r in _df_to_records(basic_df):
        basic_map[str(r.get("trade_date", ""))] = r

    records = _df_to_records(fina_df)
    if not records:
        return []

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

        metrics.append(
            FinancialMetrics(
                ticker=ticker,
                report_period=_from_tushare_date(end_dt) if end_dt else end_date,
                period=period,
                currency="CNY",
                market_cap=market_cap,
                enterprise_value=None,
                price_to_earnings_ratio=_to_float(basic.get("pe_ttm")),
                price_to_book_ratio=_to_float(basic.get("pb")),
                price_to_sales_ratio=_to_float(basic.get("ps")),
                enterprise_value_to_ebitda_ratio=None,
                enterprise_value_to_revenue_ratio=None,
                free_cash_flow_yield=None,
                peg_ratio=_to_float(r.get("trady_3")) if "trady_3" in r else None,  # Tushare 无标准 PEG
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
                debt_to_equity=_to_float(r.get("debt_to_eqt")),
                debt_to_assets=_to_pct(r.get("debt_to_assets")),
                interest_coverage=_to_float(r.get("int_to_talcap")),
                revenue_growth=_to_pct(r.get("q_sales_yoy")),
                earnings_growth=_to_pct(r.get("netprofit_yoy")),
                book_value_growth=None,  # 在下面通过相邻期 bps 计算
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=_to_pct(r.get("q_op_yoy")),
                ebitda_growth=None,
                payout_ratio=_to_pct(r.get("profit_to_gr")),
                earnings_per_share=_to_float(r.get("eps")),
                book_value_per_share=_to_float(r.get("bps")),
                free_cash_flow_per_share=_to_float(r.get("fcff_ps")),
            )
        )

    # 通过相邻期 bps 计算 book_value_growth
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
            if table == "income":
                df = pro.income(ts_code=ticker, end_date=ts_end, limit=limit, fields=",".join(["ts_code", "end_date"] + tushare_fields))
            elif table == "balancesheet":
                df = pro.balancesheet(ts_code=ticker, end_date=ts_end, limit=limit, fields=",".join(["ts_code", "end_date"] + tushare_fields))
            elif table == "cashflow":
                df = pro.cashflow(ts_code=ticker, end_date=ts_end, limit=limit, fields=",".join(["ts_code", "end_date"] + tushare_fields))
            else:
                return
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
        try:
            div_df = pro.dividend(ts_code=ticker)
        except Exception as e:
            logger.warning("Failed to fetch dividend for %s: %s", ticker, e)
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
    try:
        df = pro.stk_holdertrade(
            ts_code=ticker,
            start_date=_to_tushare_date(start_date) if start_date else None,
            end_date=_to_tushare_date(end_date),
        )
    except Exception as e:
        logger.warning("Failed to fetch insider trades for %s: %s", ticker, e)
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
        trades.append(
            InsiderTrade(
                ticker=ticker,
                issuer=r.get("holder_name") or ticker,
                name=r.get("holder_name"),
                title=r.get("holder_type"),
                is_board_director=None,
                transaction_date=_from_tushare_date(ann_date) if ann_date else None,
                transaction_shares=change_vol,
                transaction_price_per_share=_to_float(r.get("avg_price")),
                transaction_value=change_vol * (_to_float(r.get("avg_price")) or 0.0) if change_vol else None,
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
    """Fetch market cap from Tushare (daily_basic.total_mv, 万元 -> 元)."""
    pro = _get_pro_api(api_key)
    ts_end = _to_tushare_date(end_date)
    try:
        df = pro.daily_basic(ts_code=ticker, trade_date=ts_end)
    except Exception as e:
        logger.warning("Failed to fetch market cap for %s: %s", ticker, e)
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
