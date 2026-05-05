from __future__ import annotations

"""Valuation Agent

Implements four complementary valuation methodologies and aggregates them with
configurable weights. 
"""

import json
import math
import statistics
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state
from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
)

def valuation_analyst_agent(state: AgentState, agent_id: str = "valuation_analyst_agent"):
    """Run valuation across tickers and write signals back to `state`."""

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TUSHARE_TOKEN")
    valuation_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial data")

        # --- Historical financial metrics ---
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=8,
            api_key=api_key,
        )
        if not financial_metrics:
            progress.update_status(agent_id, ticker, "Failed: No financial metrics found")
            continue
        most_recent_metrics = financial_metrics[0]

        # --- Enhanced line‑items ---
        progress.update_status(agent_id, ticker, "Gathering comprehensive line items")
        line_items = search_line_items(
            ticker=ticker,
            line_items=[
                "free_cash_flow",
                "net_income",
                "depreciation_and_amortization",
                "capital_expenditure",
                "working_capital",
                "total_debt",
                "cash_and_equivalents",
            ],
            end_date=end_date,
            period="ttm",
            limit=8,
            api_key=api_key,
        )
        if len(line_items) < 2:
            progress.update_status(agent_id, ticker, "Failed: Insufficient financial line items")
            continue
        li_curr, li_prev = line_items[0], line_items[1]

        # ------------------------------------------------------------------
        # 口径对齐：Tushare income/balancesheet 返回原始报表数据
        # Q1(03-31)和Q3(09-30)为单季度，H1(06-30)和年报(12-31)为累计数。
        # Owner Earnings / Residual Income 需要全年口径，若当前期为单季度，
        # 则回退到最近一期完整报表（line_items[1] 通常为年报或半年报）。
        # ------------------------------------------------------------------
        is_quarterly = (
            li_curr.report_period
            and li_curr.report_period.endswith(("-03-31", "-09-30"))
        )
        li_oe = line_items[1] if is_quarterly else li_curr

        # 找到与 li_oe 同类型（年报/半年报）的前一期数据，确保口径一致
        li_prev_oe = li_prev
        if is_quarterly and li_oe.report_period:
            target_suffix = li_oe.report_period[-5:]  # "-12-31" or "-06-30"
            for li in line_items[2:]:
                if li.report_period and li.report_period.endswith(target_suffix):
                    li_prev_oe = li
                    break

        # Working capital change (使用口径对齐后的期数)
        if li_oe.working_capital is not None and li_prev_oe.working_capital is not None:
            wc_change = li_oe.working_capital - li_prev_oe.working_capital
        else:
            wc_change = 0

        # Owner Earnings — fall back to most recent valid depreciation
        depreciation = li_oe.depreciation_and_amortization
        if depreciation is None or (isinstance(depreciation, float) and math.isnan(depreciation)):
            for li in line_items:
                d = li.depreciation_and_amortization
                if d is not None and isinstance(d, (int, float)) and not math.isnan(d):
                    depreciation = d
                    break

        # Growth rate: 若当前期是单季度，most_recent_metrics 对应 Q1 数据，
        # earnings_growth 可能极低（如 1.47%）。尝试使用与 li_oe 同期（年报/
        # 半年报）的 earnings_growth；若仍异常低，设下限 5%。
        growth_rate = most_recent_metrics.earnings_growth or 0.05
        if is_quarterly:
            for fm in financial_metrics[1:]:
                if fm.report_period == li_oe.report_period:
                    if fm.earnings_growth is not None and fm.earnings_growth > 0:
                        growth_rate = fm.earnings_growth
                    break
        if growth_rate < 0.03:
            growth_rate = max(growth_rate, 0.05)

        # Calculate WACC (needed for Owner Earnings discount rate and DCF)
        progress.update_status(agent_id, ticker, "Calculating WACC and enhanced DCF")
        wacc_result = calculate_wacc(
            market_cap=most_recent_metrics.market_cap or 0,
            total_debt=getattr(li_curr, 'total_debt', None),
            cash=getattr(li_curr, 'cash_and_equivalents', None),
            interest_coverage=most_recent_metrics.interest_coverage,
            debt_to_equity=most_recent_metrics.debt_to_equity,
            ticker=ticker,
        )
        wacc = wacc_result['wacc']
        cost_of_equity = wacc_result['cost_of_equity']

        owner_val = calculate_owner_earnings_value(
            net_income=li_oe.net_income,
            depreciation=depreciation,
            capex=li_oe.capital_expenditure,
            working_capital_change=wc_change,
            growth_rate=growth_rate,
            required_return=max(cost_of_equity, 0.08),
            margin_of_safety=0.0,
        )

        # Enhanced Discounted Cash Flow with WACC and scenarios

        # Prepare FCF history for enhanced DCF
        fcf_history = []
        for li in line_items:
            if hasattr(li, 'free_cash_flow') and li.free_cash_flow is not None:
                fcf_history.append(li.free_cash_flow)

        # Enhanced DCF with scenarios
        dcf_results = calculate_dcf_scenarios(
            fcf_history=fcf_history,
            growth_metrics={
                'revenue_growth': most_recent_metrics.revenue_growth,
                'fcf_growth': most_recent_metrics.free_cash_flow_growth,
                'earnings_growth': most_recent_metrics.earnings_growth
            },
            wacc=wacc,
            market_cap=most_recent_metrics.market_cap or 0,
            revenue_growth=most_recent_metrics.revenue_growth
        )

        dcf_val = dcf_results['expected_value']

        # Implied Equity Value
        ev_ebitda_val = calculate_ev_ebitda_value(financial_metrics)

        # Residual Income Model — 使用口径对齐后的净利润
        rim_val = calculate_residual_income_value(
            market_cap=most_recent_metrics.market_cap,
            net_income=li_oe.net_income,
            price_to_book_ratio=most_recent_metrics.price_to_book_ratio,
            book_value_growth=most_recent_metrics.book_value_growth or 0.03,
            cost_of_equity=cost_of_equity,
        )

        # ------------------------------------------------------------------
        # Aggregate & signal
        # ------------------------------------------------------------------
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        if not market_cap:
            progress.update_status(agent_id, ticker, "Failed: Market cap unavailable")
            continue

        method_values = {
            "dcf": {"value": dcf_val, "weight": 0.35},
            "owner_earnings": {"value": owner_val, "weight": 0.35},
            "ev_ebitda": {"value": ev_ebitda_val, "weight": 0.20},
            "residual_income": {"value": rim_val, "weight": 0.10},
        }

        total_weight = sum(v["weight"] for v in method_values.values() if v["value"] > 0)
        if total_weight == 0:
            progress.update_status(agent_id, ticker, "Failed: All valuation methods zero")
            continue

        for v in method_values.values():
            v["gap"] = (v["value"] - market_cap) / market_cap if v["value"] > 0 else None

        weighted_gap = sum(
            v["weight"] * v["gap"] for v in method_values.values() if v["gap"] is not None
        ) / total_weight

        # Composite growth for repair-space estimation
        rg = most_recent_metrics.revenue_growth
        eg = most_recent_metrics.earnings_growth
        fg = most_recent_metrics.free_cash_flow_growth
        valid_growths = [g for g in [rg, eg, fg] if g is not None and -0.5 < g < 2.0]
        composite_growth = statistics.median(valid_growths) if valid_growths else 0.03
        composite_growth = max(composite_growth, 0.0)

        # Valuation repair space = gap amplified by growth support
        repair_space = weighted_gap * (1 + composite_growth)

        # Signal: bullish only if undervalued AND supported by growth (avoid value traps)
        if weighted_gap > 0.15 and composite_growth >= 0.05:
            signal = "bullish"
        elif weighted_gap < -0.15:
            signal = "bearish"
        else:
            signal = "neutral"

        confidence = round(min(abs(repair_space) / 0.30 * 100, 100))

        # Enhanced reasoning with DCF scenario details
        reasoning = {}
        for m, vals in method_values.items():
            if vals["value"] > 0:
                base_details = (
                    f"Value: ¥{vals['value']:,.2f}, Market Cap: ¥{market_cap:,.2f}, "
                    f"Gap: {vals['gap']:.1%}, Weight: {vals['weight']*100:.0f}%"
                )

                # Add enhanced DCF details
                if m == "dcf" and 'dcf_results' in locals():
                    enhanced_details = (
                        f"{base_details}\n"
                        f"  WACC: {wacc:.1%}, Bear: ¥{dcf_results['downside']:,.2f}, "
                        f"Bull: ¥{dcf_results['upside']:,.2f}, Range: ¥{dcf_results['range']:,.2f}"
                    )
                else:
                    enhanced_details = base_details
                
                reasoning[f"{m}_analysis"] = {
                    "signal": (
                        "bullish" if vals["gap"] and vals["gap"] > 0.15 else
                        "bearish" if vals["gap"] and vals["gap"] < -0.15 else "neutral"
                    ),
                    "details": enhanced_details,
                }
        
        # Add overall DCF scenario summary if available
        if 'dcf_results' in locals():
            reasoning["dcf_scenario_analysis"] = {
                "bear_case": f"¥{dcf_results['downside']:,.2f}",
                "base_case": f"¥{dcf_results['scenarios']['base']:,.2f}",
                "bull_case": f"¥{dcf_results['upside']:,.2f}",
                "wacc_used": f"{wacc:.1%}",
                "fcf_periods_analyzed": len(fcf_history)
            }

        valuation_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # ---- Emit message (for LLM tool chain) ----
    msg = HumanMessage(content=json.dumps(valuation_analysis), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(valuation_analysis, "Valuation Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = valuation_analysis

    progress.update_status(agent_id, None, "Done")
    
    return {"messages": [msg], "data": data}

#############################
# Helper Valuation Functions
#############################

def calculate_owner_earnings_value(
    net_income: float | None,
    depreciation: float | None,
    capex: float | None,
    working_capital_change: float | None,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    """Buffett owner‑earnings valuation with margin‑of‑safety."""
    if net_income is None or not isinstance(net_income, (int, float)) or math.isnan(net_income):
        return 0

    # Allow missing depreciation/capex/wc_change by defaulting to 0 (conservative)
    depreciation = depreciation if isinstance(depreciation, (int, float)) and not math.isnan(depreciation) else 0
    capex = capex if isinstance(capex, (int, float)) and not math.isnan(capex) else 0
    working_capital_change = working_capital_change if isinstance(working_capital_change, (int, float)) and not math.isnan(working_capital_change) else 0

    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:
        return 0

    pv = 0.0
    for yr in range(1, num_years + 1):
        future = owner_earnings * (1 + growth_rate) ** yr
        pv += future / (1 + required_return) ** yr

    terminal_growth = min(growth_rate, 0.03)
    term_val = (owner_earnings * (1 + growth_rate) ** num_years * (1 + terminal_growth)) / (
        required_return - terminal_growth
    )
    pv_term = term_val / (1 + required_return) ** num_years

    intrinsic = pv + pv_term
    return intrinsic * (1 - margin_of_safety)


def calculate_intrinsic_value(
    free_cash_flow: float | None,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """Classic DCF on FCF with constant growth and terminal value."""
    if free_cash_flow is None or free_cash_flow <= 0:
        return 0

    pv = 0.0
    for yr in range(1, num_years + 1):
        fcft = free_cash_flow * (1 + growth_rate) ** yr
        pv += fcft / (1 + discount_rate) ** yr

    term_val = (
        free_cash_flow * (1 + growth_rate) ** num_years * (1 + terminal_growth_rate)
    ) / (discount_rate - terminal_growth_rate)
    pv_term = term_val / (1 + discount_rate) ** num_years

    return pv + pv_term


def calculate_ev_ebitda_value(financial_metrics: list):
    """Implied equity value via median EV/EBITDA multiple."""
    if not financial_metrics:
        return 0
    m0 = financial_metrics[0]
    if not m0.enterprise_value:
        return 0

    # Current EBITDA: prefer the explicit field;
    # if the current period lacks an original EBITDA (enterprise_value_to_ebitda_ratio is None),
    # use the most recent full-period EBITDA from historical records instead of the
    # single-quarter fallback, which badly distorts the valuation.
    ebitda_now = m0.ebitda
    if ebitda_now is None or ebitda_now <= 0:
        if m0.enterprise_value_to_ebitda_ratio and m0.enterprise_value_to_ebitda_ratio > 0:
            ebitda_now = m0.enterprise_value / m0.enterprise_value_to_ebitda_ratio
        else:
            return 0

    # Detect fallback: if current period has no original EV/EBITDA ratio, the EBITDA is
    # a single-quarter estimate and not comparable to annual/H1 multiples.
    # Substitute with the most recent full-period EBITDA from history.
    if m0.enterprise_value_to_ebitda_ratio is None:
        for m in financial_metrics[1:]:
            if (
                m.enterprise_value_to_ebitda_ratio is not None
                and m.ebitda is not None
                and m.ebitda > 0
            ):
                # Sanity check: only substitute if the historical EBITDA is materially larger,
                # confirming it's a cumulative (annual/H1) figure vs a single-quarter fallback.
                if m.ebitda > ebitda_now * 1.3:
                    ebitda_now = m.ebitda
                break

    # Historical median: ONLY use periods where the original EBITDA was present.
    # Q1/Q3 reports omit EBITDA; our fallback produces a single-quarter figure
    # that is not comparable to annual/H1 cumulative multiples and badly distorts
    # the median (e.g. 43x vs 15x for the same stock).
    valid_ratios = [
        m.enterprise_value_to_ebitda_ratio
        for m in financial_metrics
        if m.enterprise_value_to_ebitda_ratio is not None and m.enterprise_value_to_ebitda_ratio > 0
    ]
    if not valid_ratios:
        return 0

    med_mult = statistics.median(valid_ratios)
    ev_implied = med_mult * ebitda_now
    net_debt = (m0.enterprise_value or 0) - (m0.market_cap or 0)
    return max(ev_implied - net_debt, 0)


def calculate_residual_income_value(
    market_cap: float | None,
    net_income: float | None,
    price_to_book_ratio: float | None,
    book_value_growth: float = 0.03,
    cost_of_equity: float = 0.10,
    terminal_growth_rate: float = 0.03,
    num_years: int = 5,
):
    """Residual Income Model (Edwards‑Bell‑Ohlson)."""
    if not (market_cap and net_income and price_to_book_ratio and price_to_book_ratio > 0):
        return 0

    # RIM is highly sensitive to cost_of_equity; low-beta stocks in a low-rate
    # environment can produce implausibly low Ke via CAPM. Floor at 8%.
    cost_of_equity = max(cost_of_equity, 0.08)

    # Constrain terminal growth so that (Ke - g) is never too small,
    # which would explode the terminal multiplier (1 / (Ke - g)).
    terminal_growth_rate = min(terminal_growth_rate, cost_of_equity * 0.35)

    # Cap book-value growth to avoid implausible multi-period compounding.
    # If g >= Ke the present-value formula degenerates; keep it well below Ke.
    book_value_growth = min(book_value_growth, cost_of_equity * 0.90)

    book_val = market_cap / price_to_book_ratio
    ri0 = net_income - cost_of_equity * book_val
    if ri0 <= 0:
        return 0

    pv_ri = 0.0
    for yr in range(1, num_years + 1):
        ri_t = ri0 * (1 + book_value_growth) ** yr
        pv_ri += ri_t / (1 + cost_of_equity) ** yr

    term_ri = ri0 * (1 + book_value_growth) ** (num_years + 1) / (
        cost_of_equity - terminal_growth_rate
    )
    pv_term = term_ri / (1 + cost_of_equity) ** num_years

    intrinsic = book_val + pv_ri + pv_term
    return intrinsic


####################################
# Enhanced DCF Helper Functions
####################################

def calculate_wacc(
    market_cap: float,
    total_debt: float | None,
    cash: float | None,
    interest_coverage: float | None,
    debt_to_equity: float | None,
    ticker: str | None = None,
    beta_proxy: float | None = None,
    risk_free_rate: float | None = None,
    market_risk_premium: float = 0.06
) -> dict[str, float]:
    """Calculate WACC using available financial data."""

    # Fetch risk-free rate from China bond yield if not explicitly provided
    if risk_free_rate is None:
        from src.tools.api import get_china_bond_yield
        fetched = get_china_bond_yield(term=10.0)
        risk_free_rate = fetched if fetched is not None else 0.045

    # Fetch beta from price history if not explicitly provided
    if beta_proxy is None:
        if ticker is not None:
            from src.tools.api import get_beta
            beta_proxy = get_beta(ticker) or 1.0
        else:
            beta_proxy = 1.0

    # Cost of Equity (CAPM)
    cost_of_equity = risk_free_rate + beta_proxy * market_risk_premium
    
    # Cost of Debt - estimate from interest coverage and debt_to_equity
    if interest_coverage and interest_coverage > 0:
        base_spread = 0.10 / interest_coverage
    else:
        base_spread = 0.05  # Default spread

    # Adjust by debt_to_equity if available (higher leverage = higher cost)
    if debt_to_equity is not None and debt_to_equity > 0:
        leverage_premium = min(debt_to_equity * 0.01, 0.05)
        base_spread += leverage_premium

    cost_of_debt = max(risk_free_rate + 0.01, risk_free_rate + base_spread)
    
    # Weights
    net_debt = max((total_debt or 0) - (cash or 0), 0)
    total_value = market_cap + net_debt
    
    if total_value > 0:
        weight_equity = market_cap / total_value
        weight_debt = net_debt / total_value
        
        # Tax shield (assume 25% corporate tax rate)
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * 0.75)
    else:
        wacc = cost_of_equity
    
    wacc = min(max(wacc, 0.06), 0.20)  # Floor 6%, cap 20%
    return {"wacc": wacc, "cost_of_equity": cost_of_equity}


def calculate_fcf_volatility(fcf_history: list[float]) -> float:
    """Calculate FCF volatility as coefficient of variation."""
    if len(fcf_history) < 3:
        return 0.5  # Default moderate volatility
    
    # Filter out zeros and negatives for volatility calc
    positive_fcf = [fcf for fcf in fcf_history if fcf > 0]
    if len(positive_fcf) < 2:
        return 0.8  # High volatility if mostly negative FCF
    
    try:
        mean_fcf = statistics.mean(positive_fcf)
        std_fcf = statistics.stdev(positive_fcf)
        return min(std_fcf / mean_fcf, 1.0) if mean_fcf > 0 else 0.8
    except:
        return 0.5


def calculate_enhanced_dcf_value(
    fcf_history: list[float],
    growth_metrics: dict,
    wacc: float,
    market_cap: float,
    revenue_growth: float | None = None
) -> float:
    """Enhanced DCF with multi-stage growth."""
    import math

    # Filter out NaN / None values
    clean_fcf = [f for f in fcf_history if isinstance(f, (int, float)) and not math.isnan(f)]
    if not clean_fcf:
        return 0

    # If latest FCF is negative (common for growth companies in capex cycles),
    # use the most recent positive FCF as base instead of giving up.
    fcf_current = clean_fcf[0]
    if fcf_current <= 0:
        positive_fcf = [f for f in clean_fcf if f > 0]
        if positive_fcf:
            fcf_current = positive_fcf[0]
        else:
            return 0

    # Analyze FCF trend and quality
    fcf_avg_3yr = sum(clean_fcf[:3]) / min(3, len(clean_fcf))
    fcf_volatility = calculate_fcf_volatility(clean_fcf)
    
    # Stage 1: High Growth (Years 1-3)
    # Use revenue growth but cap based on business maturity
    high_growth = min(revenue_growth or 0.05, 0.25) if revenue_growth else 0.05
    if market_cap > 50_000_000_000:  # Large cap
        high_growth = min(high_growth, 0.10)
    
    # Stage 2: Transition (Years 4-7)
    transition_growth = (high_growth + 0.03) / 2
    
    # Stage 3: Terminal (steady state)
    terminal_growth = min(0.03, high_growth * 0.6)
    
    # Project FCF with stages
    pv = 0
    base_fcf = max(fcf_current, fcf_avg_3yr * 0.85)  # Conservative base
    
    # High growth stage
    for year in range(1, 4):
        fcf_projected = base_fcf * (1 + high_growth) ** year
        pv += fcf_projected / (1 + wacc) ** year
    
    # Transition stage
    for year in range(4, 8):
        transition_rate = transition_growth * (8 - year) / 4  # Declining
        fcf_projected = base_fcf * (1 + high_growth) ** 3 * (1 + transition_rate) ** (year - 3)
        pv += fcf_projected / (1 + wacc) ** year
    
    # Terminal value
    final_fcf = base_fcf * (1 + high_growth) ** 3 * (1 + transition_growth) ** 4
    if wacc <= terminal_growth:
        terminal_growth = wacc * 0.8  # Adjust if invalid
    terminal_value = (final_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** 7
    
    # Quality adjustment based on FCF volatility
    quality_factor = max(0.7, 1 - (fcf_volatility * 0.5))
    
    return (pv + pv_terminal) * quality_factor


def calculate_dcf_scenarios(
    fcf_history: list[float],
    growth_metrics: dict,
    wacc: float,
    market_cap: float,
    revenue_growth: float | None = None
) -> dict:
    """Calculate DCF under multiple scenarios."""
    
    scenarios = {
        'bear': {'growth_adj': 0.5, 'wacc_adj': 1.2, 'terminal_adj': 0.8},
        'base': {'growth_adj': 1.0, 'wacc_adj': 1.0, 'terminal_adj': 1.0},
        'bull': {'growth_adj': 1.5, 'wacc_adj': 0.9, 'terminal_adj': 1.2}
    }
    
    results = {}
    base_revenue_growth = revenue_growth or 0.05
    
    for scenario, adjustments in scenarios.items():
        adjusted_revenue_growth = base_revenue_growth * adjustments['growth_adj']
        adjusted_wacc = wacc * adjustments['wacc_adj']
        
        results[scenario] = calculate_enhanced_dcf_value(
            fcf_history=fcf_history,
            growth_metrics=growth_metrics,
            wacc=adjusted_wacc,
            market_cap=market_cap,
            revenue_growth=adjusted_revenue_growth
        )
    
    # Probability-weighted average
    expected_value = (
        results['bear'] * 0.2 + 
        results['base'] * 0.6 + 
        results['bull'] * 0.2
    )
    
    return {
        'scenarios': results,
        'expected_value': expected_value,
        'range': results['bull'] - results['bear'],
        'upside': results['bull'],
        'downside': results['bear']
    }
