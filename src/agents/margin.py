"""Margin Trading Analyst (融资融券)

Tracks leveraged sentiment via margin-trading balances.
Rising margin balance = retail/speculator optimism (bullish short-term fuel).
Falling margin balance = deleveraging / risk-off (bearish).
"""

import json
import warnings
from datetime import datetime, timedelta

import numpy as np
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_margin_data
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress

LOOKBACK_DAYS = 35
TREND_DAYS = 20
NEAR_DAYS = 5


def _linear_trend(values: list[float]) -> tuple[float, float]:
    """Return (slope, r_squared) for a simple linear regression over index x."""
    if len(values) < 2:
        return 0.0, 0.0
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    if np.all(y == y[0]):
        return 0.0, 0.0
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            coeffs = np.polyfit(x, y, 1)
    if np.isnan(coeffs).any():
        return 0.0, 0.0
    slope = float(coeffs[0])
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return slope, float(r_squared)


def margin_analyst_agent(state: AgentState, agent_id: str = "margin_analyst_agent"):
    """Margin trading analyst. 融资融券余额变化反映市场杠杆情绪."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "TUSHARE_TOKEN")
    analysis = {}

    try:
        end_dt = datetime.strptime(str(end_date), "%Y-%m-%d")
    except (ValueError, TypeError):
        end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
    start_date = start_dt.strftime("%Y-%m-%d")

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching margin data")
        margin_items = get_margin_data(
            ticker=ticker,
            end_date=str(end_date),
            start_date=start_date,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing leverage trend")

        if not margin_items:
            analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": {
                    "margin": {
                        "signal": "neutral",
                        "confidence": 0,
                        "metrics": {
                            "data_points": 0,
                            "latest_rzye": None,
                            "trend_slope": None,
                            "r_squared": None,
                            "change_pct": None,
                            "net_buy_near": None,
                            "overheated": False,
                            "lookback_days": LOOKBACK_DAYS,
                        },
                    }
                },
            }
            continue

        # Extract rzye (融资余额) and net buy (rzmre - rzche)
        rzye_list = [m.rzye for m in margin_items if m.rzye is not None]
        net_buy_list = [
            (m.rzmre or 0.0) - (m.rzche or 0.0)
            for m in margin_items
            if m.rzmre is not None or m.rzche is not None
        ]

        if len(rzye_list) < 2:
            signal = "neutral"
            confidence = 0
            metrics = {
                "data_points": len(margin_items),
                "latest_rzye": rzye_list[-1] if rzye_list else None,
                "trend_slope": None,
                "r_squared": None,
                "change_pct": None,
                "net_buy_near": None,
                "overheated": False,
                "lookback_days": LOOKBACK_DAYS,
            }
        else:
            # Use last TREND_DAYS points for trend (or all if fewer)
            trend_vals = rzye_list[-TREND_DAYS:] if len(rzye_list) >= TREND_DAYS else rzye_list
            slope, r2 = _linear_trend(trend_vals)

            first_val = trend_vals[0]
            last_val = trend_vals[-1]
            change_pct = (last_val - first_val) / first_val if first_val and first_val != 0 else 0.0

            # Near-term net buy (last NEAR_DAYS)
            near_buys = net_buy_list[-NEAR_DAYS:] if len(net_buy_list) >= NEAR_DAYS else net_buy_list
            net_buy_near = sum(near_buys) if near_buys else 0.0

            # Overheat: near-term margin balance jumped >20% in last 5 days
            near_window = min(5, len(trend_vals))
            near_first = trend_vals[-near_window]
            near_change_pct = (last_val - near_first) / near_first if near_first and near_first != 0 else 0.0
            overheated = near_change_pct > 0.20

            # When linear fit is poor, trust change_pct more than slope
            if r2 >= 0.30:
                if slope > 0 and change_pct > 0.02:
                    signal = "bullish"
                elif slope < 0 and change_pct < -0.02:
                    signal = "bearish"
                else:
                    signal = "neutral"
            else:
                if change_pct > 0.02:
                    signal = "bullish"
                elif change_pct < -0.02:
                    signal = "bearish"
                else:
                    signal = "neutral"

            # Confidence: abs(change_pct) * 100, scaled by fit quality
            confidence = round(min(abs(change_pct) * 100 * (0.5 + 0.5 * r2), 100), 2)

            if overheated and signal == "bullish":
                signal = "neutral"
                confidence = min(confidence, 40.0)

            metrics = {
                "data_points": len(margin_items),
                "latest_rzye": last_val,
                "trend_slope": round(slope, 2),
                "r_squared": round(r2, 4),
                "change_pct": round(change_pct * 100, 2),
                "net_buy_near": round(net_buy_near, 2),
                "overheated": overheated,
                "lookback_days": LOOKBACK_DAYS,
            }

        reasoning = {
            "margin": {
                "signal": signal,
                "confidence": confidence,
                "metrics": metrics,
            }
        }

        analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(
            agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4)
        )

    message = HumanMessage(content=json.dumps(analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis, "Margin Trading Analyst")

    state["data"]["analyst_signals"][agent_id] = analysis
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }
