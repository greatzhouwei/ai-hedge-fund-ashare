"""Smart Money Analyst (Northbound / Stock Connect holdings)

Tracks foreign institutional flows via the Stock Connect programme.
Tushare hk_hold provides low-frequency snapshots (typically month-end),
so trend detection is based on the latest 2-3 observable points.
"""

import json
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_northbound_holdings
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress

LOOKBACK_DAYS = 180


def smart_money_analyst_agent(state: AgentState, agent_id: str = "smart_money_analyst_agent"):
    """Northbound holdings analyst. 外资通过沪深港通的持股变化."""
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
        progress.update_status(agent_id, ticker, "Fetching northbound holdings")
        holdings = get_northbound_holdings(
            ticker=ticker,
            end_date=str(end_date),
            start_date=start_date,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing foreign flow trend")

        if not holdings:
            analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": {
                    "smart_money": {
                        "signal": "neutral",
                        "confidence": 0,
                        "metrics": {
                            "data_points": 0,
                            "latest_vol": None,
                            "latest_ratio": None,
                            "previous_vol": None,
                            "change_pct": None,
                            "lookback_days": LOOKBACK_DAYS,
                        },
                    }
                },
            }
            continue

        # holdings are sorted ascending by trade_date
        latest = holdings[-1]
        previous = holdings[-2] if len(holdings) >= 2 else None

        latest_vol = latest.vol or 0.0
        latest_ratio = latest.ratio or 0.0
        prev_vol = previous.vol if previous else None
        prev_ratio = previous.ratio if previous else None

        if prev_vol and prev_vol != 0:
            change_pct = (latest_vol - prev_vol) / prev_vol
        else:
            change_pct = 0.0

        if previous is None:
            # Only one data point: can't judge trend, but presence of holdings is mildly positive
            if latest_ratio and latest_ratio > 0:
                signal = "neutral"
                confidence = min(round(latest_ratio * 5, 2), 30)
            else:
                signal = "neutral"
                confidence = 0
        else:
            if change_pct > 0.02:  # >2% increase
                signal = "bullish"
            elif change_pct < -0.02:  # >2% decrease
                signal = "bearish"
            else:
                signal = "neutral"

            confidence = round(min(abs(change_pct) * 100, 100), 2)

        reasoning = {
            "smart_money": {
                "signal": signal,
                "confidence": confidence,
                "metrics": {
                    "data_points": len(holdings),
                    "latest_vol": latest_vol,
                    "latest_ratio": latest_ratio,
                    "previous_vol": prev_vol,
                    "previous_ratio": prev_ratio,
                    "change_pct": round(change_pct * 100, 2) if change_pct else None,
                    "lookback_days": LOOKBACK_DAYS,
                },
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
        show_agent_reasoning(analysis, "Smart Money Analyst")

    state["data"]["analyst_signals"][agent_id] = analysis
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }
