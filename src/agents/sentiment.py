from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.tools.api import get_insider_trades, get_company_news
from pydantic import BaseModel


class NewsSentimentItem(BaseModel):
    index: int
    sentiment: str


class NewsSentimentBatch(BaseModel):
    results: list[NewsSentimentItem]


def _analyze_news_sentiment_with_llm(
    ticker: str,
    news_list: list,
    state: AgentState,
    agent_name: str,
    batch_size: int = 50,
) -> list:
    """Batch analyze news sentiment using LLM. Falls back to neutral on failure."""
    if not news_list:
        return news_list

    for start in range(0, len(news_list), batch_size):
        batch = news_list[start : start + batch_size]
        titles_text = "\n".join(
            [f"{i + 1}. {item.title}" for i, item in enumerate(batch)]
        )

        prompt = f"""你是中文金融新闻情感分析专家。请判断以下关于股票 {ticker} 的新闻标题对股价的短期影响。

分类标准：
- positive: 明显利好（如业绩增长、大单签约、政策支持、回购增持、技术突破等）
- negative: 明显利空（如业绩下滑、违规处罚、大额减持、诉讼亏损、裁员停产等）
- neutral: 影响不明显、行业常规新闻、公告披露、难以判断

请严格按 JSON 格式输出，不要添加任何解释：
{{"results": [{{"index": 0, "sentiment": "positive"}}, ...]}}

新闻标题列表：
{titles_text}
"""

        def _default_factory():
            return NewsSentimentBatch(
                results=[
                    NewsSentimentItem(index=i, sentiment="neutral")
                    for i in range(len(batch))
                ]
            )

        try:
            response: NewsSentimentBatch = call_llm(
                prompt=prompt,
                pydantic_model=NewsSentimentBatch,
                agent_name=agent_name,
                state=state,
                default_factory=_default_factory,
            )
            for item in response.results:
                idx = item.index
                if 0 <= idx < len(batch):
                    batch[idx].sentiment = (
                        item.sentiment.lower() if item.sentiment else "neutral"
                    )
        except Exception:
            # Fallback: mark entire batch as neutral
            for item in batch:
                item.sentiment = "neutral"

    return news_list


##### Sentiment Agent #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "TUSHARE_TOKEN")
    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        # Determine the 14-day lookback window
        try:
            end_dt = datetime.strptime(str(end_date), "%Y-%m-%d")
        except (ValueError, TypeError):
            end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=60)
        start_date_str = start_dt.strftime("%Y-%m-%d")

        progress.update_status(agent_id, ticker, "Fetching insider trades")

        # Get the insider trades (recent 60 days only)
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            start_date=start_date_str,
            limit=1000,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status(agent_id, ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)

        # Filter to recent 14 days only
        cutoff_dt = start_dt

        def _parse_news_date(date_str):
            if not date_str:
                return None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
                try:
                    return datetime.strptime(str(date_str).strip(), fmt)
                except ValueError:
                    continue
            return None

        company_news = [
            n for n in company_news
            if _parse_news_date(n.date) is not None and _parse_news_date(n.date) >= cutoff_dt
        ]

        # Limit to most recent 50 articles to control LLM call volume
        company_news = company_news[:50]

        # Analyze sentiment via LLM if not already present
        if company_news and all(n.sentiment is None for n in company_news):
            progress.update_status(agent_id, ticker, "Analyzing news sentiment via LLM")
            company_news = _analyze_news_sentiment_with_llm(
                ticker, company_news, state, agent_id
            )

        # Get the sentiment from the company news
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish", 
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status(agent_id, ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7
        
        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round((max(bullish_signals, bearish_signals) / total_weighted_signals) * 100, 2)
        
        # Create structured reasoning similar to technical analysis
        reasoning = {
            "insider_trading": {
                "signal": "bullish" if insider_signals.count("bullish") > insider_signals.count("bearish") else 
                         "bearish" if insider_signals.count("bearish") > insider_signals.count("bullish") else "neutral",
                "confidence": round((max(insider_signals.count("bullish"), insider_signals.count("bearish")) / max(len(insider_signals), 1)) * 100),
                "metrics": {
                    "total_trades": len(insider_signals),
                    "bullish_trades": insider_signals.count("bullish"),
                    "bearish_trades": insider_signals.count("bearish"),
                    "weight": insider_weight,
                    "weighted_bullish": round(insider_signals.count("bullish") * insider_weight, 1),
                    "weighted_bearish": round(insider_signals.count("bearish") * insider_weight, 1),
                }
            },
            "news_sentiment": {
                "signal": "bullish" if news_signals.count("bullish") > news_signals.count("bearish") else 
                         "bearish" if news_signals.count("bearish") > news_signals.count("bullish") else "neutral",
                "confidence": round((max(news_signals.count("bullish"), news_signals.count("bearish")) / max(len(news_signals), 1)) * 100),
                "metrics": {
                    "total_articles": len(news_signals),
                    "bullish_articles": news_signals.count("bullish"),
                    "bearish_articles": news_signals.count("bearish"),
                    "neutral_articles": news_signals.count("neutral"),
                    "weight": news_weight,
                    "weighted_bullish": round(news_signals.count("bullish") * news_weight, 1),
                    "weighted_bearish": round(news_signals.count("bearish") * news_weight, 1),
                }
            },
            "combined_analysis": {
                "total_weighted_bullish": round(bullish_signals, 1),
                "total_weighted_bearish": round(bearish_signals, 1),
                "signal_determination": f"{'Bullish' if bullish_signals > bearish_signals else 'Bearish' if bearish_signals > bullish_signals else 'Neutral'} based on weighted signal comparison"
            }
        }

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }
