from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.tools.api import get_company_news
from pydantic import BaseModel


# Lookback window for news. Long enough to cover quarterly catalysts and
# avoid overfitting to one-day spikes; short enough that stale headlines
# don't dominate the current sentiment read.
NEWS_LOOKBACK_DAYS = 60


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
    """News-driven sentiment analyst.

    Scope: short-term news flow only.

    Insider/holder trades are intentionally NOT used here — they are already
    consumed by `growth_agent.analyze_insider_conviction` with holder-type and
    amount weighting. Including them again here would double-count the same
    raw `stk_holdertrade` feed across two analysts and inflate confidence in
    portfolio_manager's signal aggregation.
    """
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "TUSHARE_TOKEN")
    sentiment_analysis = {}

    for ticker in tickers:
        # Determine the lookback window for filtering recent news
        try:
            end_dt = datetime.strptime(str(end_date), "%Y-%m-%d")
        except (ValueError, TypeError):
            end_dt = datetime.now()
        cutoff_dt = end_dt - timedelta(days=NEWS_LOOKBACK_DAYS)

        progress.update_status(agent_id, ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)

        # Filter to recent window only
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

        # Map news sentiment to bullish/bearish/neutral
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(
            sentiment == "negative", "bearish",
            np.where(sentiment == "positive", "bullish", "neutral"),
        ).tolist()

        progress.update_status(agent_id, ticker, "Aggregating signals")

        # Aggregate by simple vote count (each article = one vote)
        bullish_signals = news_signals.count("bullish")
        bearish_signals = news_signals.count("bearish")
        neutral_signals = news_signals.count("neutral")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        total_signals = len(news_signals)
        confidence = 0
        if total_signals > 0:
            confidence = round(
                max(bullish_signals, bearish_signals) / total_signals * 100, 2
            )

        reasoning = {
            "news_sentiment": {
                "signal": overall_signal,
                "confidence": confidence,
                "metrics": {
                    "total_articles": total_signals,
                    "bullish_articles": bullish_signals,
                    "bearish_articles": bearish_signals,
                    "neutral_articles": neutral_signals,
                    "lookback_days": NEWS_LOOKBACK_DAYS,
                },
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
