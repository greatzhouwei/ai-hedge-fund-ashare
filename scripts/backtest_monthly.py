"""Monthly rebalancing backtest for jq_screener strategy.

Usage:
    poetry run python scripts/backtest_monthly.py

Buys top-10 stocks on first trading day of each month,
equal-weight, holds until next rebalance day.
Benchmark: CSI 300 (000300.SH).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.backtesting.jq_adapter import JQDataAdapter
from src.backtesting.jq_screener import run_screener

DB_PATH = Path("src/data/tushare_data.db")
OUTPUT_PATH = Path("backtest_monthly_result.json")
PORTFOLIO_CACHE = Path("monthly_portfolios_2022.json")
BENCHMARK_CACHE = Path("benchmark_000300_2022.json")

# Trained weights from ridge regression
RIDGE_WEIGHTS = {
    "fundamentals": 0.1094,
    "growth": 0.2996,
    "technical": 0.5911,
}

# Negative-tech experiment: reverse technical selection
NEGATIVE_TECH_WEIGHTS = {
    "fundamentals": 0,
    "growth": 0,
    "technical": -1,
}


def get_monthly_trade_dates(conn, start_date: str, end_date: str) -> list[str]:
    start_num = start_date.replace("-", "")
    end_num = end_date.replace("-", "")
    df = conn.execute(
        """
        SELECT MIN(cal_date) as first_date
        FROM trade_cal
        WHERE is_open = 1 AND cal_date BETWEEN ? AND ?
        GROUP BY SUBSTR(cal_date, 1, 6)
        ORDER BY first_date
        """,
        [start_num, end_num],
    ).fetchdf()
    return [
        pd.to_datetime(d, format="%Y%m%d").strftime("%Y-%m-%d")
        for d in df["first_date"]
    ]


def main():
    adapter = JQDataAdapter(str(DB_PATH))
    conn = adapter.conn

    rebalance_dates = get_monthly_trade_dates(conn, "2022-01-01", "2022-12-31")
    print(f"Rebalance dates ({len(rebalance_dates)}): {rebalance_dates}")

    if len(rebalance_dates) < 2:
        print("Not enough dates for backtest")
        return

    # Phase 1: stock selection for each month (use cache if available)
    # Use negative technical weights for experiment
    experiment_weights = NEGATIVE_TECH_WEIGHTS
    weights_label = "neg_tech"
    portfolio_cache = Path(f"monthly_portfolios_2022_{weights_label}.json")
    output_path = Path(f"backtest_monthly_result_{weights_label}.json")

    if portfolio_cache.exists():
        print(f"Loading cached portfolios from {portfolio_cache}")
        with open(portfolio_cache, "r", encoding="utf-8") as f:
            monthly_portfolios = json.load(f)
    else:
        monthly_portfolios: dict[str, list[str]] = {}
        for i, date in enumerate(rebalance_dates):
            print(f"\n[{i+1}/{len(rebalance_dates)}] Selecting stocks for {date}...", flush=True)
            top10, _ = run_screener(adapter, date, top_n=10, max_per_industry=100, weights=experiment_weights)
            tickers = [t for t, s in top10]
            monthly_portfolios[date] = tickers
            print(f"  Selected: {tickers}", flush=True)
        with open(portfolio_cache, "w", encoding="utf-8") as f:
            json.dump(monthly_portfolios, f, ensure_ascii=False, indent=2)
        print(f"\nSaved portfolios to {portfolio_cache}")

    # Phase 2: fetch all daily prices for involved tickers + benchmark
    all_tickers = list(
        set(t for tickers in monthly_portfolios.values() for t in tickers)
    )
    print(f"\nTotal unique tickers to fetch: {len(all_tickers)}")

    start_num = "20220101"
    end_num = "20221231"

    price_data: dict[str, dict[str, float]] = {}
    batch_size = 500
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        placeholders = ",".join(["?"] * len(batch))
        df = conn.execute(
            f"""
            SELECT ts_code, trade_date, close
            FROM daily
            WHERE ts_code IN ({placeholders})
              AND trade_date BETWEEN ? AND ?
            """,
            batch + [start_num, end_num],
        ).fetchdf()
        for _, row in df.iterrows():
            t = row["ts_code"]
            d = pd.to_datetime(str(row["trade_date"]), format="%Y%m%d").strftime(
                "%Y-%m-%d"
            )
            price_data.setdefault(t, {})[d] = float(row["close"])

    # Benchmark daily prices — try cache first, then local DB, then Tushare API
    bench_series: dict[str, float] = {}
    if BENCHMARK_CACHE.exists():
        print(f"Loading cached benchmark from {BENCHMARK_CACHE}")
        with open(BENCHMARK_CACHE, "r", encoding="utf-8") as f:
            bench_series = {
                pd.to_datetime(k).strftime("%Y-%m-%d"): v
                for k, v in json.load(f).items()
            }
    else:
        bench_df = conn.execute(
            """
            SELECT trade_date, close
            FROM daily
            WHERE ts_code = '000300.SH'
              AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
            """,
            [start_num, end_num],
        ).fetchdf()

        if bench_df.empty:
            print("Benchmark 000300.SH not found in local DB, fetching from Tushare...")
            from dotenv import load_dotenv
            load_dotenv()
            from src.tools.api import _call_tushare, _get_pro_api
            pro = _get_pro_api()
            bench_df = _call_tushare(
                "index_daily",
                pro,
                ts_code="000300.SH",
                start_date=start_num,
                end_date=end_num,
            )
            if bench_df is None or bench_df.empty:
                raise RuntimeError(
                    "Failed to fetch benchmark 000300.SH from both local DB and Tushare API."
                )
            bench_df = bench_df.sort_values("trade_date")

        for _, row in bench_df.iterrows():
            d = pd.to_datetime(str(row["trade_date"]), format="%Y%m%d").strftime(
                "%Y-%m-%d"
            )
            bench_series[d] = float(row["close"])

        with open(BENCHMARK_CACHE, "w", encoding="utf-8") as f:
            json.dump(bench_series, f, ensure_ascii=False, indent=2)
        print(f"Saved benchmark to {BENCHMARK_CACHE}")

    if not bench_series:
        raise RuntimeError("No benchmark data available.")

    all_dates = sorted(bench_series.keys())

    # Phase 3: simulate daily portfolio value
    cash = 1.0
    positions: dict[str, float] = {}  # ticker -> shares
    portfolio_values: list[dict] = []

    for date_str in all_dates:
        # Rebalance on first trading day of month
        if date_str in monthly_portfolios:
            # Sell current positions at today's close
            if positions:
                sell_val = sum(
                    positions[t] * price_data[t].get(date_str, 0)
                    for t in positions
                    if t in price_data and date_str in price_data[t]
                )
                cash = sell_val
                positions = {}

            # Buy new positions at today's close (equal weight)
            tickers = monthly_portfolios[date_str]
            buy_prices = {
                t: price_data[t][date_str]
                for t in tickers
                if t in price_data
                and date_str in price_data[t]
                and price_data[t][date_str] > 0
            }
            if buy_prices:
                alloc = cash / len(buy_prices)
                positions = {t: alloc / p for t, p in buy_prices.items()}
                cash = 0

        # Record end-of-day portfolio value
        if positions:
            val = sum(
                positions[t] * price_data[t].get(date_str, 0)
                for t in positions
                if t in price_data and date_str in price_data[t]
            )
        else:
            val = cash

        portfolio_values.append({"date": date_str, "value": val})

    # Build DataFrames
    port_df = pd.DataFrame(portfolio_values)
    port_df["date"] = pd.to_datetime(port_df["date"])
    port_df = port_df.set_index("date").sort_index()
    port_df["daily_return"] = port_df["value"].pct_change()

    bench_df2 = pd.DataFrame(
        [{"date": d, "close": p} for d, p in bench_series.items()]
    )
    bench_df2["date"] = pd.to_datetime(bench_df2["date"])
    bench_df2 = bench_df2.set_index("date").sort_index()
    bench_df2["daily_return"] = bench_df2["close"].pct_change()

    # Align
    combined = port_df.join(bench_df2, how="inner", rsuffix="_bench")
    combined = combined.dropna(subset=["daily_return", "daily_return_bench"])

    # Core returns
    start_val = float(port_df["value"].iloc[0])
    end_val = float(port_df["value"].iloc[-1])
    strategy_return = end_val / start_val - 1

    bench_start = float(bench_df2["close"].iloc[0])
    bench_end = float(bench_df2["close"].iloc[-1])
    benchmark_return = bench_end / bench_start - 1

    excess_return = strategy_return - benchmark_return

    trading_days = len(combined)
    annual_factor = 252
    if trading_days > 0:
        ann_strategy = (1 + strategy_return) ** (annual_factor / trading_days) - 1
        ann_benchmark = (1 + benchmark_return) ** (annual_factor / trading_days) - 1
    else:
        ann_strategy = 0.0
        ann_benchmark = 0.0

    # Alpha / Beta
    port_ret = combined["daily_return"].values
    bench_ret = combined["daily_return_bench"].values
    if len(port_ret) > 1 and np.std(bench_ret) > 1e-12:
        beta, alpha_daily = np.polyfit(bench_ret, port_ret, 1)
        alpha = alpha_daily * annual_factor
    else:
        beta, alpha = 0.0, 0.0

    # Sharpe
    rf = 0.034
    daily_rf = rf / annual_factor
    excess = port_ret - daily_rf
    sharpe = (
        np.sqrt(annual_factor) * np.mean(excess) / np.std(excess)
        if np.std(excess) > 1e-12
        else 0.0
    )

    # Sortino
    downside = np.minimum(excess, 0)
    downside_std = np.sqrt(np.mean(downside**2))
    sortino = (
        np.sqrt(annual_factor) * np.mean(excess) / downside_std
        if downside_std > 1e-12
        else (float("inf") if np.mean(excess) > 0 else 0.0)
    )

    # Monthly metrics
    port_monthly = port_df["value"].resample("ME").last()
    monthly_rets = (port_monthly / port_monthly.shift(1) - 1).dropna()
    monthly_wins = int((monthly_rets > 0).sum())
    monthly_losses = int((monthly_rets < 0).sum())
    monthly_win_rate = (
        monthly_wins / (monthly_wins + monthly_losses)
        if (monthly_wins + monthly_losses) > 0
        else 0.0
    )

    # Daily win rate
    daily_wins = int((port_ret > 0).sum())
    daily_losses = int((port_ret < 0).sum())
    daily_win_rate = (
        daily_wins / (daily_wins + daily_losses)
        if (daily_wins + daily_losses) > 0
        else 0.0
    )

    # P/L ratio
    pos_rets = port_ret[port_ret > 0]
    neg_rets = port_ret[port_ret < 0]
    pl_ratio = (
        float(np.mean(pos_rets) / abs(np.mean(neg_rets)))
        if len(neg_rets) > 0 and np.mean(neg_rets) != 0
        else float("inf")
    )

    # Max drawdown
    rolling_max = port_df["value"].cummax()
    drawdown = (port_df["value"] - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    max_dd_date = str(drawdown.idxmin()) if max_dd < 0 else None
    peak_date = None
    if max_dd < 0 and max_dd_date:
        peak_candidates = rolling_max.loc[:max_dd_date]
        if not peak_candidates.empty:
            peak_date = str(peak_candidates.idxmax())

    # Information ratio
    excess_daily = port_ret - bench_ret
    info_ratio = (
        np.sqrt(annual_factor) * np.mean(excess_daily) / np.std(excess_daily)
        if np.std(excess_daily) > 1e-12
        else 0.0
    )

    # Excess return max drawdown
    excess_curve = pd.Series((1 + excess_daily).cumprod(), index=combined.index)
    exc_rolling_max = excess_curve.cummax()
    exc_drawdown = (excess_curve - exc_rolling_max) / exc_rolling_max
    max_excess_dd = float(exc_drawdown.min())

    # Volatility
    strategy_vol = float(np.std(port_ret) * np.sqrt(annual_factor))
    benchmark_vol = float(np.std(bench_ret) * np.sqrt(annual_factor))

    result = {
        "strategy_return_pct": round(strategy_return * 100, 2),
        "strategy_annual_return_pct": round(ann_strategy * 100, 2),
        "excess_return_pct": round(excess_return * 100, 2),
        "benchmark_return_pct": round(benchmark_return * 100, 2),
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "monthly_win_rate": round(monthly_win_rate, 3),
        "daily_win_rate": round(daily_win_rate, 3),
        "profit_loss_ratio": round(pl_ratio, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_date": max_dd_date,
        "max_drawdown_peak_date": peak_date,
        "info_ratio": round(info_ratio, 3),
        "daily_excess_return_pct": round(float(np.mean(excess_daily)) * 100, 4),
        "max_excess_drawdown_pct": round(max_excess_dd * 100, 2),
        "monthly_wins": monthly_wins,
        "monthly_losses": monthly_losses,
        "daily_wins": daily_wins,
        "daily_losses": daily_losses,
        "strategy_volatility_pct": round(strategy_vol * 100, 2),
        "benchmark_volatility_pct": round(benchmark_vol * 100, 2),
        "trading_days": trading_days,
        "rebalance_dates": rebalance_dates,
    }

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (2022-01-01 ~ 2022-12-31)")
    print("=" * 70)
    print(f"  策略收益:          {result['strategy_return_pct']:>8.2f}%")
    print(f"  策略年化收益:      {result['strategy_annual_return_pct']:>8.2f}%")
    print(f"  超额收益:          {result['excess_return_pct']:>8.2f}%")
    print(f"  基准收益:          {result['benchmark_return_pct']:>8.2f}%")
    print(f"  阿尔法:            {result['alpha']:>8.4f}")
    print(f"  贝塔:              {result['beta']:>8.4f}")
    print(f"  夏普比率:          {result['sharpe_ratio']:>8.3f}")
    print(f"  索提诺比率:        {result['sortino_ratio']:>8.3f}")
    print(f"  胜率(月度):        {result['monthly_win_rate']:>8.3f}")
    print(f"  日胜率:            {result['daily_win_rate']:>8.3f}")
    print(f"  盈亏比:            {result['profit_loss_ratio']:>8.3f}")
    print(f"  最大回撤:          {result['max_drawdown_pct']:>8.2f}%")
    print(f"  最大回撤区间:      {result['max_drawdown_peak_date']} ~ {result['max_drawdown_date']}")
    print(f"  信息比率:          {result['info_ratio']:>8.3f}")
    print(f"  日均超额收益:      {result['daily_excess_return_pct']:>8.4f}%")
    print(f"  超额收益最大回撤:  {result['max_excess_drawdown_pct']:>8.2f}%")
    print(f"  盈利次数(月度):    {result['monthly_wins']:>8d}")
    print(f"  亏损次数(月度):    {result['monthly_losses']:>8d}")
    print(f"  策略波动率:        {result['strategy_volatility_pct']:>8.2f}%")
    print(f"  基准波动率:        {result['benchmark_volatility_pct']:>8.2f}%")
    print("=" * 70)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")

    adapter.close()


if __name__ == "__main__":
    main()
