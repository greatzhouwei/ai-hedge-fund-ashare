"""Backtrader strategy for periodic rebalancing based on 3agents scores."""

import json
from datetime import datetime
from pathlib import Path

import backtrader as bt


class RebalanceStrategy(bt.Strategy):
    params = (
        ("signals_dir", Path("batch_screener_results")),
        ("signals_prefix", "backtest_3agents_"),
        ("top_n", 20),
        ("rebalance_dates", None),
    )

    def __init__(self):
        self.signals: dict[str, dict[str, float]] = {}
        self.rebalance_dates_set: set[str] = set()
        if self.p.rebalance_dates:
            for d in self.p.rebalance_dates:
                ds = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
                self.rebalance_dates_set.add(ds)
        self._load_signals()

    def _load_signals(self):
        signals_dir = Path(self.p.signals_dir)
        pattern = f"{self.p.signals_prefix}*.jsonl"
        for f in sorted(signals_dir.glob(pattern)):
            prefix_len = len(self.p.signals_prefix)
            date_str = f.stem[prefix_len:]
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            day_sigs: dict[str, float] = {}
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "score" in rec:
                            day_sigs[rec["ticker"]] = rec["score"]
                    except json.JSONDecodeError:
                        pass
            if day_sigs:
                self.signals[date_str] = day_sigs

    def log(self, txt: str, dt: datetime | None = None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"[{dt}] {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            self.log(
                f"{action} EXECUTED {order.data._name} "
                f"size={order.executed.size} price={order.executed.price:.2f}"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED {order.data._name}: status={order.status}")

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        date_str = current_date.strftime("%Y-%m-%d")

        if date_str not in self.rebalance_dates_set:
            return

        day_signals = self.signals.get(date_str, {})
        if not day_signals:
            self.log(f"No signals for rebalance date {date_str}")
            return

        ranked = sorted(day_signals.items(), key=lambda x: x[1], reverse=True)
        top_tickers = {ticker for ticker, _ in ranked[: self.p.top_n]}

        self.log(f"Rebalancing -> top {self.p.top_n}  (date={date_str})")
        self.log(f"  top tickers: {sorted(top_tickers)[:5]}...")

        # Close positions not in top_n
        for data in self.datas:
            if len(data) == 0:
                continue
            pos = self.getposition(data)
            if pos.size != 0 and data._name not in top_tickers:
                self.close(data=data)
                self.log(f"  CLOSE {data._name} size={pos.size}")

        # Rebalance top_n to equal weight
        if not top_tickers:
            return

        target_pct = 1.0 / len(top_tickers)
        total_value = self.broker.getvalue()

        for data in self.datas:
            if len(data) == 0:
                continue
            if data._name in top_tickers:
                target_value = total_value * target_pct
                price = data.close[0]
                if price == 0:
                    continue
                target_size = (int(target_value / price) // 100) * 100
                current_size = self.getposition(data).size

                if target_size == 0 and current_size > 0:
                    pass
                elif target_size > current_size:
                    self.buy(data=data, size=target_size - current_size)
                    self.log(f"  BUY {data._name} x{target_size - current_size} @ {price:.2f}")
                elif target_size < current_size:
                    self.sell(data=data, size=current_size - target_size)
                    self.log(f"  SELL {data._name} x{current_size - target_size} @ {price:.2f}")

    def stop(self):
        final_value = self.broker.getvalue()
        self.log(f"Final portfolio value: {final_value:,.2f}")