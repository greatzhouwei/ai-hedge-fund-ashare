"""Backtest screening: score stocks on a given date and save results.

Usage:
    poetry run python run_backtest_by_date.py --target-date 2024-02-05
"""
import argparse
import io
import os
import sys
import json
import time
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from dotenv import load_dotenv
load_dotenv("D:/app/hedge-fund/.env")

# Use read-only DB copy to avoid locking the active sync DB
import os
os.environ["TUSHARE_DB_PATH"] = str(Path("src/data/tushare_data.db").resolve())

import duckdb
from src.data.cache import get_cache

from src.agents.fundamentals import fundamentals_analyst_agent
from src.agents.growth_agent import growth_analyst_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.valuation import valuation_analyst_agent

AGENTS = {
    "fundamentals": fundamentals_analyst_agent,
    "growth": growth_analyst_agent,
    "technical": technical_analyst_agent,
    "valuation": valuation_analyst_agent,
}

WEIGHTS = {
    "fundamentals": 0.25,
    "growth": 0.30,
    "technical": 0.25,
    "valuation": 0.20,
}

SIGNAL_MAP = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

WORKERS = 4
OUTPUT_DIR = Path("batch_screener_results")


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest stock screening on a target date")
    parser.add_argument("--target-date", required=True, help="Target date in YYYY-MM-DD format")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    return parser.parse_args()


def load_tickers_from_fina() -> list[str]:
    with open("batch_screener_results/ok465_tickers.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_state(ticker: str, end_date: str, start_date: str) -> dict:
    return {
        "messages": [],
        "data": {
            "tickers": [ticker],
            "end_date": end_date,
            "start_date": start_date,
            "analyst_signals": {},
            "portfolio": {
                "cash": 0.0,
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 0.0,
                "positions": {},
                "realized_gains": {},
            },
        },
        "metadata": {
            "show_reasoning": False,
            "model_name": "kimi-k2.5",
            "model_provider": "Moonshot",
        },
    }


def analyze_ticker(ticker: str, end_date: str, start_date: str) -> dict:
    get_cache()._financial_metrics_cache.clear()
    signals = {}
    for name, agent_func in AGENTS.items():
        state = build_state(ticker, end_date, start_date)
        try:
            agent_func(state, agent_id=f"{name}_agent")
            sig_data = state["data"]["analyst_signals"].get(f"{name}_agent", {}).get(ticker, {})
            signals[name] = {
                "signal": sig_data.get("signal", "neutral"),
                "confidence": sig_data.get("confidence", 0) or 0,
            }
        except Exception as e:
            signals[name] = {"signal": "neutral", "confidence": 0, "error": str(e)}

    details = {}
    total = 0.0
    for dim, weight in WEIGHTS.items():
        s = signals[dim]
        val = SIGNAL_MAP.get(s["signal"], 0.0)
        conf = s.get("confidence", 0) or 0
        contrib = val * (conf / 100.0) * weight
        total += contrib
        details[dim] = {
            "signal": s["signal"],
            "confidence": conf,
            "contribution": round(contrib, 4),
        }

    return {
        "ticker": ticker,
        "signals": signals,
        "score": round(total, 4),
        "details": details,
        "analyzed_at": datetime.now().isoformat(),
    }


def worker_init():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)


def worker_task(args) -> dict:
    ticker, end_date, start_date = args
    t0 = time.time()
    try:
        result = analyze_ticker(ticker, end_date, start_date)
        elapsed = round(time.time() - t0, 1)
        result["elapsed_sec"] = elapsed
        print(f"  [OK] {ticker}  score={result['score']:+.4f}  {elapsed}s")
        return result
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        err = {"ticker": ticker, "error": str(e), "elapsed_sec": elapsed}
        print(f"  [ERR] {ticker}  {e}  {elapsed}s")
        return err


def load_done_tickers(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "ticker" in rec and "score" in rec:
                    done.add(rec["ticker"])
            except Exception:
                pass
    return done


def main():
    args = parse_args()
    target_date = args.target_date
    workers = args.workers

    # start_date = 1 year before target_date
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start_dt = target_dt - timedelta(days=365)
    start_date = start_dt.strftime("%Y-%m-%d")

    result_file = OUTPUT_DIR / f"backtest_{target_date.replace('-', '')}.jsonl"

    OUTPUT_DIR.mkdir(exist_ok=True)
    tickers = load_tickers_from_fina()
    print(f"Target date: {target_date}  |  Start date: {start_date}")
    print(f"Loaded {len(tickers)} tickers from fina_indicator")

    done = load_done_tickers(result_file)
    print(f"Already done: {len(done)}")

    pending = [t for t in tickers if t not in done]
    if not pending:
        print("All done.")
        return

    print(f"Analyzing {len(pending)} stocks with {workers} workers ...\n")

    with open(result_file, "a", encoding="utf-8") as f:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=worker_init) as pool:
            task_args = [(t, target_date, start_date) for t in pending]
            for result in pool.imap_unordered(worker_task, task_args):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    print(f"\nDone. Results saved to {result_file}")


if __name__ == "__main__":
    main()
