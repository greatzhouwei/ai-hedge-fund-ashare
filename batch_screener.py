"""Batch stock screener for top 200 A-shares by market cap.

Pipeline:
1. Fetch all A-share stocks, filter out GEM/STAR/BSE/ST
2. Get market cap on target date, sort desc, keep top 200
3. Analyze each with 4 quant analysts (fundamentals, growth, technical, valuation)
4. Parallel with 4 processes, resumable via progress log
5. Save results to JSON + CSV
"""

import io
import os
import sys
import json
import csv
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from dotenv import load_dotenv
load_dotenv("D:/app/hedge-fund/.env")

import tushare as ts
from src.data.cache import get_cache

# ---------------------------------------------------------------------------
# Analyst imports
# ---------------------------------------------------------------------------
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
    "fundamentals": 0.30,
    "growth": 0.25,
    "technical": 0.20,
    "valuation": 0.25,
}

SIGNAL_MAP = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_DATE = "2026-04-30"
TOP_N = 500
WORKERS = 4
OUTPUT_DIR = Path("batch_screener_results")
PROGRESS_FILE = OUTPUT_DIR / "progress.jsonl"
RESULT_FILE = OUTPUT_DIR / "results.jsonl"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
DETAIL_CSV = OUTPUT_DIR / "detail.csv"

# ---------------------------------------------------------------------------
# Step 1: build universe
# ---------------------------------------------------------------------------
def fetch_and_filter_universe() -> list[dict]:
    """Return list of {ts_code, name, industry} after filtering."""
    api_key = os.environ.get("TUSHARE_TOKEN")
    pro = ts.pro_api(api_key)

    print("[Step 1] Fetching stock basic info ...")
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,name,area,industry,market,list_date")
    print(f"  Total listed: {len(df)}")

    excluded = ("300", "301", "688", "8", "4", "43")
    st_kw = ("ST", "*ST")

    filtered = []
    for _, row in df.iterrows():
        code = row["ts_code"]
        prefix = code.split(".")[0]
        if prefix.startswith(excluded):
            continue
        if any(k in row["name"] for k in st_kw):
            continue
        filtered.append({
            "ts_code": code,
            "name": row["name"],
            "industry": row["industry"],
        })

    print(f"  After filtering GEM/STAR/BSE/ST: {len(filtered)}")
    return filtered


def fetch_market_caps(stocks: list[dict], trade_date: str) -> list[dict]:
    """Attach total_mv (total market cap in 万元) to each stock dict."""
    api_key = os.environ.get("TUSHARE_TOKEN")
    pro = ts.pro_api(api_key)
    td = trade_date.replace("-", "")

    print(f"[Step 2] Fetching market cap for {trade_date} ...")
    # daily_basic returns 5000 rows max per call; A-share total ~5000, one call OK
    df = pro.daily_basic(trade_date=td, fields="ts_code,total_mv")
    if df is None or df.empty:
        raise RuntimeError(f"daily_basic returned empty for {trade_date}")

    mv_map = {row["ts_code"]: row["total_mv"] for _, row in df.iterrows()}

    enriched = []
    for s in stocks:
        mv = mv_map.get(s["ts_code"])
        if mv is not None and mv > 0:
            enriched.append({**s, "total_mv": float(mv)})

    enriched.sort(key=lambda x: x["total_mv"], reverse=True)
    print(f"  Stocks with valid market cap: {len(enriched)}")
    return enriched


# ---------------------------------------------------------------------------
# Step 3: analyze single ticker
# ---------------------------------------------------------------------------
def build_state(ticker: str, end_date: str) -> dict:
    return {
        "messages": [],
        "data": {
            "tickers": [ticker],
            "end_date": end_date,
            "start_date": "2025-01-01",
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


def analyze_ticker(ticker: str, end_date: str) -> dict:
    """Run 4 quant analysts for one ticker and return signals + score."""
    get_cache()._financial_metrics_cache.clear()

    signals = {}
    for name, agent_func in AGENTS.items():
        state = build_state(ticker, end_date)
        try:
            agent_func(state, agent_id=f"{name}_agent")
            sig_data = state["data"]["analyst_signals"].get(f"{name}_agent", {}).get(ticker, {})
            signals[name] = {
                "signal": sig_data.get("signal", "neutral"),
                "confidence": sig_data.get("confidence", 0) or 0,
            }
        except Exception as e:
            signals[name] = {"signal": "neutral", "confidence": 0, "error": str(e)}

    # compute score
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


# ---------------------------------------------------------------------------
# Worker wrapper — no file I/O, results written by main process
# ---------------------------------------------------------------------------
def worker_init():
    """Reset stdout encoding in forked processes on Windows."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)


def worker_task(args) -> dict:
    """args = (ticker, name, industry, total_mv, end_date)"""
    ticker, name, industry, total_mv, end_date = args
    t0 = time.time()
    try:
        result = analyze_ticker(ticker, end_date)
        result["name"] = name
        result["industry"] = industry
        result["total_mv"] = total_mv
        elapsed = round(time.time() - t0, 1)
        result["elapsed_sec"] = elapsed
        print(f"  [OK] {ticker} ({name})  score={result['score']:+.4f}  {elapsed}s")
        return result
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        err = {"ticker": ticker, "name": name, "error": str(e), "elapsed_sec": elapsed}
        print(f"  [ERR] {ticker} ({name})  {e}  {elapsed}s")
        return err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_done_tickers(path: Path) -> set[str]:
    """Load tickers that already have a result written."""
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


def export_csv(results: list[dict]):
    """Export summary and detail CSVs."""
    # summary
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "ticker", "name", "industry", "total_mv_wan", "score", "recommendation"])
        for i, r in enumerate(results, 1):
            s = r.get("score", 0)
            action = "BUY" if s >= 0.3 else "WATCH" if s >= 0.1 else "HOLD" if s >= -0.1 else "AVOID"
            w.writerow([i, r["ticker"], r.get("name", ""), r.get("industry", ""),
                        r.get("total_mv", ""), s, action])

    # detail
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "name", "dimension", "signal", "confidence", "contribution", "total_score"])
        for r in results:
            for dim, data in r.get("details", {}).items():
                w.writerow([r["ticker"], r.get("name", ""), dim,
                            data["signal"], data["confidence"], data["contribution"], r.get("score", 0)])

    print(f"\n[Export] Summary -> {SUMMARY_CSV}")
    print(f"[Export] Detail  -> {DETAIL_CSV}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Universe
    universe = fetch_and_filter_universe()

    # 2. Market cap
    enriched = fetch_market_caps(universe, TARGET_DATE)
    top_stocks = enriched[:TOP_N]
    print(f"  Top {TOP_N} selected. Smallest mv: {top_stocks[-1]['total_mv']:,.0f} 万元")

    # Save top200 list
    with open(OUTPUT_DIR / "top200.json", "w", encoding="utf-8") as f:
        json.dump(top_stocks, f, ensure_ascii=False, indent=2)

    # 3. Resume support — check results.jsonl for already-completed tickers
    done_tickers = load_done_tickers(RESULT_FILE)
    print(f"\n[Resume] Already done: {len(done_tickers)} / {TOP_N}")

    pending = [s for s in top_stocks if s["ts_code"] not in done_tickers]
    if not pending:
        print("All stocks already analyzed. Skipping to export.")
    else:
        print(f"[Run] Analyzing {len(pending)} stocks with {WORKERS} workers ...\n")
        task_args = [
            (s["ts_code"], s["name"], s["industry"], s["total_mv"], TARGET_DATE)
            for s in pending
        ]

        # Open result / progress files once and let the main process write sequentially
        result_f = open(RESULT_FILE, "a", encoding="utf-8")
        progress_f = open(PROGRESS_FILE, "a", encoding="utf-8")

        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(WORKERS, initializer=worker_init) as pool:
                for result in pool.imap_unordered(worker_task, task_args):
                    ticker = result.get("ticker", "unknown")
                    elapsed = result.get("elapsed_sec", 0)
                    status = "done" if "score" in result else "error"
                    error_msg = result.get("error", "")

                    result_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    result_f.flush()

                    progress_f.write(
                        json.dumps(
                            {"ticker": ticker, "status": status, "elapsed": elapsed, "error": error_msg},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    progress_f.flush()
        finally:
            result_f.close()
            progress_f.close()

    # 4. Load all results and export
    print("\n[Step 4] Loading results and exporting ...")
    all_results = []
    if RESULT_FILE.exists():
        with open(RESULT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except Exception:
                        pass

    all_results.sort(key=lambda x: x.get("score", -999), reverse=True)
    export_csv(all_results)

    # 5. Final report
    print("\n" + "=" * 70)
    print("【筛选结果 TOP 20】")
    print("=" * 70)
    for i, r in enumerate(all_results[:20], 1):
        s = r.get("score", 0)
        action = "BUY" if s >= 0.3 else "WATCH" if s >= 0.1 else "HOLD" if s >= -0.1 else "AVOID"
        print(f"{i:3d}. {r['ticker']} {r.get('name',''):8s}  score={s:+.4f}  {action}")

    print(f"\nTotal analyzed: {len(all_results)}")
    print(f"Output dir: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
