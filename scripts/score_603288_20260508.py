"""对海天味业(603288.SH)在2026-05-08进行三维度打分并输出明细."""

import json
from pathlib import Path

import duckdb

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.jq_adapter import JQDataAdapter
from src.backtesting.jq_screener import (
    score_fundamentals,
    score_growth,
    score_technical,
    combine_scores,
    WEIGHTS,
)

DB_PATH = Path("src/data/tushare_data.db")
TS_CODE = "605286.SH"
DATE = "2025-01-02"


def main():
    adapter = JQDataAdapter(str(DB_PATH))
    tickers = [TS_CODE]

    print("=" * 70)
    print(f"  股票: {TS_CODE} (同力日升)")
    print(f"  日期: {DATE}")
    print("=" * 70)

    # 1. 基本面打分
    print("\n【1. 基本面打分 (权重 35%)】")
    print("-" * 70)
    fund_scores, fund_details = score_fundamentals(adapter, tickers, DATE)
    fd = fund_details.get(TS_CODE, {})

    prof = fd.get("profitability", {})
    print(f"  盈利能力:")
    print(f"    ROE           = {prof.get('roe')}")
    print(f"    净利率        = {prof.get('net_margin')}")
    print(f"    营业利润率    = {prof.get('op_margin')}")
    print(f"    信号          = {prof.get('signal')}")

    gr = fd.get("growth", {})
    print(f"\n  成长性:")
    print(f"    营收增长      = {gr.get('rev_growth')}")
    print(f"    净利润增长    = {gr.get('np_growth')}")
    print(f"    账面价值增长  = {gr.get('bv_growth')}")
    print(f"    信号          = {gr.get('signal')}")

    hl = fd.get("health", {})
    print(f"\n  财务健康:")
    print(f"    流动比率      = {hl.get('curr_ratio')}")
    print(f"    负债权益比    = {hl.get('de_ratio')}")
    print(f"    FCF/股        = {hl.get('fcf_ps')}")
    print(f"    EPS           = {hl.get('eps')}")
    print(f"    信号          = {hl.get('signal')}")

    vl = fd.get("valuation", {})
    print(f"\n  估值:")
    print(f"    PE (TTM)      = {vl.get('pe')}")
    print(f"    PB            = {vl.get('pb')}")
    print(f"    PS (TTM)      = {vl.get('ps')}")
    print(f"    信号          = {vl.get('signal')}")

    fund_score = fund_scores.get(TS_CODE, 0)
    print(f"\n  >>> 基本面总分  = {fund_score:.4f} (满分 1.0)")

    # 2. 成长性打分
    print("\n【2. 成长性打分 (权重 40%)】")
    print("-" * 70)
    growth_scores, growth_details = score_growth(adapter, tickers, DATE)
    gd = growth_details.get(TS_CODE, {})

    print(f"  营收增长        = {gd.get('revenue_growth')}")
    print(f"  营收趋势斜率    = {gd.get('revenue_trend')}")
    print(f"  EPS增长         = {gd.get('eps_growth')}")
    print(f"  EPS趋势斜率     = {gd.get('eps_trend')}")
    print(f"  FCF增长         = {gd.get('fcf_growth')}")
    print(f"  FCF趋势斜率     = {gd.get('fcf_trend')}")
    print(f"\n  毛利率          = {gd.get('gross_margin')}")
    print(f"  毛利率趋势      = {gd.get('gm_trend')}")
    print(f"  营业利润率      = {gd.get('operating_margin')}")
    print(f"  营业利润率趋势  = {gd.get('om_trend')}")
    print(f"  净利率          = {gd.get('net_margin')}")
    print(f"  净利率趋势      = {gd.get('nm_trend')}")
    print(f"\n  负债权益比      = {gd.get('debt_to_equity')}")
    print(f"  流动比率        = {gd.get('current_ratio')}")
    print(f"  PEG             = {gd.get('peg')}")
    print(f"  PS (TTM)        = {gd.get('ps')}")

    print(f"\n  子项得分:")
    print(f"    增长趋势得分  = {gd.get('growth_score', 0):.4f} (权重 50%)")
    print(f"    估值得分      = {gd.get('val_score', 0):.4f} (权重 20%)")
    print(f"    利润率得分    = {gd.get('margin_score', 0):.4f} (权重 20%)")
    print(f"    财务健康得分  = {gd.get('health_score', 0):.4f} (权重 10%)")

    growth_score = growth_scores.get(TS_CODE, 0)
    print(f"\n  >>> 成长性总分  = {growth_score:.4f} (满分 1.0)")
    print(f"      raw_score={gd.get('raw_score')}, sustainability={gd.get('sustainability')}")
    print(f"      signal={gd.get('signal')}, confidence={gd.get('confidence')}%")

    # 3. 技术面打分
    print("\n【3. 技术面打分 (权重 25%)】")
    print("-" * 70)
    tech_scores, tech_details = score_technical(adapter, tickers, DATE)
    td = tech_details.get(TS_CODE, {})

    if td:
        print(f"  ADX(14)         = {td.get('adx')}")
        print(f"  EMA8            = {td.get('ema8')}")
        print(f"  EMA21           = {td.get('ema21')}")
        print(f"  EMA55           = {td.get('ema55')}")
        print(f"  趋势强度分      = {td.get('tf_bullish')}")
        print(f"\n  RSI(14)         = {td.get('rsi14')}")
        print(f"  布林带位置      = {td.get('price_vs_bb')}")
        print(f"  均值回归分      = {td.get('mr_bullish')}")
        print(f"\n  3月动量         = {td.get('mom_3m')}")
        print(f"  6月动量         = {td.get('mom_6m')}")
        print(f"  动量分          = {td.get('mom_bullish')}")
        print(f"\n  波动率状态      = {td.get('vol_regime')}")
        print(f"  波动率分        = {td.get('vol_bullish')}")

        tech_score = tech_scores.get(TS_CODE, 0)
        print(f"\n  >>> 技术面总分  = {tech_score:.4f} (满分 1.0)")
    else:
        print("  (无足够价格数据)")
        tech_score = 0.5

    # 综合
    print("\n" + "=" * 70)
    print("【综合评分】")
    print("=" * 70)

    combined = combine_scores(fund_scores, growth_scores, tech_scores, WEIGHTS)
    total = combined.get(TS_CODE, 0)

    print(f"  基本面得分      = {fund_score:.4f} × {WEIGHTS['fundamentals']} = {fund_score * WEIGHTS['fundamentals']:.4f}")
    print(f"  成长性得分      = {growth_score:.4f} × {WEIGHTS['growth']} = {growth_score * WEIGHTS['growth']:.4f}")
    print(f"  技术面得分      = {tech_score:.4f} × {WEIGHTS['technical']} = {tech_score * WEIGHTS['technical']:.4f}")
    print(f"\n  >>> 综合总分    = {total:.4f}")
    print(f"      排名信号: {'bullish' if total > 0.6 else 'bearish' if total < 0.4 else 'neutral'}")

    adapter.close()


if __name__ == "__main__":
    main()
