import json
from collections import defaultdict
import statistics

files = {
    '2024-02-05 (熊市底)': 'batch_screener_results/backtest_20240205.jsonl',
    '2024-06-28 (震荡回调)': 'batch_screener_results/backtest_20240628.jsonl',
    '2024-09-24 (牛市启动)': 'batch_screener_results/backtest_20240924.jsonl',
}

for label, path in files.items():
    scores = []
    all_bullish = []
    all_bearish = []
    bullish_counts = defaultdict(int)
    bearish_counts = defaultdict(int)
    neutral_counts = defaultdict(int)
    errors = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if 'error' in rec and 'score' not in rec:
                errors += 1
                continue
            score = rec.get('score', 0)
            scores.append(score)

            signals = rec.get('signals', {})
            dims = ['fundamentals', 'growth', 'technical', 'valuation']
            sigs = [signals.get(d, {}).get('signal', 'neutral') for d in dims]

            if all(s == 'bullish' for s in sigs):
                all_bullish.append((rec['ticker'], score))
            if all(s == 'bearish' for s in sigs):
                all_bearish.append((rec['ticker'], score))

            for d in dims:
                s = signals.get(d, {}).get('signal', 'neutral')
                if s == 'bullish':
                    bullish_counts[d] += 1
                elif s == 'bearish':
                    bearish_counts[d] += 1
                else:
                    neutral_counts[d] += 1

    n = len(scores)
    mean = statistics.mean(scores)
    med = statistics.median(scores)
    std = statistics.stdev(scores) if n > 1 else 0

    print(f'\n=== {label} ===')
    print(f'  有效样本: {n}  |  报错: {errors}')
    print(f'  Score 均值: {mean:+.4f}  中位数: {med:+.4f}  标准差: {std:.4f}')
    print(f'  Score 范围: [{min(scores):+.4f}, {max(scores):+.4f}]')
    print(f'  Score > +0.3: {sum(1 for s in scores if s > 0.3)} 只')
    print(f'  Score < -0.3: {sum(1 for s in scores if s < -0.3)} 只')
    print(f'  全 bullish: {len(all_bullish)} 只  平均得分: {statistics.mean([s for _, s in all_bullish]) if all_bullish else 0:+.4f}')
    print(f'  全 bearish: {len(all_bearish)} 只  平均得分: {statistics.mean([s for _, s in all_bearish]) if all_bearish else 0:+.4f}')
    print('  各维度 bullish 比例:')
    for d in ['fundamentals', 'growth', 'technical', 'valuation']:
        total = bullish_counts[d] + bearish_counts[d] + neutral_counts[d]
        if total > 0:
            print(f'    {d:14s}: bullish={bullish_counts[d]/total:.1%}  bearish={bearish_counts[d]/total:.1%}  neutral={neutral_counts[d]/total:.1%}')

    if all_bullish:
        all_bullish.sort(key=lambda x: -x[1])
        print(f'  全 bullish TOP 5: {all_bullish[:5]}')
    if all_bearish:
        all_bearish.sort(key=lambda x: x[1])
        print(f'  全 bearish BOTTOM 5: {all_bearish[:5]}')
