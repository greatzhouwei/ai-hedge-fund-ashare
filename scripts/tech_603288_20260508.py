"""技术面分析师对603288.SH 2026-05-08打分."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
import numpy as np
import math

TS_CODE = "603288.SH"
END_DATE = "20260508"
START_DATE = "20251001"


def safe_float(value, default=0.0):
    try:
        if pd.isna(value) or np.isnan(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default


def calculate_ema(df, window):
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df, period=14):
    df = df.copy()
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]
    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)
    wilder_alpha = 1.0 / period
    df["+di"] = 100 * (
        df["plus_dm"].ewm(alpha=wilder_alpha, adjust=False).mean()
        / df["tr"].ewm(alpha=wilder_alpha, adjust=False).mean()
    )
    df["-di"] = 100 * (
        df["minus_dm"].ewm(alpha=wilder_alpha, adjust=False).mean()
        / df["tr"].ewm(alpha=wilder_alpha, adjust=False).mean()
    )
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(alpha=wilder_alpha, adjust=False).mean()
    return df[["adx", "+di", "-di"]]


def calculate_rsi(prices_df, period=14):
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df, window=20):
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series, max_lag=20):
    lags = range(2, max_lag)
    arr = np.asarray(price_series, dtype=float)
    tau = [max(1e-8, np.sqrt(np.std(arr[lag:] - arr[:-lag]))) for lag in lags]
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0] * 2.0
    except (ValueError, RuntimeWarning):
        return 0.5


def calculate_trend_signals(prices_df):
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)
    adx = calculate_adx(prices_df, 14)
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55
    trend_strength = adx["adx"].iloc[-1] / 100.0
    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": safe_float(adx["adx"].iloc[-1]),
            "trend_strength": safe_float(trend_strength),
            "ema8": safe_float(ema_8.iloc[-1]),
            "ema21": safe_float(ema_21.iloc[-1]),
            "ema55": safe_float(ema_55.iloc[-1]),
        },
    }


def calculate_mean_reversion_signals(prices_df):
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": safe_float(z_score.iloc[-1]),
            "price_vs_bb": safe_float(price_vs_bb),
            "rsi_14": safe_float(rsi_14.iloc[-1]),
            "rsi_28": safe_float(rsi_28.iloc[-1]),
        },
    }


def calculate_momentum_signals(prices_df):
    close = prices_df["close"]
    vol = prices_df["volume"]
    n = len(close)
    mom_1m_val = close.iloc[-1] / close.iloc[-22] - 1 if n >= 22 else np.nan
    mom_3m_val = close.iloc[-1] / close.iloc[-66] - 1 if n >= 66 else np.nan
    mom_6m_val = close.iloc[-1] / close.iloc[-132] - 1 if n >= 132 else np.nan
    vol_20_mean = vol.iloc[-20:].mean() if n >= 20 else np.nan
    vol_60_mean = vol.iloc[-60:].mean() if n >= 60 else np.nan
    volume_momentum_val = vol_20_mean / vol_60_mean if vol_60_mean and vol_60_mean > 0 else np.nan
    momentum_score = 0.4 * mom_1m_val + 0.3 * mom_3m_val + 0.3 * mom_6m_val
    volume_confirmation = volume_momentum_val > 1.0 if not pd.isna(volume_momentum_val) else False
    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
        if not volume_confirmation:
            confidence *= 0.7
    else:
        signal = "neutral"
        confidence = 0.5
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": safe_float(mom_1m_val),
            "momentum_3m": safe_float(mom_3m_val),
            "momentum_6m": safe_float(mom_6m_val),
            "volume_momentum": safe_float(volume_momentum_val),
        },
    }


def calculate_volatility_signals(prices_df):
    returns = prices_df["close"].pct_change()
    hist_vol = returns.rolling(21).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()
    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]
    if current_vol_regime < 0.8 and vol_z < -1:
        signal = "bullish"
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = "bearish"
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": safe_float(hist_vol.iloc[-1]),
            "volatility_regime": safe_float(current_vol_regime),
            "volatility_z_score": safe_float(vol_z),
            "atr_ratio": safe_float(atr_ratio.iloc[-1]),
        },
    }


def calculate_stat_arb_signals(prices_df):
    returns = prices_df["close"].pct_change()
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()
    hurst = calculate_hurst_exponent(prices_df["close"])
    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = "bullish"
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = "bearish"
        confidence = (0.5 - hurst) * 2
    else:
        signal = "neutral"
        confidence = 0.5
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": safe_float(hurst),
            "skewness": safe_float(skew.iloc[-1]),
            "kurtosis": safe_float(kurt.iloc[-1]),
        },
    }


def weighted_signal_combination(signals, weights):
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}
    weighted_sum = 0
    total_confidence = 0
    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]
        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"
    return {"signal": signal, "confidence": abs(final_score)}


def main():
    conn = duckdb.connect("db/tushare_data.db", read_only=True)
    df = conn.execute("""
        SELECT trade_date, open, high, low, close, vol as volume
        FROM daily
        WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date ASC
    """, [TS_CODE, START_DATE, END_DATE]).fetchdf()
    conn.close()

    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.set_index("trade_date")
    df = df.sort_index()

    print("=" * 70)
    print(f"  股票: {TS_CODE} (海天味业)")
    print(f"  日期: 2026-05-08")
    print(f"  价格数据: {len(df)} 个交易日 ({df.index[0].date()} ~ {df.index[-1].date()})")
    print("=" * 70)

    print("\n【1. 趋势跟踪 (Trend Following) — 权重 25%】")
    print("-" * 70)
    trend = calculate_trend_signals(df)
    print(f"  信号: {trend['signal']}")
    print(f"  置信度: {trend['confidence']*100:.1f}%")
    for k, v in trend["metrics"].items():
        print(f"    {k}: {v}")

    print("\n【2. 均值回归 (Mean Reversion) — 权重 20%】")
    print("-" * 70)
    mr = calculate_mean_reversion_signals(df)
    print(f"  信号: {mr['signal']}")
    print(f"  置信度: {mr['confidence']*100:.1f}%")
    for k, v in mr["metrics"].items():
        print(f"    {k}: {v}")

    print("\n【3. 动量 (Momentum) — 权重 25%】")
    print("-" * 70)
    mom = calculate_momentum_signals(df)
    print(f"  信号: {mom['signal']}")
    print(f"  置信度: {mom['confidence']*100:.1f}%")
    for k, v in mom["metrics"].items():
        print(f"    {k}: {v}")

    print("\n【4. 波动率 (Volatility) — 权重 15%】")
    print("-" * 70)
    vol = calculate_volatility_signals(df)
    print(f"  信号: {vol['signal']}")
    print(f"  置信度: {vol['confidence']*100:.1f}%")
    for k, v in vol["metrics"].items():
        print(f"    {k}: {v}")

    print("\n【5. 统计套利 (Statistical Arbitrage) — 权重 15%】")
    print("-" * 70)
    stat = calculate_stat_arb_signals(df)
    print(f"  信号: {stat['signal']}")
    print(f"  置信度: {stat['confidence']*100:.1f}%")
    for k, v in stat["metrics"].items():
        print(f"    {k}: {v}")

    strategy_weights = {
        "trend": 0.25,
        "mean_reversion": 0.20,
        "momentum": 0.25,
        "volatility": 0.15,
        "stat_arb": 0.15,
    }
    combined = weighted_signal_combination(
        {"trend": trend, "mean_reversion": mr, "momentum": mom, "volatility": vol, "stat_arb": stat},
        strategy_weights,
    )

    print("\n" + "=" * 70)
    print("【综合技术面评分】")
    print("=" * 70)

    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}
    for name, key in [("趋势跟踪", "trend"), ("均值回归", "mean_reversion"), ("动量", "momentum"), ("波动率", "volatility"), ("统计套利", "stat_arb")]:
        sig = {"trend": trend, "mean_reversion": mr, "momentum": mom, "volatility": vol, "stat_arb": stat}[key]
        w = strategy_weights[key]
        numeric = signal_values[sig["signal"]]
        conf = sig["confidence"]
        contrib = numeric * w * conf
        print(f"  {name:12s} | {sig['signal']:8s} | conf={conf*100:5.1f}% | weight={w*100:4.0f}% | 贡献={contrib:+.4f}")

    print(f"\n  >>> 综合信号: {combined['signal']}")
    print(f"  >>> 综合置信度: {combined['confidence']*100:.1f}%")
    print(f"  >>> 综合得分: {combined['confidence']:.4f}")


if __name__ == "__main__":
    main()
