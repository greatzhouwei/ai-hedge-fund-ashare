"""Pure-Python technical indicators (no TA-Lib dependency)."""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def bbands(series: pd.Series, period: int = 20, nbdev: float = 2.0):
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + nbdev * std
    lower = middle - nbdev * std
    return upper, middle, lower


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Welles Wilder ADX."""
    shift_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - shift_close).abs()
    tr3 = (low - shift_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm_raw = high.diff()
    minus_dm_raw = -low.diff()
    plus_dm = plus_dm_raw.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), 0.0)
    minus_dm = minus_dm_raw.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0.0)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    atr_safe = atr.replace(0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr_safe
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr_safe

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100.0
    adx_series = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx_series
