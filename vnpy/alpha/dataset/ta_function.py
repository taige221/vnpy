"""
Technical Analysis Operators
"""

from collections.abc import Callable
from typing import Literal

import talib
import polars as pl
import pandas as pd

from .utility import DataProxy


MacdComponent = Literal["dif", "dea", "hist"]


def _result_from_pandas(frame: pd.DataFrame) -> DataProxy:
    """Build a DataProxy from a pandas result frame in original row order."""
    result = frame.sort_values("_row_nr")[["datetime", "vt_symbol", "data"]]
    return DataProxy(pl.from_pandas(result))


def _apply_unary_by_symbol(
    feature: DataProxy,
    func: Callable[[pd.Series], object],
) -> DataProxy:
    """Apply a TA-Lib unary function independently for each symbol."""
    source = feature.df.with_row_index("_row_nr").to_pandas()
    rows: list[pd.DataFrame] = []

    for _, group in source.groupby("vt_symbol", sort=False):
        ordered = group.sort_values("datetime").copy()
        series = pd.to_numeric(ordered["data"], errors="coerce")
        values = func(series)

        output = ordered[["_row_nr", "datetime", "vt_symbol"]].copy()
        output["data"] = pd.Series(values, index=ordered.index).to_numpy()
        rows.append(output)

    return _result_from_pandas(pd.concat(rows, ignore_index=True))


def _join_ohlc(high: DataProxy, low: DataProxy, close: DataProxy) -> pd.DataFrame:
    """Join high, low, and close proxies while preserving high's row order."""
    return (
        high.df.with_row_index("_row_nr")
        .rename({"data": "high"})
        .join(low.df.rename({"data": "low"}), on=["datetime", "vt_symbol"], how="inner")
        .join(close.df.rename({"data": "close"}), on=["datetime", "vt_symbol"], how="inner")
        .to_pandas()
    )


def _apply_ohlc_by_symbol(
    high: DataProxy,
    low: DataProxy,
    close: DataProxy,
    func: Callable[[pd.Series, pd.Series, pd.Series], object],
) -> DataProxy:
    """Apply a TA-Lib OHLC function independently for each symbol."""
    source = _join_ohlc(high, low, close)
    rows: list[pd.DataFrame] = []

    for _, group in source.groupby("vt_symbol", sort=False):
        ordered = group.sort_values("datetime").copy()
        high_series = pd.to_numeric(ordered["high"], errors="coerce")
        low_series = pd.to_numeric(ordered["low"], errors="coerce")
        close_series = pd.to_numeric(ordered["close"], errors="coerce")
        values = func(high_series, low_series, close_series)

        output = ordered[["_row_nr", "datetime", "vt_symbol"]].copy()
        output["data"] = pd.Series(values, index=ordered.index).to_numpy()
        rows.append(output)

    return _result_from_pandas(pd.concat(rows, ignore_index=True))


def _macd_component(
    close: DataProxy,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    component: MacdComponent,
) -> DataProxy:
    """Return one MACD component calculated independently for each symbol."""
    def calculate(series: pd.Series) -> object:
        values = series.to_numpy(dtype="float64")
        dif, dea, hist = talib.MACD(
            values,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period,
        )
        if component == "dif":
            return dif
        if component == "dea":
            return dea
        return hist

    return _apply_unary_by_symbol(close, calculate)


def ta_rsi(close: DataProxy, window: int) -> DataProxy:
    """Calculate RSI indicator by contract"""
    return _apply_unary_by_symbol(
        close,
        lambda series: talib.RSI(series.to_numpy(dtype="float64"), timeperiod=window),
    )


def ta_atr(high: DataProxy, low: DataProxy, close: DataProxy, window: int) -> DataProxy:
    """Calculate ATR indicator by contract"""
    return _apply_ohlc_by_symbol(
        high,
        low,
        close,
        lambda high_series, low_series, close_series: talib.ATR(
            high_series.to_numpy(dtype="float64"),
            low_series.to_numpy(dtype="float64"),
            close_series.to_numpy(dtype="float64"),
            timeperiod=window,
        ),
    )


def ta_ema(close: DataProxy, window: int) -> DataProxy:
    """Calculate EMA indicator by contract."""
    return _apply_unary_by_symbol(
        close,
        lambda series: talib.EMA(series.to_numpy(dtype="float64"), timeperiod=window),
    )


def ta_macd_dif(close: DataProxy, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> DataProxy:
    """Calculate MACD DIF line by contract."""
    return _macd_component(close, fast_period, slow_period, signal_period, "dif")


def ta_macd_dea(close: DataProxy, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> DataProxy:
    """Calculate MACD DEA/signal line by contract."""
    return _macd_component(close, fast_period, slow_period, signal_period, "dea")


def ta_macd_hist(close: DataProxy, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> DataProxy:
    """Calculate MACD histogram by contract."""
    return _macd_component(close, fast_period, slow_period, signal_period, "hist")
