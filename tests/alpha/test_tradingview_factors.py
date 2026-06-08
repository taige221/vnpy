from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from vnpy.alpha.dataset.utility import calculate_by_expression
from vnpy.alpha.factors import (
    TRADINGVIEW_SKIPPED_INDICATORS,
    TRADINGVIEW_TREND_RSI_FAMILY,
    TRADINGVIEW_TREND_RSI_V2_FAMILY,
    TradingViewFactorSpec,
    tradingview_expressions,
    tradingview_factor_names,
    tradingview_factor_specs,
    tradingview_trend_rsi_factor_specs,
    tradingview_trend_rsi_v2_factor_specs,
)


def make_tradingview_bars(days: int = 160) -> pl.DataFrame:
    """Create a small OHLCV panel for TradingView factor tests."""
    rows: list[dict[str, datetime | float | str]] = []
    start = datetime(2024, 1, 1)

    for index in range(days):
        date = start + timedelta(days=index)
        wave = 1.0 if index % 13 < 7 else -0.6

        for symbol, base in (("UP.TEST", 10.0), ("ALT.TEST", 18.0)):
            close = base + index * 0.08 + wave
            rows.append(
                {
                    "datetime": date,
                    "vt_symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.03,
                    "low": close * 0.97,
                    "close": close,
                    "volume": 1_000_000.0 + index * 1_000.0,
                    "turnover": close * 1_000_000.0,
                    "open_interest": 0.0,
                }
            )

    return pl.DataFrame(rows).sort(["datetime", "vt_symbol"])


def make_rsi_recovery_bars(days: int = 90) -> pl.DataFrame:
    """Create one symbol with a selloff followed by a recovery above RSI 40."""
    rows: list[dict[str, datetime | float | str]] = []
    start = datetime(2024, 1, 1)

    for index in range(days):
        if index < 45:
            close = 100.0 - index * 1.2
        else:
            close = 46.0 + (index - 45) * 1.5
        rows.append(
            {
                "datetime": start + timedelta(days=index),
                "vt_symbol": "RSI.TEST",
                "open": close - 0.4,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 1_000_000.0 + index,
                "turnover": close * (1_000_000.0 + index),
                "open_interest": 0.0,
            }
        )

    return pl.DataFrame(rows)


def test_tradingview_factor_specs_are_named_and_grouped() -> None:
    """Test TradingView factor helpers expose stable families and names."""
    trend_specs = tradingview_trend_rsi_factor_specs()
    v2_specs = tradingview_trend_rsi_v2_factor_specs()
    all_specs = tradingview_factor_specs()

    assert len(trend_specs) == 26
    assert len(v2_specs) == 32
    assert len(all_specs) == 58
    assert all(isinstance(spec, TradingViewFactorSpec) for spec in all_specs)
    assert {spec.family for spec in trend_specs} == {TRADINGVIEW_TREND_RSI_FAMILY}
    assert {spec.family for spec in v2_specs} == {TRADINGVIEW_TREND_RSI_V2_FAMILY}
    assert len(tradingview_factor_names()) == len(set(tradingview_factor_names()))
    assert "tv_trend_rsi_ema_fast" in tradingview_factor_names(TRADINGVIEW_TREND_RSI_FAMILY)
    assert "tv_trend_rsi_v2_rsi_band_height" in tradingview_factor_names(TRADINGVIEW_TREND_RSI_V2_FAMILY)
    assert len(TRADINGVIEW_SKIPPED_INDICATORS) == 3


def test_tradingview_expressions_are_stable() -> None:
    """Test expression helper mirrors the factor specs."""
    specs = tradingview_factor_specs()
    expressions = tradingview_expressions()
    rsi_specs = [
        spec
        for spec in specs
        if spec.source_name in {"rsiVal", "recentOversold", "rsiReturnNow", "rsiRecovered", "plRsi", "phRsi"}
    ]

    assert expressions == [spec.expression for spec in specs]
    assert len(expressions) == len(specs)
    assert all("ts_delay(40.0, 1)" not in expression for expression in expressions)
    assert all("/ 100.0" not in spec.expression for spec in rsi_specs)


def test_tradingview_factor_expressions_calculate() -> None:
    """Test all default TradingView factors run through the alpha DSL."""
    bars = make_tradingview_bars()
    factor_df = bars.select("datetime", "vt_symbol")

    for spec in tradingview_factor_specs():
        values = calculate_by_expression(bars, spec.expression).rename({"data": spec.name})
        factor_df = factor_df.join(values, on=["datetime", "vt_symbol"], how="left")

    assert factor_df.height == bars.height
    assert "tv_trend_rsi_rsi_return_now" in factor_df.columns
    assert "tv_trend_rsi_v2_strong_bull_body" in factor_df.columns
    assert factor_df.get_column("tv_trend_rsi_v2_rsi_band_height").drop_nulls().len() > 0


def test_tradingview_rsi_recovery_signals_use_pine_scale() -> None:
    """Test translated RSI recovery signals can fire on the original 0-100 RSI scale."""
    bars = make_rsi_recovery_bars()
    names = {
        "tv_trend_rsi_recent_oversold",
        "tv_trend_rsi_rsi_return_now",
        "tv_trend_rsi_rsi_recovered",
    }

    for spec in tradingview_factor_specs(TRADINGVIEW_TREND_RSI_FAMILY):
        if spec.name not in names:
            continue
        values = calculate_by_expression(bars, spec.expression)["data"].drop_nulls()
        assert values.max() == 1
