from __future__ import annotations

from datetime import datetime, timedelta
from numbers import Real

import polars as pl

from vnpy.alpha.dataset.utility import calculate_by_expression
from vnpy.alpha.factors import (
    common_trade_filter_expressions,
    common_trade_filter_factor_names,
    common_trade_filter_factor_specs,
    common_technical_expressions,
    common_technical_factor_names,
    common_technical_factor_specs,
)


def make_directional_bars(days: int = 40) -> pl.DataFrame:
    """Create two symbols with opposite trends for TA operator tests."""
    start = datetime(2024, 1, 1)
    rows: list[dict[str, float | str | datetime]] = []

    for day_index in range(days):
        current_date = start + timedelta(days=day_index)
        up_close = 10.0 + day_index * 0.2
        down_close = 20.0 - day_index * 0.2

        for vt_symbol, close in (("DOWN.TEST", down_close), ("UP.TEST", up_close)):
            rows.append(
                {
                    "datetime": current_date,
                    "vt_symbol": vt_symbol,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000.0 + day_index * 10_000.0,
                    "turnover": close * 1_000_000.0,
                    "open_interest": 0.0,
                    "vwap": close,
                }
            )

    return pl.DataFrame(rows).sort(["datetime", "vt_symbol"])


def test_ta_operators_calculate_by_symbol() -> None:
    """Test TA operators do not leak values across symbols."""
    bars = make_directional_bars()
    last_datetime = bars["datetime"].max()

    rsi = calculate_by_expression(bars, "ta_rsi(close, 2)")
    rsi_last = dict(
        zip(
            rsi.filter(pl.col("datetime") == last_datetime)["vt_symbol"].to_list(),
            rsi.filter(pl.col("datetime") == last_datetime)["data"].to_list(),
            strict=True,
        )
    )

    macd = calculate_by_expression(bars, "ta_macd_hist(close, 3, 6, 2)")
    macd_last = dict(
        zip(
            macd.filter(pl.col("datetime") == last_datetime)["vt_symbol"].to_list(),
            macd.filter(pl.col("datetime") == last_datetime)["data"].to_list(),
            strict=True,
        )
    )

    assert rsi_last["UP.TEST"] > 99.0
    assert rsi_last["DOWN.TEST"] < 1.0
    assert macd_last["UP.TEST"] > 0
    assert macd_last["DOWN.TEST"] < 0


def test_common_technical_factor_specs_are_named_and_unique() -> None:
    """Test common technical factor helpers expose stable names."""
    specs = common_technical_factor_specs(windows=(3,), macd_periods=(3, 6, 2))
    names = common_technical_factor_names(windows=(3,), macd_periods=(3, 6, 2))
    expressions = common_technical_expressions(windows=(3,), macd_periods=(3, 6, 2))

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "common_ma_bias_3" in names
    assert "common_volume_ratio_3" in names
    assert "common_macd_hist_3_6_2" in names


def test_common_technical_expressions_calculate() -> None:
    """Test common technical expressions run through the existing alpha DSL."""
    bars = make_directional_bars()
    specs = common_technical_factor_specs(windows=(3,), macd_periods=(3, 6, 2))
    factor_df = bars.select("datetime", "vt_symbol")

    for spec in specs:
        values = calculate_by_expression(bars, spec.expression).rename({"data": spec.name})
        factor_df = factor_df.join(values, on=["datetime", "vt_symbol"], how="left")

    assert factor_df.height == bars.height
    assert "common_rsi_3" in factor_df.columns
    assert "common_atr_pct_3" in factor_df.columns
    assert factor_df.get_column("common_macd_hist_3_6_2").drop_nulls().len() > 0
    max_bullish_state = factor_df.get_column("common_macd_bullish_state_3_6_2").drop_nulls().max()
    assert isinstance(max_bullish_state, Real)
    assert max_bullish_state <= 1


def test_common_trade_filter_factor_specs_are_optional_panel_factors() -> None:
    """Test reusable trade-filter factors are separate from default OHLCV factors."""
    specs = common_trade_filter_factor_specs(ma_window=3, min_ma_bias=0.02, min_turnover_rate=3.0, max_turnover_rate=6.5)
    names = common_trade_filter_factor_names(ma_window=3, min_ma_bias=0.02, min_turnover_rate=3.0, max_turnover_rate=6.5)
    expressions = common_trade_filter_expressions(ma_window=3, min_ma_bias=0.02, min_turnover_rate=3.0, max_turnover_rate=6.5)

    assert names == [spec.name for spec in specs]
    assert len(expressions) == len(set(expressions))
    assert "common_ma_bias_pass_3" in names
    assert "common_turnover_rate_band" in names
    assert "turnover_rate" not in common_technical_factor_names(windows=(3,), macd_periods=(3, 6, 2))


def test_common_trade_filter_expressions_calculate_on_enriched_panel() -> None:
    """Test optional common trade filters run when turnover_rate is present."""
    bars = make_directional_bars().with_columns(pl.lit(4.0).alias("turnover_rate"))
    specs = common_trade_filter_factor_specs(ma_window=3, min_ma_bias=0.005, min_turnover_rate=3.0, max_turnover_rate=6.5)
    factor_df = bars.select("datetime", "vt_symbol")

    for spec in specs:
        values = calculate_by_expression(bars, spec.expression).rename({"data": spec.name})
        factor_df = factor_df.join(values, on=["datetime", "vt_symbol"], how="left")

    assert factor_df.get_column("common_turnover_rate_band").drop_nulls().max() == 1
    assert factor_df.get_column("common_ma_bias_pass_3").drop_nulls().max() == 1
