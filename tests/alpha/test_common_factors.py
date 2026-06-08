from __future__ import annotations

from datetime import datetime, timedelta
from numbers import Real

import polars as pl

from vnpy.alpha.dataset.utility import calculate_by_expression
from vnpy.alpha.factors import (
    CommonFactorSpec,
    common_a_share_limit_state_expressions,
    common_a_share_limit_state_factor_names,
    common_a_share_limit_state_factor_specs,
    common_candlestick_expressions,
    common_candlestick_factor_names,
    common_candlestick_factor_specs,
    common_fundamental_expressions,
    common_fundamental_factor_names,
    common_fundamental_factor_specs,
    common_gmma_expressions,
    common_gmma_factor_names,
    common_gmma_factor_specs,
    common_liquidity_crowding_expressions,
    common_liquidity_crowding_factor_names,
    common_liquidity_crowding_factor_specs,
    common_oscillator_expressions,
    common_oscillator_factor_names,
    common_oscillator_factor_specs,
    common_position_breakout_expressions,
    common_position_breakout_factor_names,
    common_position_breakout_factor_specs,
    common_price_volume_accumulation_expressions,
    common_price_volume_accumulation_factor_names,
    common_price_volume_accumulation_factor_specs,
    common_relative_strength_expressions,
    common_relative_strength_factor_names,
    common_relative_strength_factor_specs,
    common_trade_filter_expressions,
    common_trade_filter_factor_names,
    common_trade_filter_factor_specs,
    common_technical_expressions,
    common_technical_factor_names,
    common_technical_factor_specs,
    common_vegas_expressions,
    common_vegas_factor_names,
    common_vegas_factor_specs,
    common_volatility_compression_expressions,
    common_volatility_compression_factor_names,
    common_volatility_compression_factor_specs,
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


def make_enriched_common_panel(days: int = 40) -> pl.DataFrame:
    """Create an enriched panel with optional A-share and relative-strength columns."""
    bars = make_directional_bars(days)
    return bars.with_columns(
        pl.col("open").alias("raw_open"),
        pl.col("high").alias("raw_high"),
        pl.col("low").alias("raw_low"),
        pl.col("close").alias("raw_close"),
        pl.col("close").alias("up_limit"),
        (pl.col("close") * 0.9).alias("down_limit"),
        pl.lit(4.0).alias("turnover_rate"),
        (pl.col("close") * 0.98 + 1.0).alias("benchmark_close"),
        (pl.col("close") * 1.02 + 0.5).alias("industry_close"),
        (pl.col("close") * 0.12).alias("pb"),
        (pl.col("close") * 0.8).alias("pe_ttm"),
        (pl.col("close") * 0.3).alias("ps_ttm"),
        pl.lit(2.0).alias("dv_ttm"),
        (pl.col("close") * 10_000.0).alias("total_mv"),
        (pl.col("close") * 8_000.0).alias("circ_mv"),
        (pl.col("close") * 0.01).alias("fi_roe"),
        (pl.col("close") * 0.02).alias("fi_grossprofit_margin"),
        pl.lit(35.0).alias("fi_debt_to_assets"),
        (pl.col("close") * 0.05).alias("fi_netprofit_yoy"),
        (pl.col("close") * 0.03).alias("fi_or_yoy"),
    )


def calculate_specs(source: pl.DataFrame, specs: list[CommonFactorSpec]) -> pl.DataFrame:
    """Calculate a list of factor specs into a single frame."""
    factor_df = source.select("datetime", "vt_symbol")

    for spec in specs:
        values = calculate_by_expression(source, spec.expression).rename({"data": spec.name})
        factor_df = factor_df.join(values, on=["datetime", "vt_symbol"], how="left")

    return factor_df


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

    ema_bias = calculate_by_expression(bars, "close / (ta_ema(close, 3) + 1e-12) - 1")
    ema_bias_last = dict(
        zip(
            ema_bias.filter(pl.col("datetime") == last_datetime)["vt_symbol"].to_list(),
            ema_bias.filter(pl.col("datetime") == last_datetime)["data"].to_list(),
            strict=True,
        )
    )

    assert rsi_last["UP.TEST"] > 99.0
    assert rsi_last["DOWN.TEST"] < 1.0
    assert macd_last["UP.TEST"] > 0
    assert macd_last["DOWN.TEST"] < 0
    assert ema_bias_last["UP.TEST"] > 0
    assert ema_bias_last["DOWN.TEST"] < 0


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


def test_common_candlestick_factor_specs_are_named_and_unique() -> None:
    """Test candlestick factor helpers expose stable names."""
    specs = common_candlestick_factor_specs()
    names = common_candlestick_factor_names()
    expressions = common_candlestick_expressions()

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "common_open_gap" in names
    assert "common_close_position" in names
    assert "common_lower_shadow_ratio" in names


def test_common_price_volume_accumulation_specs_are_named_and_unique() -> None:
    """Test price-volume accumulation factor helpers expose stable names."""
    specs = common_price_volume_accumulation_factor_specs(windows=(3,))
    names = common_price_volume_accumulation_factor_names(windows=(3,))
    expressions = common_price_volume_accumulation_expressions(windows=(3,))

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "common_obv_pressure_3" in names
    assert "common_cmf_3" in names
    assert "common_money_flow_pressure_3" in names


def test_common_liquidity_crowding_specs_are_named_and_unique() -> None:
    """Test liquidity/crowding factor helpers expose stable names and optional columns."""
    specs = common_liquidity_crowding_factor_specs(windows=(3,))
    names = common_liquidity_crowding_factor_names(windows=(3,))
    expressions = common_liquidity_crowding_expressions(windows=(3,))
    turnover_rate_spec = next(spec for spec in specs if spec.name == "common_turnover_rate_zscore_3")

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "common_amihud_illiquidity_3" in names
    assert "common_liquidity_score_3" in names
    assert turnover_rate_spec.required_columns == ("turnover_rate",)


def test_common_position_breakout_specs_are_named_and_unique() -> None:
    """Test position/breakout factor helpers expose stable names."""
    specs = common_position_breakout_factor_specs(windows=(3,))
    names = common_position_breakout_factor_names(windows=(3,))
    expressions = common_position_breakout_expressions(windows=(3,))

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "common_days_since_high_3" in names
    assert "common_prior_high_breakout_3" in names
    assert "common_low_reclaim_strength_3" in names


def test_common_oscillator_specs_are_named_and_unique() -> None:
    """Test oscillator factor helpers expose stable names."""
    specs = common_oscillator_factor_specs(windows=(3,))
    names = common_oscillator_factor_names(windows=(3,))
    expressions = common_oscillator_expressions(windows=(3,))

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "common_stoch_k_3" in names
    assert "common_cci_3" in names
    assert "common_adx_proxy_3" in names


def test_common_relative_strength_specs_are_named_and_unique() -> None:
    """Test optional relative-strength helpers expose stable names and required columns."""
    specs = common_relative_strength_factor_specs(windows=(3,))
    names = common_relative_strength_factor_names(windows=(3,))
    expressions = common_relative_strength_expressions(windows=(3,))
    benchmark_spec = next(spec for spec in specs if spec.name == "common_relative_benchmark_strength_3")
    industry_spec = next(spec for spec in specs if spec.name == "common_relative_industry_strength_3")

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert benchmark_spec.required_columns == ("benchmark_close",)
    assert industry_spec.required_columns == ("industry_close",)


def test_common_a_share_limit_state_specs_are_named_and_unique() -> None:
    """Test optional A-share limit-state helpers expose stable names and required columns."""
    specs = common_a_share_limit_state_factor_specs(windows=(3,))
    names = common_a_share_limit_state_factor_names(windows=(3,))
    expressions = common_a_share_limit_state_expressions(windows=(3,))
    limit_spec = next(spec for spec in specs if spec.name == "common_is_limit_up")

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "common_limit_up_count_3" in names
    assert "common_anti_limit_up_heat_3" in names
    assert "up_limit" in limit_spec.required_columns
    assert "raw_close" in limit_spec.required_columns


def test_common_fundamental_specs_are_named_and_unique() -> None:
    """Test optional fundamental helpers expose stable names and required columns."""
    specs = common_fundamental_factor_specs(windows=(3,))
    names = common_fundamental_factor_names(windows=(3,))
    expressions = common_fundamental_expressions(windows=(3,))
    value_spec = next(spec for spec in specs if spec.name == "common_value_pb")
    blend_spec = next(spec for spec in specs if spec.name == "common_quality_value_blend_3")

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert value_spec.required_columns == ("pb",)
    assert "fi_roe" in blend_spec.required_columns
    assert "common_growth_netprofit_yoy" in names


def test_common_new_factor_families_calculate() -> None:
    """Test the newly added common factor families run through the alpha DSL."""
    panel = make_enriched_common_panel()
    specs = [
        *common_candlestick_factor_specs(),
        *common_price_volume_accumulation_factor_specs(windows=(3,)),
        *common_liquidity_crowding_factor_specs(windows=(3,)),
        *common_position_breakout_factor_specs(windows=(3,)),
        *common_oscillator_factor_specs(windows=(3,)),
        *common_relative_strength_factor_specs(windows=(3,)),
        *common_a_share_limit_state_factor_specs(windows=(3,)),
        *common_fundamental_factor_specs(windows=(3,)),
    ]

    factor_df = calculate_specs(panel, specs)

    assert factor_df.height == panel.height
    for column in (
        "common_close_position",
        "common_obv_pressure_3",
        "common_amihud_illiquidity_3",
        "common_prior_high_breakout_3",
        "common_cci_3",
        "common_relative_benchmark_strength_3",
        "common_is_limit_up",
        "common_quality_value_blend_3",
    ):
        assert factor_df.get_column(column).drop_nulls().len() > 0

    max_limit_up = factor_df.get_column("common_is_limit_up").drop_nulls().max()
    max_stoch = factor_df.get_column("common_stoch_k_3").drop_nulls().max()
    assert isinstance(max_limit_up, Real)
    assert isinstance(max_stoch, Real)
    assert max_limit_up <= 1
    assert max_stoch <= 1


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


def test_common_gmma_factor_specs_are_named_and_unique() -> None:
    """Test GMMA factor helpers expose stable names."""
    specs = common_gmma_factor_specs(short_periods=(3, 5), long_periods=(8, 10), slope_window=2, min_gap=0.001)
    names = common_gmma_factor_names(short_periods=(3, 5), long_periods=(8, 10), slope_window=2, min_gap=0.001)
    expressions = common_gmma_expressions(short_periods=(3, 5), long_periods=(8, 10), slope_window=2, min_gap=0.001)

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "common_gmma_group_gap_s3_5_l8_10" in names
    assert "common_gmma_anti_group_gap_s3_5_l8_10" in names
    assert "common_gmma_pullback_to_short_group_s3_5_l8_10" in names
    assert "common_gmma_bull_trend_s3_5_l8_10_2" in names


def test_common_vegas_factor_specs_are_named_and_unique() -> None:
    """Test Vegas factor helpers expose stable names."""
    specs = common_vegas_factor_specs(
        ema_filter_period=3,
        fast_channel=(5, 6),
        slow_channel=(10, 12),
        slope_window=2,
        ema_slope_window=1,
        prior_high_window=5,
    )
    names = common_vegas_factor_names(
        ema_filter_period=3,
        fast_channel=(5, 6),
        slow_channel=(10, 12),
        slope_window=2,
        ema_slope_window=1,
        prior_high_window=5,
    )
    expressions = common_vegas_expressions(
        ema_filter_period=3,
        fast_channel=(5, 6),
        slow_channel=(10, 12),
        slope_window=2,
        ema_slope_window=1,
        prior_high_window=5,
    )

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "common_vegas_fast_position_f5_6_s10_12" in names
    assert "common_vegas_anti_breakout_extension_f5_6_s10_12" in names
    assert "common_vegas_fast_top_reclaim_strength_f5_6_s10_12" in names
    assert "common_vegas_long_ready_3_f5_6_s10_12" in names


def test_common_volatility_compression_specs_are_named_and_unique() -> None:
    """Test volatility-compression factor helpers expose stable names."""
    specs = common_volatility_compression_factor_specs(windows=(3, 5))
    names = common_volatility_compression_factor_names(windows=(3, 5))
    expressions = common_volatility_compression_expressions(windows=(3, 5))

    assert names == [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "common_bollinger_compression_3" in names
    assert "common_squeeze_compression_3" in names
    assert "common_range_compression_3_5" in names
    assert "common_volume_contraction_3_5" in names


def test_common_gmma_and_vegas_expressions_calculate() -> None:
    """Test GMMA and Vegas expressions run through the existing alpha DSL."""
    bars = make_directional_bars()
    specs = [
        *common_gmma_factor_specs(short_periods=(3, 5), long_periods=(8, 10), slope_window=2, min_gap=0.001),
        *common_vegas_factor_specs(
            ema_filter_period=3,
            fast_channel=(5, 6),
            slow_channel=(10, 12),
            slope_window=2,
            ema_slope_window=1,
            prior_high_window=5,
        ),
        *common_volatility_compression_factor_specs(windows=(3, 5)),
    ]
    factor_df = bars.select("datetime", "vt_symbol")

    for spec in specs:
        values = calculate_by_expression(bars, spec.expression).rename({"data": spec.name})
        factor_df = factor_df.join(values, on=["datetime", "vt_symbol"], how="left")

    assert factor_df.height == bars.height
    assert factor_df.get_column("common_gmma_group_gap_s3_5_l8_10").drop_nulls().len() > 0
    assert factor_df.get_column("common_vegas_fast_position_f5_6_s10_12").drop_nulls().len() > 0
    assert factor_df.get_column("common_bollinger_compression_3").drop_nulls().len() > 0
    max_gmma_bull = factor_df.get_column("common_gmma_bull_trend_s3_5_l8_10_2").drop_nulls().max()
    max_vegas_long = factor_df.get_column("common_vegas_long_ready_3_f5_6_s10_12").drop_nulls().max()
    max_squeeze_ratio = factor_df.get_column("common_squeeze_ratio_3").drop_nulls().max()
    assert isinstance(max_gmma_bull, Real)
    assert isinstance(max_vegas_long, Real)
    assert isinstance(max_squeeze_ratio, Real)
    assert max_gmma_bull <= 1
    assert max_vegas_long <= 1
    assert max_squeeze_ratio >= 0
