from __future__ import annotations

from datetime import datetime, timedelta
from numbers import Real

import polars as pl

from vnpy.alpha.factors import (
    strategy_box_expressions,
    strategy_box_factor_names,
    strategy_box_factor_specs,
    strategy_box_trade_rule_factor_specs,
)
from vnpy.alpha.utils.strategyByBox_extract import extract_strategy_by_box_factor_frame


def make_box_bars() -> pl.DataFrame:
    """Create deterministic bars with one resistance breakout."""
    rows: list[dict[str, float | str | datetime]] = []
    start = datetime(2024, 1, 1)
    close_path = [9.8, 10.0, 10.1, 9.9, 10.0, 10.6, 10.7, 10.4, 10.2, 10.5, 10.8, 10.9]

    for symbol_index, vt_symbol in enumerate(["S001.TEST", "S002.TEST"]):
        scale = 1.0 + symbol_index * 0.03
        for day_index, close in enumerate(close_path):
            current_close = close * scale
            rows.append(
                {
                    "datetime": start + timedelta(days=day_index),
                    "vt_symbol": vt_symbol,
                    "open": current_close * (0.985 if day_index == 5 else 0.995),
                    "high": current_close * 1.01,
                    "low": current_close * 0.98,
                    "close": current_close,
                    "volume": 1_000_000.0 + day_index * 40_000.0 + symbol_index * 20_000.0,
                    "turnover": current_close * 1_000_000.0,
                    "open_interest": 0.0,
                    "vwap": current_close,
                }
            )

    return pl.DataFrame(rows).sort(["datetime", "vt_symbol"])


def test_strategy_box_factor_specs_are_stable() -> None:
    """Test strategyByBox factor definitions are named and filterable."""
    specs = strategy_box_factor_specs(windows=(5,), volume_windows=(3,))
    names = [spec.name for spec in specs]

    assert names
    assert len(names) == len(set(names))
    assert any(spec.signal_type == "breakout_long" for spec in specs)
    assert any(spec.signal_type == "pullback_bounce" for spec in specs)
    assert any(spec.signal_type == "box_reclaim" for spec in specs)
    assert any(spec.signal_type == "breakout_retest" for spec in specs)
    assert any(spec.signal_type == "range_reclaim2" for spec in specs)
    assert "box_breakout_quality_5" in names
    assert "box_pullback_quality_5" in names
    assert "box_reclaim_quality_5" in names
    assert "box_breakout_retest_quality_5_5" in names
    assert "box_range_reclaim2_quality_5" in names
    assert "box_stack_lift_reversal_5" in names


def test_strategy_box_signal_type_filters_include_shared_factors() -> None:
    """Test expression/name helpers filter by signal family."""
    breakout_names = strategy_box_factor_names(windows=(5,), volume_windows=(3,), signal_type="breakout_long")
    pullback_names = strategy_box_factor_names(windows=(5,), volume_windows=(3,), signal_type="pullback_bounce")
    reclaim_names = strategy_box_factor_names(windows=(5,), volume_windows=(3,), signal_type="box_reclaim")
    retest_names = strategy_box_factor_names(windows=(5,), volume_windows=(3,), signal_type="breakout_retest")
    reclaim2_names = strategy_box_factor_names(windows=(5,), volume_windows=(3,), signal_type="range_reclaim2")
    breakout_exprs = strategy_box_expressions(windows=(5,), volume_windows=(3,), signal_type="breakout_long")

    assert "box_volume_ratio_3" in breakout_names
    assert "box_breakout_strength_5" in breakout_names
    assert "box_pullback_quality_5" not in breakout_names
    assert "box_pullback_quality_5" in pullback_names
    assert "box_reclaim_quality_5" in pullback_names
    assert "box_breakout_retest_quality_5_5" in pullback_names
    assert "box_range_reclaim2_quality_5" in pullback_names
    assert "box_reclaim_quality_5" in reclaim_names
    assert "box_breakout_retest_quality_5_5" not in reclaim_names
    assert "box_breakout_retest_quality_5_5" in retest_names
    assert "box_reclaim_quality_5" not in retest_names
    assert "box_range_reclaim2_quality_5" in reclaim2_names
    assert "box_reclaim_quality_5" not in reclaim2_names
    assert len(breakout_exprs) == len(breakout_names)


def test_extract_strategy_box_factor_frame_calculates_named_columns() -> None:
    """Test strategyByBox expressions run through the existing alpha DSL."""
    specs = strategy_box_factor_specs(windows=(5,), volume_windows=(3,))
    factor_df = extract_strategy_by_box_factor_frame(make_box_bars(), specs=specs)

    assert factor_df.height == 24
    for spec in specs:
        assert spec.name in factor_df.columns

    max_breakout_strength = factor_df.get_column("box_breakout_strength_5").drop_nulls().max()
    max_retest_count = factor_df.get_column("box_breakout_retest_prior_count_5_5").drop_nulls().max()
    max_reclaim2_close_above = factor_df.get_column("box_range_reclaim2_close_above_resistance_pct_5").drop_nulls().max()
    assert isinstance(max_breakout_strength, Real)
    assert isinstance(max_retest_count, Real)
    assert isinstance(max_reclaim2_close_above, Real)
    assert float(max_breakout_strength) > 0
    assert float(max_retest_count) > 0
    assert float(max_reclaim2_close_above) > 0


def test_extract_strategy_box_pullback_filter_matches_name_helper() -> None:
    """Test pullback extraction includes all pullback subfamilies."""
    expected_names = strategy_box_factor_names(windows=(5,), volume_windows=(3,), signal_type="pullback_bounce")
    factor_df = extract_strategy_by_box_factor_frame(
        make_box_bars(),
        windows=(5,),
        volume_windows=(3,),
        signal_type="pullback_bounce",
    )

    missing = [name for name in expected_names if name not in factor_df.columns]
    assert not missing
    assert "box_range_reclaim2_quality_5" in factor_df.columns


def test_box_stack_lift_reversal_is_negative_stack_lift() -> None:
    """Test stack-lift reversal keeps the inverse factor direction explicit."""
    specs = [
        spec
        for spec in strategy_box_factor_specs(windows=(5,), volume_windows=(3,))
        if spec.name in {"box_stack_lift_5", "box_stack_lift_reversal_5"}
    ]
    factor_df = extract_strategy_by_box_factor_frame(make_box_bars(), specs=specs)

    check_df = (
        factor_df.select("box_stack_lift_5", "box_stack_lift_reversal_5")
        .drop_nulls()
        .with_columns((pl.col("box_stack_lift_5") + pl.col("box_stack_lift_reversal_5")).abs().alias("diff"))
    )

    max_diff = check_df.get_column("diff").max()
    assert isinstance(max_diff, Real)
    assert float(max_diff) < 1e-12


def test_strategy_box_trade_rule_factors_calculate_from_enriched_panel() -> None:
    """Test optional trade-rule factors can use enriched ranking/regime columns."""
    bars = make_box_bars().with_columns(
        pl.lit(60.0).alias("rank_score"),
        pl.lit(50.0).alias("heat_score"),
        pl.lit(4.0).alias("turnover_rate"),
        pl.lit("bull_active").alias("market_regime"),
    )
    specs = strategy_box_trade_rule_factor_specs(windows=(5,))
    factor_df = extract_strategy_by_box_factor_frame(bars, specs=specs)

    assert "box_rule_regime_bull_active" in factor_df.columns
    assert "box_rule_tradable_pullback_mask_5" in factor_df.columns
    assert factor_df.get_column("box_rule_rank_pass").drop_nulls().max() == 1
    assert factor_df.get_column("box_rule_heat_pass").drop_nulls().max() == 1
    assert factor_df.get_column("box_rule_turnover_rate_band").drop_nulls().max() == 1


def test_strategy_box_trade_rule_factors_report_missing_columns() -> None:
    """Test optional trade-rule factors fail clearly when enriched fields are absent."""
    specs = strategy_box_trade_rule_factor_specs(windows=(5,))

    try:
        extract_strategy_by_box_factor_frame(make_box_bars(), specs=specs)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing column error")

    assert "rank_score" in message
    assert "turnover_rate" in message
