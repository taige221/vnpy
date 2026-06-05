from __future__ import annotations

from datetime import datetime, timedelta
from typing import cast

import polars as pl

from vnpy.alpha.factors import (
    double_bottom_expressions,
    double_bottom_factor_names,
    double_bottom_factor_specs,
)
from vnpy.alpha.utils.doubleBottom_extract import extract_double_bottom_factor_frame


def make_double_bottom_bars() -> pl.DataFrame:
    """Create deterministic bars with a W-bottom-like path."""
    rows: list[dict[str, float | str | datetime]] = []
    start = datetime(2024, 1, 1)
    close_path = [
        10.0,
        9.5,
        9.0,
        9.4,
        10.0,
        10.4,
        9.8,
        9.1,
        9.3,
        9.8,
        10.8,
        10.5,
        10.9,
        11.1,
    ]

    for symbol_index, vt_symbol in enumerate(["W001.TEST", "W002.TEST"]):
        scale = 1.0 + symbol_index * 0.02
        for day_index, close in enumerate(close_path):
            current_close = close * scale
            rows.append(
                {
                    "datetime": start + timedelta(days=day_index),
                    "vt_symbol": vt_symbol,
                    "open": current_close * (0.985 if day_index in {8, 10} else 0.995),
                    "high": current_close * 1.012,
                    "low": current_close * (0.965 if day_index in {2, 7} else 0.985),
                    "close": current_close,
                    "volume": 1_000_000.0 + day_index * 60_000.0 + symbol_index * 25_000.0,
                    "turnover": current_close * 1_000_000.0,
                    "open_interest": 0.0,
                    "vwap": current_close,
                }
            )

    return pl.DataFrame(rows).sort(["datetime", "vt_symbol"])


def max_float(series: pl.Series) -> float:
    """Return a non-null series max as float for typed assertions."""
    value = series.drop_nulls().max()
    assert value is not None
    return float(cast(float, value))


def test_double_bottom_factor_specs_are_stable() -> None:
    """Test double-bottom factor definitions are named and filterable."""
    specs = double_bottom_factor_specs(windows=(6,), volume_windows=(3,), confirm_windows=(2,))
    names = [spec.name for spec in specs]

    assert names
    assert len(names) == len(set(names))
    assert any(spec.signal_type == "second_low_confirm" for spec in specs)
    assert any(spec.signal_type == "neckline_breakout" for spec in specs)
    assert any(spec.signal_type == "neckline_retest" for spec in specs)
    assert "double_bottom_second_low_quality_6_2" in names
    assert "double_bottom_neckline_breakout_quality_6" in names


def test_double_bottom_signal_type_filters_include_shared_factors() -> None:
    """Test expression/name helpers filter by signal family."""
    second_low_names = double_bottom_factor_names(
        windows=(6,),
        volume_windows=(3,),
        confirm_windows=(2,),
        signal_type="second_low_confirm",
    )
    breakout_names = double_bottom_factor_names(
        windows=(6,),
        volume_windows=(3,),
        confirm_windows=(2,),
        signal_type="neckline_breakout",
    )
    second_low_exprs = double_bottom_expressions(
        windows=(6,),
        volume_windows=(3,),
        confirm_windows=(2,),
        signal_type="second_low_confirm",
    )

    assert "double_bottom_volume_ratio_3" in second_low_names
    assert "double_bottom_second_low_quality_6_2" in second_low_names
    assert "double_bottom_neckline_breakout_quality_6" not in second_low_names
    assert "double_bottom_neckline_breakout_quality_6" in breakout_names
    assert len(second_low_exprs) == len(second_low_names)


def test_extract_double_bottom_factor_frame_calculates_named_columns() -> None:
    """Test double-bottom expressions run through the existing alpha DSL."""
    specs = double_bottom_factor_specs(windows=(6,), volume_windows=(3,), confirm_windows=(2,))
    factor_df = extract_double_bottom_factor_frame(make_double_bottom_bars(), specs=specs)

    assert factor_df.height == 28
    for spec in specs:
        assert spec.name in factor_df.columns

    assert max_float(factor_df.get_column("double_bottom_neckline_height_6")) > 0
    assert max_float(factor_df.get_column("double_bottom_low_similarity_6")) > -0.05
    assert max_float(factor_df.get_column("double_bottom_neckline_breakout_strength_6")) > 0
