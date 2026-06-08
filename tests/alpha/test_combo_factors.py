from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from vnpy.alpha.dataset.utility import calculate_by_expression
from vnpy.alpha.factors import (
    COMBO_CANDIDATE_SET,
    ComboFactorSpec,
    candidate_combo_factor_specs,
    combo_expressions,
    combo_factor_names,
    combo_factor_specs,
)


def make_combo_bars(days: int = 180) -> pl.DataFrame:
    """Create an enriched multi-symbol panel for combo factor tests."""
    start = datetime(2024, 1, 1)
    rows: list[dict[str, datetime | float | str]] = []

    for index in range(days):
        current = start + timedelta(days=index)
        benchmark_close = 100.0 + index * 0.05

        for symbol_index, (symbol, base, drift) in enumerate(
            (
                ("UP.TEST", 10.0, 0.06),
                ("FLAT.TEST", 16.0, 0.01),
                ("DOWN.TEST", 22.0, -0.03),
            )
        ):
            wave = 0.4 if (index + symbol_index) % 11 < 5 else -0.2
            close = base + index * drift + wave
            volume = 1_000_000.0 + index * 2_000.0 + symbol_index * 50_000.0
            rows.append(
                {
                    "datetime": current,
                    "vt_symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.03,
                    "low": close * 0.97,
                    "close": close,
                    "volume": volume,
                    "turnover": close * volume,
                    "raw_close": close,
                    "up_limit": close * 1.1,
                    "benchmark_close": benchmark_close,
                    "open_interest": 0.0,
                }
            )

    return pl.DataFrame(rows).sort(["datetime", "vt_symbol"])


def test_combo_factor_specs_are_named_and_grouped() -> None:
    """Test combo factor helpers expose stable names."""
    specs = combo_factor_specs()
    candidate_specs = candidate_combo_factor_specs()
    names = combo_factor_names()
    expressions = combo_expressions()

    assert COMBO_CANDIDATE_SET == "a_share_research_2023_20260430"
    assert all(isinstance(spec, ComboFactorSpec) for spec in specs)
    assert names == [spec.name for spec in specs]
    assert expressions == [spec.expression for spec in specs]
    assert len(names) == len(set(names))
    assert len(expressions) == len(set(expressions))
    assert "combo_core4_control" in names
    assert "combo_core4_pvcorr60_15pct" in names
    assert "combo_broad_top5" in names
    assert "combo_pvcorr15_broad_top5_10pct" in names
    assert [spec.name for spec in candidate_specs] == combo_factor_names(include_components=False)
    assert "combo_core4_control" not in [spec.name for spec in candidate_specs]


def test_combo_factor_expressions_calculate() -> None:
    """Test combo factor expressions run through the existing alpha DSL."""
    bars = make_combo_bars()
    factor_df = bars.select("datetime", "vt_symbol")

    for spec in combo_factor_specs():
        values = calculate_by_expression(bars, spec.expression).rename({"data": spec.name})
        factor_df = factor_df.join(values, on=["datetime", "vt_symbol"], how="left")

    assert factor_df.height == bars.height
    assert factor_df.get_column("combo_core4_pvcorr60_15pct").drop_nulls().len() > 0
    assert factor_df.get_column("combo_pvcorr15_broad_antitrix14_10pct").drop_nulls().len() > 0
    assert factor_df.get_column("combo_pvcorr15_broad_top5_10pct").drop_nulls().len() > 0
