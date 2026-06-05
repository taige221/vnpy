from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from vnpy.alpha.dataset.utility import calculate_by_expression
from vnpy.alpha.factors.strategyByBox import (
    DEFAULT_BREAKOUT_RETEST_WINDOWS,
    DEFAULT_BOX_WINDOWS,
    DEFAULT_VOLUME_WINDOWS,
    PULLBACK_SIGNAL_TYPES,
    StrategyBoxFactorSpec,
    strategy_box_factor_specs,
)
from vnpy.alpha.lab import AlphaLab


@dataclass(frozen=True)
class StrategyByBoxExtractConfig:
    """Configuration for extracting box-strategy factors from AlphaLab bars."""

    lab_path: str | Path
    component: str
    start_date: str
    end_date: str
    output: str | Path
    interval: str = "d"
    extended_days: int = 120
    windows: tuple[int, ...] = DEFAULT_BOX_WINDOWS
    volume_windows: tuple[int, ...] = DEFAULT_VOLUME_WINDOWS
    retest_windows: tuple[int, ...] = DEFAULT_BREAKOUT_RETEST_WINDOWS
    signal_type: str | None = None
    include_trade_rule_factors: bool = False
    symbol_limit: int | None = None


def extract_strategy_by_box_factor_frame(
    bar_df: pl.DataFrame,
    specs: list[StrategyBoxFactorSpec] | None = None,
    windows: tuple[int, ...] = DEFAULT_BOX_WINDOWS,
    volume_windows: tuple[int, ...] = DEFAULT_VOLUME_WINDOWS,
    retest_windows: tuple[int, ...] = DEFAULT_BREAKOUT_RETEST_WINDOWS,
    signal_type: str | None = None,
    include_trade_rule_factors: bool = False,
) -> pl.DataFrame:
    """Calculate named box-strategy factor columns for a bar panel."""
    if bar_df.is_empty():
        raise ValueError("bar_df is empty")

    factor_specs = specs or strategy_box_factor_specs(
        windows=windows,
        volume_windows=volume_windows,
        retest_windows=retest_windows,
        include_trade_rule_factors=include_trade_rule_factors,
    )
    if signal_type:
        factor_specs = [
            spec
            for spec in factor_specs
            if spec.signal_type in {signal_type, "shared"}
            or (signal_type == "pullback_bounce" and spec.signal_type in PULLBACK_SIGNAL_TYPES)
        ]

    result = (
        bar_df.select("datetime", "vt_symbol")
        .unique(subset=["datetime", "vt_symbol"])
        .sort(["datetime", "vt_symbol"])
    )
    source = _prepare_strategy_by_box_trade_rule_source(bar_df).sort(["datetime", "vt_symbol"])
    _validate_required_columns(source, factor_specs)

    for spec in factor_specs:
        factor_df = calculate_by_expression(source, spec.expression).rename({"data": spec.name})
        result = result.join(factor_df, on=["datetime", "vt_symbol"], how="left")

    return result.sort(["datetime", "vt_symbol"])


def load_strategy_by_box_bars(config: StrategyByBoxExtractConfig) -> pl.DataFrame:
    """Load component bars from AlphaLab for factor extraction."""
    lab = AlphaLab(str(config.lab_path))
    symbols = sorted(lab.load_component_symbols(config.component, config.start_date, config.end_date))
    if config.symbol_limit is not None:
        symbols = symbols[: config.symbol_limit]
    if not symbols:
        raise RuntimeError(f"No symbols found for component {config.component!r}")

    bar_df = lab.load_bar_df(
        symbols,
        config.interval,
        config.start_date,
        config.end_date,
        config.extended_days,
    )
    if bar_df is None or bar_df.is_empty():
        raise RuntimeError("No bars loaded from AlphaLab")
    return bar_df


def extract_strategy_by_box_factors(config: StrategyByBoxExtractConfig) -> pl.DataFrame:
    """Load bars, calculate factors, and save the factor panel."""
    bar_df = load_strategy_by_box_bars(config)
    factor_df = extract_strategy_by_box_factor_frame(
        bar_df,
        windows=config.windows,
        volume_windows=config.volume_windows,
        retest_windows=config.retest_windows,
        signal_type=config.signal_type,
        include_trade_rule_factors=config.include_trade_rule_factors,
    )
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        factor_df.write_csv(output_path)
    else:
        factor_df.write_parquet(output_path)
    return factor_df


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not items:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return items


def _prepare_strategy_by_box_trade_rule_source(df: pl.DataFrame) -> pl.DataFrame:
    """Add numeric helper columns used by optional trade-rule factor expressions."""
    result = df

    if "rank_score" not in result.columns and "rank" in result.columns:
        result = result.with_columns(pl.col("rank").cast(pl.Float64).alias("rank_score"))

    if "heat_score" not in result.columns and "rank_score" in result.columns:
        result = result.with_columns(pl.col("rank_score").cast(pl.Float64).alias("heat_score"))

    if "market_regime" in result.columns:
        regime_exprs: list[pl.Expr] = []
        for regime in ("bull_active", "range_neutral", "weak_defensive", "bear_pause", "unknown"):
            column = f"regime_{regime}"
            if column not in result.columns:
                regime_exprs.append((pl.col("market_regime").cast(pl.Utf8) == regime).cast(pl.Int32).alias(column))
        if regime_exprs:
            result = result.with_columns(regime_exprs)

    return result


def _validate_required_columns(df: pl.DataFrame, specs: list[StrategyBoxFactorSpec]) -> None:
    """Raise a readable error when optional factor specs need missing source columns."""
    missing = sorted({column for spec in specs for column in spec.required_columns if column not in df.columns})
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing columns required by selected strategyByBox factors: {joined}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract strategyByBox factors from AlphaLab bars")
    parser.add_argument("--lab-path", required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", default="d")
    parser.add_argument("--extended-days", type=int, default=120)
    parser.add_argument("--windows", type=_parse_int_tuple, default=DEFAULT_BOX_WINDOWS)
    parser.add_argument("--volume-windows", type=_parse_int_tuple, default=DEFAULT_VOLUME_WINDOWS)
    parser.add_argument("--retest-windows", type=_parse_int_tuple, default=DEFAULT_BREAKOUT_RETEST_WINDOWS)
    parser.add_argument(
        "--signal-type",
        choices=["breakout_long", "pullback_bounce", "box_reclaim", "breakout_retest", "range_reclaim2", "trade_rule", "shared"],
    )
    parser.add_argument("--include-trade-rule-factors", action="store_true")
    parser.add_argument("--symbol-limit", type=int)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = StrategyByBoxExtractConfig(
        lab_path=args.lab_path,
        component=args.component,
        start_date=args.start_date,
        end_date=args.end_date,
        output=args.output,
        interval=args.interval,
        extended_days=args.extended_days,
        windows=args.windows,
        volume_windows=args.volume_windows,
        retest_windows=args.retest_windows,
        signal_type=args.signal_type,
        include_trade_rule_factors=args.include_trade_rule_factors,
        symbol_limit=args.symbol_limit,
    )
    factor_df = extract_strategy_by_box_factors(config)
    print(f"Saved {factor_df.height} rows x {len(factor_df.columns)} columns to {config.output}")


if __name__ == "__main__":
    main()


__all__ = [
    "StrategyByBoxExtractConfig",
    "extract_strategy_by_box_factor_frame",
    "extract_strategy_by_box_factors",
    "load_strategy_by_box_bars",
]
