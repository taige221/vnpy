from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from vnpy.alpha.dataset.utility import calculate_by_expression
from vnpy.alpha.factors.doubleBottom import (
    DEFAULT_DOUBLE_BOTTOM_CONFIRM_WINDOWS,
    DEFAULT_DOUBLE_BOTTOM_VOLUME_WINDOWS,
    DEFAULT_DOUBLE_BOTTOM_WINDOWS,
    DoubleBottomFactorSpec,
    double_bottom_factor_specs,
)
from vnpy.alpha.lab import AlphaLab


@dataclass(frozen=True)
class DoubleBottomExtractConfig:
    """Configuration for extracting double-bottom factors from AlphaLab bars."""

    lab_path: str | Path
    component: str
    start_date: str
    end_date: str
    output: str | Path
    interval: str = "d"
    extended_days: int = 180
    windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_WINDOWS
    volume_windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_VOLUME_WINDOWS
    confirm_windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_CONFIRM_WINDOWS
    signal_type: str | None = None
    symbol_limit: int | None = None


def extract_double_bottom_factor_frame(
    bar_df: pl.DataFrame,
    specs: list[DoubleBottomFactorSpec] | None = None,
    windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_WINDOWS,
    volume_windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_VOLUME_WINDOWS,
    confirm_windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_CONFIRM_WINDOWS,
    signal_type: str | None = None,
) -> pl.DataFrame:
    """Calculate named double-bottom factor columns for a bar panel."""
    if bar_df.is_empty():
        raise ValueError("bar_df is empty")

    factor_specs = specs or double_bottom_factor_specs(
        windows=windows,
        volume_windows=volume_windows,
        confirm_windows=confirm_windows,
    )
    if signal_type:
        factor_specs = [spec for spec in factor_specs if spec.signal_type in {signal_type, "shared"}]

    result = (
        bar_df.select("datetime", "vt_symbol")
        .unique(subset=["datetime", "vt_symbol"])
        .sort(["datetime", "vt_symbol"])
    )
    source = bar_df.sort(["datetime", "vt_symbol"])

    for spec in factor_specs:
        factor_df = calculate_by_expression(source, spec.expression).rename({"data": spec.name})
        result = result.join(factor_df, on=["datetime", "vt_symbol"], how="left")

    return result.sort(["datetime", "vt_symbol"])


def load_double_bottom_bars(config: DoubleBottomExtractConfig) -> pl.DataFrame:
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


def extract_double_bottom_factors(config: DoubleBottomExtractConfig) -> pl.DataFrame:
    """Load bars, calculate factors, and save the factor panel."""
    bar_df = load_double_bottom_bars(config)
    factor_df = extract_double_bottom_factor_frame(
        bar_df,
        windows=config.windows,
        volume_windows=config.volume_windows,
        confirm_windows=config.confirm_windows,
        signal_type=config.signal_type,
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract double-bottom factors from AlphaLab bars")
    parser.add_argument("--lab-path", required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", default="d")
    parser.add_argument("--extended-days", type=int, default=180)
    parser.add_argument("--windows", type=_parse_int_tuple, default=DEFAULT_DOUBLE_BOTTOM_WINDOWS)
    parser.add_argument("--volume-windows", type=_parse_int_tuple, default=DEFAULT_DOUBLE_BOTTOM_VOLUME_WINDOWS)
    parser.add_argument("--confirm-windows", type=_parse_int_tuple, default=DEFAULT_DOUBLE_BOTTOM_CONFIRM_WINDOWS)
    parser.add_argument("--signal-type", choices=["second_low_confirm", "neckline_breakout", "neckline_retest", "shared"])
    parser.add_argument("--symbol-limit", type=int)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = DoubleBottomExtractConfig(
        lab_path=args.lab_path,
        component=args.component,
        start_date=args.start_date,
        end_date=args.end_date,
        output=args.output,
        interval=args.interval,
        extended_days=args.extended_days,
        windows=args.windows,
        volume_windows=args.volume_windows,
        confirm_windows=args.confirm_windows,
        signal_type=args.signal_type,
        symbol_limit=args.symbol_limit,
    )
    factor_df = extract_double_bottom_factors(config)
    print(f"Saved {factor_df.height} rows x {len(factor_df.columns)} columns to {config.output}")


if __name__ == "__main__":
    main()


__all__ = [
    "DoubleBottomExtractConfig",
    "extract_double_bottom_factor_frame",
    "extract_double_bottom_factors",
    "load_double_bottom_bars",
]
