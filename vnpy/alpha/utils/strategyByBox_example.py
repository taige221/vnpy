from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from vnpy.alpha.factors.strategyByBox import (
    DEFAULT_BREAKOUT_RETEST_WINDOWS,
    DEFAULT_BOX_WINDOWS,
    DEFAULT_VOLUME_WINDOWS,
    strategy_box_expressions,
)
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.utils.stat_alpha_loop import DEFAULT_HORIZONS, StatAlphaLoop


@dataclass(frozen=True)
class StrategyByBoxExampleConfig:
    """Configuration for scoring strategyByBox factors with StatAlphaLoop."""

    lab_path: str | Path
    component: str
    start_date: str
    end_date: str
    train_period: tuple[str, str]
    valid_period: tuple[str, str]
    test_period: tuple[str, str]
    interval: str = "d"
    extended_days: int = 120
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    windows: tuple[int, ...] = DEFAULT_BOX_WINDOWS
    volume_windows: tuple[int, ...] = DEFAULT_VOLUME_WINDOWS
    retest_windows: tuple[int, ...] = DEFAULT_BREAKOUT_RETEST_WINDOWS
    signal_type: str | None = None
    include_trade_rule_factors: bool = False
    entry_lag: int = 1
    min_universe: int = 300
    quantile: float = 0.2
    symbol_limit: int | None = None
    top: int = 20
    output_history: str | Path | None = None


def load_strategy_by_box_example_bars(config: StrategyByBoxExampleConfig) -> pl.DataFrame:
    """Load AlphaLab bars for the example scoring run."""
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


def run_strategy_by_box_example(config: StrategyByBoxExampleConfig) -> pl.DataFrame:
    """Score strategyByBox expressions and return the history frame."""
    bar_df = load_strategy_by_box_example_bars(config)
    expressions = strategy_box_expressions(
        windows=config.windows,
        volume_windows=config.volume_windows,
        retest_windows=config.retest_windows,
        signal_type=config.signal_type,
        include_trade_rule_factors=config.include_trade_rule_factors,
    )
    loop = StatAlphaLoop(
        bar_df,
        train_period=config.train_period,
        valid_period=config.valid_period,
        test_period=config.test_period,
        horizons=config.horizons,
        entry_lag=config.entry_lag,
        min_universe=config.min_universe,
        quantile=config.quantile,
    )
    loop.score_batch(expressions, top=config.top)
    history = loop.history_frame()

    if config.output_history:
        output_path = Path(config.output_history)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        history.write_csv(output_path)

    return history


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not items:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return items


def _parse_period(value: str) -> tuple[str, str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != 2:
        raise argparse.ArgumentTypeError("period must be start,end")
    return items[0], items[1]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Score strategyByBox factors with StatAlphaLoop")
    parser.add_argument("--lab-path", required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--train-period", type=_parse_period, required=True, help="start,end")
    parser.add_argument("--valid-period", type=_parse_period, required=True, help="start,end")
    parser.add_argument("--test-period", type=_parse_period, required=True, help="start,end")
    parser.add_argument("--interval", default="d")
    parser.add_argument("--extended-days", type=int, default=120)
    parser.add_argument("--horizons", type=_parse_int_tuple, default=DEFAULT_HORIZONS)
    parser.add_argument("--windows", type=_parse_int_tuple, default=DEFAULT_BOX_WINDOWS)
    parser.add_argument("--volume-windows", type=_parse_int_tuple, default=DEFAULT_VOLUME_WINDOWS)
    parser.add_argument("--retest-windows", type=_parse_int_tuple, default=DEFAULT_BREAKOUT_RETEST_WINDOWS)
    parser.add_argument(
        "--signal-type",
        choices=["breakout_long", "pullback_bounce", "box_reclaim", "breakout_retest", "range_reclaim2", "trade_rule", "shared"],
    )
    parser.add_argument("--include-trade-rule-factors", action="store_true")
    parser.add_argument("--entry-lag", type=int, default=1)
    parser.add_argument("--min-universe", type=int, default=300)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--symbol-limit", type=int)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output-history")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = StrategyByBoxExampleConfig(
        lab_path=args.lab_path,
        component=args.component,
        start_date=args.start_date,
        end_date=args.end_date,
        train_period=args.train_period,
        valid_period=args.valid_period,
        test_period=args.test_period,
        interval=args.interval,
        extended_days=args.extended_days,
        horizons=args.horizons,
        windows=args.windows,
        volume_windows=args.volume_windows,
        retest_windows=args.retest_windows,
        signal_type=args.signal_type,
        include_trade_rule_factors=args.include_trade_rule_factors,
        entry_lag=args.entry_lag,
        min_universe=args.min_universe,
        quantile=args.quantile,
        symbol_limit=args.symbol_limit,
        top=args.top,
        output_history=args.output_history,
    )
    history = run_strategy_by_box_example(config)
    print(f"Scored {history.height} strategyByBox factor records")


if __name__ == "__main__":
    main()


__all__ = [
    "StrategyByBoxExampleConfig",
    "load_strategy_by_box_example_bars",
    "run_strategy_by_box_example",
]
