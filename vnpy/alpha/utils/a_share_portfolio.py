from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from vnpy.alpha.factors import (
    combo_price_flow3_base_exprs,
    combo_price_flow3_expr,
    combo_price_flow3_rank_exprs,
)


DEFAULT_A_SHARE_SEGMENTS: dict[str, tuple[date, date]] = {
    "train": (date(2020, 1, 2), date(2022, 12, 31)),
    "valid": (date(2023, 1, 1), date(2024, 12, 31)),
    "test": (date(2025, 1, 1), date(2026, 6, 4)),
}
ANNUAL_DAYS: float = 243.0


@dataclass(frozen=True)
class ASharePortfolioConfig:
    """Configuration for A-share panel portfolio research."""

    top_n: int = 200
    rebalance_days: int = 20
    entry_lag: int = 1
    fee_bps: float = 15.0
    component_column: str = "in_a_share_ge1000"
    min_turnover: float = 50_000_000
    min_circ_mv: float = 100_000
    exclude_new_listing_days: int = 120


@dataclass(frozen=True)
class ASharePortfolioBacktestGridResult:
    """Backtest grid outputs and load diagnostics."""

    daily: pl.DataFrame
    summary: pl.DataFrame
    yearly: pl.DataFrame
    row_count: int
    signal_eligible_count: int


@dataclass(frozen=True)
class ASharePortfolioExposureResult:
    """Exposure diagnostic outputs for one portfolio configuration."""

    daily: pl.DataFrame
    summary: pl.DataFrame
    industry_overweight: pl.DataFrame
    market_overweight: pl.DataFrame
    index_exposure: pl.DataFrame


def panel_glob(panel_path: str | Path) -> str:
    path = Path(panel_path)
    if path.is_dir():
        return str(path / "**" / "*.parquet")
    return str(path)


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _segment_bounds(segments: dict[str, tuple[date, date]]) -> tuple[date, date]:
    return min(start for start, _ in segments.values()), max(end for _, end in segments.values())


def _base_combo_price_flow3_scan(
    panel_path: str | Path,
    config: ASharePortfolioConfig,
    *,
    segments: dict[str, tuple[date, date]],
    columns: list[str],
) -> pl.LazyFrame:
    min_start, max_end = _segment_bounds(segments)

    return (
        pl.scan_parquet(panel_glob(panel_path))
        .select(columns)
        .filter((pl.col("trade_date") >= min_start) & (pl.col("trade_date") <= max_end))
        .filter(pl.col(config.component_column).fill_null(False))
        .sort(["vt_symbol", "trade_date"])
        .with_columns(
            pl.col("is_limit_up").shift(-config.entry_lag).over("vt_symbol").fill_null(False).alias("entry_limit_up"),
            pl.col("is_limit_down").shift(-config.entry_lag).over("vt_symbol").fill_null(False).alias("entry_limit_down"),
            *combo_price_flow3_base_exprs(),
        )
        .with_columns(
            (
                pl.col("has_valid_bar").fill_null(False)
                & pl.col("is_list_life_valid").fill_null(False)
                & pl.col("list_date").is_not_null()
                & pl.col("close").is_not_null()
                & (pl.col("close") > 0)
                & (pl.col("trade_date") >= pl.col("list_date").dt.offset_by(f"{config.exclude_new_listing_days}d"))
                & ~pl.col("name").fill_null("").str.contains("ST|退")
                & ~(pl.col("entry_limit_up") | pl.col("entry_limit_down"))
                & (pl.col("turnover").fill_null(0) >= config.min_turnover)
                & (pl.col("circ_mv").fill_null(0) >= config.min_circ_mv)
            ).alias("signal_eligible"),
            (
                pl.col("has_valid_bar").fill_null(False)
                & pl.col("is_list_life_valid").fill_null(False)
                & pl.col("close").is_not_null()
                & (pl.col("close") > 0)
                & ~pl.col("name").fill_null("").str.contains("ST|退")
                & (pl.col("turnover").fill_null(0) >= config.min_turnover)
                & (pl.col("circ_mv").fill_null(0) >= config.min_circ_mv)
            ).alias("benchmark_eligible"),
        )
        .with_columns(combo_price_flow3_rank_exprs())
        .with_columns(combo_price_flow3_expr())
    )


def load_combo_price_flow3_backtest_frame(
    panel_path: str | Path,
    config: ASharePortfolioConfig,
    *,
    segments: dict[str, tuple[date, date]] = DEFAULT_A_SHARE_SEGMENTS,
) -> pl.DataFrame:
    columns = [
        "trade_date",
        "vt_symbol",
        "name",
        "list_date",
        "is_limit_up",
        "is_limit_down",
        "has_valid_bar",
        "is_list_life_valid",
        config.component_column,
        "close",
        "volume",
        "turnover",
        "circ_mv",
    ]

    return (
        _base_combo_price_flow3_scan(panel_path, config, segments=segments, columns=columns)
        .with_columns((pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1.0).alias("daily_return"))
        .select(
            "trade_date",
            "vt_symbol",
            "daily_return",
            "signal_eligible",
            "benchmark_eligible",
            "combo_price_flow_3",
        )
        .collect()
    )


def load_combo_price_flow3_exposure_frame(
    panel_path: str | Path,
    config: ASharePortfolioConfig,
    *,
    segments: dict[str, tuple[date, date]] = DEFAULT_A_SHARE_SEGMENTS,
) -> pl.DataFrame:
    columns = [
        "trade_date",
        "vt_symbol",
        "name",
        "list_date",
        "is_limit_up",
        "is_limit_down",
        "has_valid_bar",
        "is_list_life_valid",
        config.component_column,
        "close",
        "volume",
        "turnover",
        "turnover_rate",
        "circ_mv",
        "total_mv",
        "pb",
        "pe_ttm",
        "market",
        "sw_l1_name",
        "is_hs300",
        "is_zz500",
        "is_zz1000",
    ]
    entry_close = pl.col("close").shift(-config.entry_lag).over("vt_symbol")
    exit_close = pl.col("close").shift(-(config.rebalance_days + config.entry_lag)).over("vt_symbol")

    return (
        _base_combo_price_flow3_scan(panel_path, config, segments=segments, columns=columns)
        .with_columns((exit_close / entry_close - 1.0).alias("forward_holding_return"))
        .collect()
    )


def rebalance_calendar(dates: list[date], rebalance_days: int, entry_lag: int) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    signal_indices = list(range(0, len(dates), rebalance_days))
    for rebalance_id, signal_index in enumerate(signal_indices):
        effective_start_index = signal_index + entry_lag + 1
        if effective_start_index >= len(dates):
            continue

        next_signal_index = signal_indices[rebalance_id + 1] if rebalance_id + 1 < len(signal_indices) else len(dates)
        effective_end_index = min(next_signal_index + entry_lag, len(dates) - 1)
        if effective_end_index < effective_start_index:
            continue

        rows.append(
            {
                "rebalance_id": rebalance_id,
                "signal_date": dates[signal_index],
                "effective_start": dates[effective_start_index],
                "effective_end": dates[effective_end_index],
            }
        )
    return pl.DataFrame(rows)


def active_rebalance_days(dates: list[date], calendar: pl.DataFrame) -> pl.DataFrame:
    date_index = {trade_date: index for index, trade_date in enumerate(dates)}
    rows: list[dict[str, Any]] = []
    for row in calendar.iter_rows(named=True):
        start_index = date_index[row["effective_start"]]
        end_index = date_index[row["effective_end"]]
        for trade_date in dates[start_index : end_index + 1]:
            rows.append({"trade_date": trade_date, "rebalance_id": row["rebalance_id"]})
    return pl.DataFrame(rows)


def target_positions(frame: pl.DataFrame, calendar: pl.DataFrame, top_n: int) -> pl.DataFrame:
    return (
        frame.join(calendar.select("rebalance_id", "signal_date"), left_on="trade_date", right_on="signal_date", how="inner")
        .filter(pl.col("signal_eligible") & pl.col("combo_price_flow_3").is_not_null() & pl.col("combo_price_flow_3").is_not_nan())
        .with_columns(
            pl.col("combo_price_flow_3").rank(method="ordinal", descending=True).over("trade_date").alias("rank")
        )
        .filter(pl.col("rank") <= top_n)
        .with_columns(
            pl.len().over("rebalance_id").alias("selected_count"),
            (1.0 / pl.len().over("rebalance_id")).alias("weight"),
        )
        .join(calendar, on="rebalance_id", how="left")
        .select("rebalance_id", "effective_start", "effective_end", "vt_symbol", "rank", "weight", "selected_count")
    )


def benchmark_positions(frame: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.join(calendar.select("rebalance_id", "signal_date"), left_on="trade_date", right_on="signal_date", how="inner")
        .filter(pl.col("benchmark_eligible"))
        .with_columns(
            pl.len().over("rebalance_id").alias("benchmark_count"),
            (1.0 / pl.len().over("rebalance_id")).alias("weight"),
        )
        .join(calendar, on="rebalance_id", how="left")
        .select("rebalance_id", "effective_start", "effective_end", "vt_symbol", "weight", "benchmark_count")
    )


def turnover_costs(positions: pl.DataFrame, fee_rate: float) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    previous: dict[str, float] = {}
    for rebalance_id in positions.get_column("rebalance_id").unique().sort().to_list():
        group = positions.filter(pl.col("rebalance_id") == rebalance_id)
        current = dict(zip(group["vt_symbol"].to_list(), group["weight"].to_list(), strict=True))
        symbols = set(previous).union(current)
        turnover = sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols)
        rows.append(
            {
                "trade_date": group["effective_start"][0],
                "rebalance_id": int(rebalance_id),
                "turnover": turnover,
                "cost": turnover * fee_rate,
                "holding_count": group.height,
            }
        )
        previous = current
    return pl.DataFrame(rows)


def gross_returns_for_positions(
    returns: pl.DataFrame,
    positions: pl.DataFrame,
    active_days: pl.DataFrame,
    column_name: str,
) -> pl.DataFrame:
    joined = (
        returns.join(active_days, on="trade_date", how="inner")
        .join(
            positions.select("rebalance_id", "vt_symbol", "weight"),
            on=["rebalance_id", "vt_symbol"],
            how="inner",
        )
    )
    return (
        joined.group_by("trade_date")
        .agg((pl.col("daily_return") * pl.col("weight")).sum().alias(column_name))
        .sort("trade_date")
    )


def run_combo_price_flow3_backtest(frame: pl.DataFrame, config: ASharePortfolioConfig) -> pl.DataFrame:
    dates = frame.get_column("trade_date").unique().sort().to_list()
    calendar = rebalance_calendar(dates, config.rebalance_days, config.entry_lag)
    active_days = active_rebalance_days(dates, calendar)
    positions = target_positions(frame, calendar, config.top_n)
    matched_benchmark_positions = benchmark_positions(frame, calendar)
    fee_rate = config.fee_bps / 10_000.0
    costs = turnover_costs(positions, fee_rate)
    benchmark_costs = turnover_costs(matched_benchmark_positions, fee_rate)

    returns = frame.select("trade_date", "vt_symbol", "daily_return").drop_nulls("daily_return")
    gross = gross_returns_for_positions(returns, positions, active_days, "gross_return")
    matched_benchmark = gross_returns_for_positions(returns, matched_benchmark_positions, active_days, "benchmark_gross_return")
    daily_equal_weight_benchmark = (
        frame.filter(pl.col("benchmark_eligible"))
        .select("trade_date", "daily_return")
        .drop_nulls("daily_return")
        .group_by("trade_date")
        .agg(pl.col("daily_return").mean().alias("daily_equal_weight_benchmark_return"))
        .sort("trade_date")
    )
    return (
        gross.join(matched_benchmark, on="trade_date", how="left")
        .join(daily_equal_weight_benchmark, on="trade_date", how="left")
        .join(costs.select("trade_date", "turnover", "cost", "holding_count"), on="trade_date", how="left")
        .join(
            benchmark_costs.select("trade_date", "turnover", "cost", "holding_count").rename(
                {
                    "turnover": "benchmark_turnover",
                    "cost": "benchmark_cost",
                    "holding_count": "benchmark_holding_count",
                }
            ),
            on="trade_date",
            how="left",
        )
        .with_columns(
            pl.col("turnover").fill_null(0.0),
            pl.col("cost").fill_null(0.0),
            pl.col("holding_count").fill_null(strategy="forward"),
            pl.col("benchmark_turnover").fill_null(0.0),
            pl.col("benchmark_cost").fill_null(0.0),
            pl.col("benchmark_holding_count").fill_null(strategy="forward"),
        )
        .with_columns(
            (pl.col("gross_return") - pl.col("cost")).alias("net_return"),
            (pl.col("benchmark_gross_return") - pl.col("benchmark_cost")).alias("benchmark_return"),
        )
        .with_columns(
            (pl.col("gross_return") - pl.col("benchmark_gross_return")).alias("gross_excess_return"),
            (pl.col("net_return") - pl.col("benchmark_return")).alias("net_excess_return"),
            (pl.col("net_return") - pl.col("daily_equal_weight_benchmark_return")).alias("daily_equal_weight_net_excess_return"),
            pl.lit(config.top_n).alias("top_n"),
            pl.lit(config.rebalance_days).alias("rebalance_days"),
        )
        .sort("trade_date")
    )


def max_drawdown(return_series: pl.Series) -> float:
    equity = (1.0 + return_series).cum_prod()
    drawdown = equity / equity.cum_max() - 1.0
    return _float_or_nan(drawdown.min())


def summarize_returns(
    daily: pl.DataFrame,
    segment: str,
    start: date,
    end: date,
    top_n: int,
    rebalance_days: int,
) -> dict[str, Any]:
    sample = daily.filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
    if sample.is_empty():
        return {
            "top_n": top_n,
            "rebalance_days": rebalance_days,
            "segment": segment,
            "days": 0,
        }

    def metric(prefix: str, column: str) -> dict[str, float]:
        returns = sample[column].fill_null(0.0)
        days = returns.len()
        total = _float_or_nan((1.0 + returns).cum_prod()[-1]) - 1.0
        annual = (1.0 + total) ** (ANNUAL_DAYS / days) - 1.0 if days > 0 and total > -1.0 else float("nan")
        mean = _float_or_nan(returns.mean())
        std = _float_or_nan(returns.std())
        sharpe = mean / std * math.sqrt(ANNUAL_DAYS) if std else float("nan")
        return {
            f"{prefix}_total_return": total,
            f"{prefix}_annual_return": annual,
            f"{prefix}_sharpe": sharpe,
            f"{prefix}_max_drawdown": max_drawdown(returns),
        }

    row: dict[str, Any] = {
        "top_n": top_n,
        "rebalance_days": rebalance_days,
        "segment": segment,
        "start": start,
        "end": end,
        "days": sample.height,
        "avg_holding_count": _float_or_nan(sample["holding_count"].mean()),
        "avg_rebalance_turnover": _float_or_nan(sample.filter(pl.col("turnover") > 0)["turnover"].mean()),
        "annual_turnover": _float_or_nan(sample["turnover"].sum()) * ANNUAL_DAYS / sample.height,
        "annual_cost": _float_or_nan(sample["cost"].sum()) * ANNUAL_DAYS / sample.height,
        "benchmark_avg_holding_count": _float_or_nan(sample["benchmark_holding_count"].mean()),
        "benchmark_avg_rebalance_turnover": _float_or_nan(sample.filter(pl.col("benchmark_turnover") > 0)["benchmark_turnover"].mean()),
        "benchmark_annual_turnover": _float_or_nan(sample["benchmark_turnover"].sum()) * ANNUAL_DAYS / sample.height,
        "benchmark_annual_cost": _float_or_nan(sample["benchmark_cost"].sum()) * ANNUAL_DAYS / sample.height,
    }
    row.update(metric("gross", "gross_return"))
    row.update(metric("net", "net_return"))
    row.update(metric("benchmark_gross", "benchmark_gross_return"))
    row.update(metric("benchmark", "benchmark_return"))
    row.update(metric("daily_equal_weight_benchmark", "daily_equal_weight_benchmark_return"))
    row.update(metric("net_excess", "net_excess_return"))
    row.update(metric("daily_equal_weight_net_excess", "daily_equal_weight_net_excess_return"))
    return row


def summarize_backtest(
    daily: pl.DataFrame,
    config: ASharePortfolioConfig,
    *,
    segments: dict[str, tuple[date, date]] = DEFAULT_A_SHARE_SEGMENTS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    summary_rows = [
        summarize_returns(daily, segment, start, end, config.top_n, config.rebalance_days)
        for segment, (start, end) in segments.items()
    ]
    years = daily.get_column("trade_date").dt.year().unique().sort().to_list()
    yearly_rows = [
        summarize_returns(
            daily,
            str(year),
            date(int(year), 1, 1),
            date(int(year), 12, 31),
            config.top_n,
            config.rebalance_days,
        )
        for year in years
    ]
    return pl.DataFrame(summary_rows), pl.DataFrame(yearly_rows)


def _assert_grid_frame_compatible(configs: list[ASharePortfolioConfig]) -> None:
    if not configs:
        raise ValueError("Expected at least one portfolio config")

    first = configs[0]
    for config in configs[1:]:
        same_frame = (
            config.entry_lag == first.entry_lag
            and config.component_column == first.component_column
            and config.min_turnover == first.min_turnover
            and config.min_circ_mv == first.min_circ_mv
            and config.exclude_new_listing_days == first.exclude_new_listing_days
        )
        if not same_frame:
            raise ValueError("All configs in one grid run must share panel-loading and eligibility settings")


def run_combo_price_flow3_backtest_grid(
    panel_path: str | Path,
    configs: list[ASharePortfolioConfig],
    *,
    segments: dict[str, tuple[date, date]] = DEFAULT_A_SHARE_SEGMENTS,
) -> ASharePortfolioBacktestGridResult:
    _assert_grid_frame_compatible(configs)
    frame = load_combo_price_flow3_backtest_frame(panel_path, configs[0], segments=segments)

    daily_frames: list[pl.DataFrame] = []
    summary_frames: list[pl.DataFrame] = []
    yearly_frames: list[pl.DataFrame] = []

    for config in configs:
        daily = run_combo_price_flow3_backtest(frame, config)
        summary, yearly = summarize_backtest(daily, config, segments=segments)
        daily_frames.append(daily)
        summary_frames.append(summary)
        yearly_frames.append(yearly)

    return ASharePortfolioBacktestGridResult(
        daily=pl.concat(daily_frames),
        summary=pl.concat(summary_frames),
        yearly=pl.concat(yearly_frames),
        row_count=frame.height,
        signal_eligible_count=frame.filter(pl.col("signal_eligible")).height,
    )


def rebalance_signal_dates(frame: pl.DataFrame, rebalance_days: int) -> pl.DataFrame:
    dates = frame.get_column("trade_date").unique().sort().to_list()
    return pl.DataFrame({"trade_date": dates[::rebalance_days]})


def split_selected_and_benchmark(frame: pl.DataFrame, signal_dates: pl.DataFrame, top_n: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    signal_rows = frame.join(signal_dates, on="trade_date", how="inner")
    benchmark = signal_rows.filter(pl.col("benchmark_eligible") & pl.col("forward_holding_return").is_not_null())
    selected = (
        signal_rows.filter(
            pl.col("signal_eligible")
            & pl.col("combo_price_flow_3").is_not_null()
            & pl.col("combo_price_flow_3").is_not_nan()
            & pl.col("forward_holding_return").is_not_null()
        )
        .with_columns(pl.col("combo_price_flow_3").rank(method="ordinal", descending=True).over("trade_date").alias("rank"))
        .filter(pl.col("rank") <= top_n)
    )
    return selected, benchmark


def daily_forward_advantage(selected: pl.DataFrame, benchmark: pl.DataFrame) -> pl.DataFrame:
    selected_daily = selected.group_by("trade_date").agg(
        pl.col("forward_holding_return").mean().alias("selected_forward_return"),
        pl.col("circ_mv").median().alias("selected_median_circ_mv"),
        pl.col("turnover").median().alias("selected_median_turnover"),
        pl.col("turnover_rate").median().alias("selected_median_turnover_rate"),
        pl.col("pb").median().alias("selected_median_pb"),
        pl.len().alias("selected_count"),
    )
    benchmark_daily = benchmark.group_by("trade_date").agg(
        pl.col("forward_holding_return").mean().alias("benchmark_forward_return"),
        pl.col("circ_mv").median().alias("benchmark_median_circ_mv"),
        pl.col("turnover").median().alias("benchmark_median_turnover"),
        pl.col("turnover_rate").median().alias("benchmark_median_turnover_rate"),
        pl.col("pb").median().alias("benchmark_median_pb"),
        pl.len().alias("benchmark_count"),
    )
    return (
        selected_daily.join(benchmark_daily, on="trade_date", how="inner")
        .with_columns(
            (pl.col("selected_forward_return") - pl.col("benchmark_forward_return")).alias("forward_excess_return"),
            (pl.col("selected_median_circ_mv") / pl.col("benchmark_median_circ_mv")).alias("median_circ_mv_ratio"),
            (pl.col("selected_median_turnover") / pl.col("benchmark_median_turnover")).alias("median_turnover_ratio"),
            (pl.col("selected_median_turnover_rate") / pl.col("benchmark_median_turnover_rate")).alias("median_turnover_rate_ratio"),
            (pl.col("selected_median_pb") / pl.col("benchmark_median_pb")).alias("median_pb_ratio"),
        )
        .sort("trade_date")
    )


def _t_stat(values: pl.Series) -> float:
    non_null = values.drop_nulls()
    clean = non_null.filter(non_null.is_not_nan())
    count = clean.len()
    if count < 2:
        return float("nan")
    std = _float_or_nan(clean.std())
    if not std:
        return float("nan")
    return _float_or_nan(clean.mean()) / std * math.sqrt(count)


def exposure_segment_summary(
    daily: pl.DataFrame,
    config: ASharePortfolioConfig,
    *,
    segments: dict[str, tuple[date, date]] = DEFAULT_A_SHARE_SEGMENTS,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment, (start, end) in segments.items():
        sample = daily.filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
        if sample.is_empty():
            continue
        excess = sample["forward_excess_return"]
        rows.append(
            {
                "top_n": config.top_n,
                "rebalance_days": config.rebalance_days,
                "segment": segment,
                "rebalance_count": sample.height,
                "selected_count_mean": _float_or_nan(sample["selected_count"].mean()),
                "benchmark_count_mean": _float_or_nan(sample["benchmark_count"].mean()),
                "selected_forward_return_mean": _float_or_nan(sample["selected_forward_return"].mean()),
                "benchmark_forward_return_mean": _float_or_nan(sample["benchmark_forward_return"].mean()),
                "forward_excess_return_mean": _float_or_nan(excess.mean()),
                "forward_excess_win_rate": _float_or_nan((excess > 0).mean()),
                "forward_excess_t_stat": _t_stat(excess),
                "median_circ_mv_ratio": _float_or_nan(sample["median_circ_mv_ratio"].median()),
                "median_turnover_ratio": _float_or_nan(sample["median_turnover_ratio"].median()),
                "median_turnover_rate_ratio": _float_or_nan(sample["median_turnover_rate_ratio"].median()),
                "median_pb_ratio": _float_or_nan(sample["median_pb_ratio"].median()),
            }
        )
    return pl.DataFrame(rows)


def category_weights(frame: pl.DataFrame, category: str, suffix: str) -> pl.DataFrame:
    return (
        frame.with_columns(pl.col(category).fill_null("UNKNOWN").alias(category))
        .group_by(["trade_date", category])
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum().over("trade_date")).alias(f"{suffix}_weight"))
        .select("trade_date", category, f"{suffix}_weight")
    )


def category_overweight(selected: pl.DataFrame, benchmark: pl.DataFrame, category: str, config: ASharePortfolioConfig) -> pl.DataFrame:
    selected_weights = category_weights(selected, category, "selected")
    benchmark_weights = category_weights(benchmark, category, "benchmark")
    return (
        selected_weights.join(benchmark_weights, on=["trade_date", category], how="full", coalesce=True)
        .with_columns(
            pl.col("selected_weight").fill_null(0.0),
            pl.col("benchmark_weight").fill_null(0.0),
        )
        .with_columns((pl.col("selected_weight") - pl.col("benchmark_weight")).alias("overweight"))
        .group_by(category)
        .agg(
            pl.col("selected_weight").mean().alias("selected_weight"),
            pl.col("benchmark_weight").mean().alias("benchmark_weight"),
            pl.col("overweight").mean().alias("overweight"),
        )
        .with_columns(pl.lit(config.top_n).alias("top_n"), pl.lit(config.rebalance_days).alias("rebalance_days"))
        .sort("overweight", descending=True)
    )


def index_exposure(selected: pl.DataFrame, benchmark: pl.DataFrame, config: ASharePortfolioConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in ("is_hs300", "is_zz500", "is_zz1000"):
        selected_daily = selected.group_by("trade_date").agg(pl.col(column).fill_null(False).mean().alias("selected_weight"))
        benchmark_daily = benchmark.group_by("trade_date").agg(pl.col(column).fill_null(False).mean().alias("benchmark_weight"))
        joined = selected_daily.join(benchmark_daily, on="trade_date", how="inner")
        rows.append(
            {
                "top_n": config.top_n,
                "rebalance_days": config.rebalance_days,
                "index": column,
                "selected_weight": _float_or_nan(joined["selected_weight"].mean()),
                "benchmark_weight": _float_or_nan(joined["benchmark_weight"].mean()),
                "overweight": _float_or_nan((joined["selected_weight"] - joined["benchmark_weight"]).mean()),
            }
        )
    return pl.DataFrame(rows)


def run_combo_price_flow3_exposure_diagnostics(
    panel_path: str | Path,
    config: ASharePortfolioConfig,
    *,
    segments: dict[str, tuple[date, date]] = DEFAULT_A_SHARE_SEGMENTS,
) -> ASharePortfolioExposureResult:
    frame = load_combo_price_flow3_exposure_frame(panel_path, config, segments=segments)
    signal_dates = rebalance_signal_dates(frame, config.rebalance_days)
    selected, benchmark = split_selected_and_benchmark(frame, signal_dates, config.top_n)
    daily = daily_forward_advantage(selected, benchmark)
    return ASharePortfolioExposureResult(
        daily=daily,
        summary=exposure_segment_summary(daily, config, segments=segments),
        industry_overweight=category_overweight(selected, benchmark, "sw_l1_name", config),
        market_overweight=category_overweight(selected, benchmark, "market", config),
        index_exposure=index_exposure(selected, benchmark, config),
    )
