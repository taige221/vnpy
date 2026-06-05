from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from vnpy.alpha.factors import (
    CANDIDATE_BASE_FACTORS,
    CANDIDATE_COMPOSITES,
    PANEL_FEATURE_COLUMNS,
    PANEL_SCAN_HORIZONS,
    PANEL_SCAN_WINDOWS,
    candidate_factor_expressions,
    classic_price_expressions,
    first_pass_factor_expressions,
    panel_expressions,
)

from ..lab import AlphaLab
from .a_share import build_eligibility_from_source_frame
from .a_share_portfolio import DEFAULT_A_SHARE_SEGMENTS, panel_glob
from .signal import to_vt_symbol
from .stat_alpha_loop import StatAlphaLoop


DATE_FORMAT: str = "%Y-%m-%d"
COMPONENT_COLUMNS: dict[str, str | None] = {
    "all": None,
    "a_share_ge1000": "in_a_share_ge1000",
    "a_share_active_ge1000": "in_a_share_active_ge1000",
}


@dataclass(frozen=True)
class ASharePanelScanConfig:
    panel_path: str | Path
    component_column: str = "in_a_share_ge1000"
    entry_lag: int = 1
    min_universe: int = 300
    quantile: float = 0.2
    min_turnover: float = 50_000_000
    min_circ_mv: float = 100_000
    exclude_new_listing_days: int = 120
    windows: tuple[int, ...] = PANEL_SCAN_WINDOWS
    horizons: tuple[int, ...] = PANEL_SCAN_HORIZONS
    segments: dict[str, tuple[date, date]] | None = None


@dataclass(frozen=True)
class AShareFirstPassScanResult:
    scores: pl.DataFrame
    row_count: int
    eligible_row_count: int
    factor_names: list[str]
    label_names: list[str]


@dataclass(frozen=True)
class AShareCandidateComboResult:
    scores: pl.DataFrame
    yearly: pl.DataFrame
    correlations: dict[str, pl.DataFrame]
    row_count: int
    eligible_row_count: int
    factor_names: list[str]


@dataclass(frozen=True)
class ASharePanelAlphaSmokeConfig:
    lab_path: str | Path = "lab/a_share_research"
    panel_path: str | Path | None = None
    component: str = "a_share_ge1000"
    start_date: str = "2022-01-01"
    end_date: str = "2026-06-04"
    train_period: tuple[str, str] = ("2022-01-01", "2023-12-31")
    valid_period: tuple[str, str] = ("2024-01-01", "2024-12-31")
    test_period: tuple[str, str] = ("2025-01-01", "2026-06-04")
    horizons: tuple[int, ...] = (5, 10)
    windows: tuple[int, ...] = (20, 60)
    entry_lag: int = 1
    min_universe: int = 300
    quantile: float = 0.2
    symbol_limit: int | None = None
    exclude_new_listing_days: int = 120
    include_current_st: bool = False
    include_entry_limit: bool = False
    min_turnover: float = 50_000_000
    min_circ_mv: float = 100_000
    top: int = 20


@dataclass(frozen=True)
class ASharePanelAlphaSmokeResult:
    history: pl.DataFrame
    panel_rows: int
    symbol_count: int
    eligible_row_count: int


@dataclass(frozen=True)
class AShareStatAlphaLoopConfig:
    lab_path: str | Path = "lab/a_share_research"
    db_path: str | Path | None = None
    component: str = "a_share_ge1000"
    start_date: str = "2020-01-01"
    end_date: str = datetime.now().strftime(DATE_FORMAT)
    train_period: tuple[str, str] = ("2020-01-01", "2022-12-31")
    valid_period: tuple[str, str] = ("2023-01-01", "2024-12-31")
    test_period: tuple[str, str] = ("2025-01-01", "2026-12-31")
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    windows: tuple[int, ...] = (5, 10, 20, 60, 120)
    entry_lag: int = 1
    extended_days: int = 260
    min_universe: int = 300
    quantile: float = 0.2
    no_eligibility: bool = False
    exclude_new_listing_days: int = 120
    include_current_st: bool = False
    include_entry_limit: bool = False
    min_amount: float = 50_000
    min_circ_mv: float = 100_000
    symbol_limit: int | None = None
    top: int = 20


@dataclass(frozen=True)
class AShareStatAlphaLoopResult:
    history: pl.DataFrame
    bar_rows: int
    eligibility_rows: int | None
    eligible_row_count: int | None


def parse_date(value: str) -> date:
    return datetime.strptime(value, DATE_FORMAT).date()


def parse_datetime(value: str) -> datetime:
    return datetime.strptime(value, DATE_FORMAT)


def parse_period(value: str) -> tuple[str, str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != 2:
        raise ValueError(f"Expected period as start,end, got {value!r}")
    return items[0], items[1]


def parse_int_tuple(value: str) -> tuple[int, ...]:
    numbers = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not numbers:
        raise ValueError("At least one integer is required")
    return numbers


def _segments(config: ASharePanelScanConfig) -> dict[str, tuple[date, date]]:
    return DEFAULT_A_SHARE_SEGMENTS if config.segments is None else config.segments


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _mean_std_ir(series: pl.Series) -> tuple[float, float, float]:
    clean = series.drop_nulls().drop_nans()
    if clean.is_empty():
        return float("nan"), float("nan"), float("nan")
    mean = _float_or_nan(clean.mean())
    std = _float_or_nan(clean.std()) if clean.len() > 1 else float("nan")
    ir = mean / std if std and not math.isnan(std) else float("nan")
    return mean, std, ir


def empty_metrics() -> dict[str, float | int]:
    return {
        "ic": float("nan"),
        "icir": float("nan"),
        "days": 0,
        "positive_rate": float("nan"),
        "samples_mean": float("nan"),
        "spread": float("nan"),
        "spread_ir": float("nan"),
    }


def label_expressions(horizons: tuple[int, ...], entry_lag: int) -> tuple[list[str], list[pl.Expr]]:
    names: list[str] = []
    expressions: list[pl.Expr] = []
    entry_price = pl.col("close").shift(-entry_lag).over("vt_symbol")
    for horizon in horizons:
        name = f"label_{horizon}"
        exit_price = pl.col("close").shift(-(entry_lag + horizon)).over("vt_symbol")
        names.append(name)
        expressions.append((exit_price / entry_price - 1.0).alias(name))
    return names, expressions


def scan_eligibility_expr(config: ASharePanelScanConfig) -> pl.Expr:
    return (
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
    )


def load_first_pass_scan_frame(config: ASharePanelScanConfig) -> tuple[pl.DataFrame, list[str], list[str]]:
    factor_names, factor_exprs = first_pass_factor_expressions(config.windows)
    label_names, label_exprs = label_expressions(config.horizons, config.entry_lag)
    segments = _segments(config)
    min_start = min(start for start, _ in segments.values())
    max_end = max(end for _, end in segments.values())
    columns: list[str] = [
        "trade_date",
        "vt_symbol",
        "name",
        "list_date",
        "is_limit_up",
        "is_limit_down",
        "has_valid_bar",
        "is_list_life_valid",
        config.component_column,
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "pb",
        "pe_ttm",
        "circ_mv",
        "fi_roe",
        "fi_netprofit_yoy",
        "fi_or_yoy",
        "fi_debt_to_assets",
    ]

    frame = (
        pl.scan_parquet(panel_glob(config.panel_path))
        .select(columns)
        .filter((pl.col("trade_date") >= min_start) & (pl.col("trade_date") <= max_end))
        .filter(pl.col(config.component_column).fill_null(False))
        .sort(["vt_symbol", "trade_date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1.0).alias("ret_1d"),
            pl.col("is_limit_up").shift(-config.entry_lag).over("vt_symbol").fill_null(False).alias("entry_limit_up"),
            pl.col("is_limit_down").shift(-config.entry_lag).over("vt_symbol").fill_null(False).alias("entry_limit_down"),
        )
        .with_columns(factor_exprs + label_exprs)
        .with_columns(scan_eligibility_expr(config).alias("eligible"))
        .select("trade_date", "vt_symbol", "eligible", *factor_names, *label_names)
        .collect()
    )
    return frame, factor_names, label_names


def segment_metrics(
    frame: pl.DataFrame,
    factor: str,
    label: str,
    start: date,
    end: date,
    min_universe: int,
    quantile: float,
) -> dict[str, float | int]:
    segment = (
        frame.filter(
            pl.col("eligible")
            & (pl.col("trade_date") >= start)
            & (pl.col("trade_date") <= end)
        )
        .select("trade_date", factor, label)
        .with_columns(pl.col(factor).fill_nan(None), pl.col(label).fill_nan(None))
        .drop_nulls([factor, label])
    )
    if segment.is_empty():
        return empty_metrics()

    ranked = (
        segment.with_columns(
            pl.col(factor).rank().over("trade_date").alias("factor_rank"),
            pl.col(label).rank().over("trade_date").alias("label_rank"),
            pl.len().over("trade_date").alias("sample_count"),
        )
        .filter(pl.col("sample_count") >= min_universe)
        .with_columns((pl.col("factor_rank") / pl.col("sample_count")).alias("factor_pct"))
    )
    if ranked.is_empty():
        return empty_metrics()

    daily_ic = (
        ranked.group_by("trade_date")
        .agg(pl.corr("factor_rank", "label_rank").alias("ic"), pl.first("sample_count").alias("sample_count"))
        .sort("trade_date")
    )
    ic_mean, _, icir = _mean_std_ir(daily_ic["ic"])
    ic_clean = daily_ic["ic"].drop_nulls().drop_nans()
    positive_rate = _float_or_nan((ic_clean > 0).mean()) if not ic_clean.is_empty() else float("nan")
    samples_mean = _float_or_nan(daily_ic["sample_count"].mean())

    daily_spread = (
        ranked.group_by("trade_date")
        .agg(
            pl.col(label).filter(pl.col("factor_pct") >= 1.0 - quantile).mean().alias("top_return"),
            pl.col(label).filter(pl.col("factor_pct") <= quantile).mean().alias("bottom_return"),
        )
        .with_columns((pl.col("top_return") - pl.col("bottom_return")).alias("spread"))
        .sort("trade_date")
    )
    spread_mean, _, spread_ir = _mean_std_ir(daily_spread["spread"])

    return {
        "ic": ic_mean,
        "icir": icir,
        "days": int(ic_clean.len()),
        "positive_rate": positive_rate,
        "samples_mean": samples_mean,
        "spread": spread_mean,
        "spread_ir": spread_ir,
    }


def score_factor_frame(
    frame: pl.DataFrame,
    factor_names: list[str],
    label_names: list[str],
    config: ASharePanelScanConfig,
    *,
    include_abs_valid_ic: bool,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    segments = _segments(config)
    for factor in factor_names:
        for label in label_names:
            horizon = int(label.split("_", maxsplit=1)[1])
            row: dict[str, Any] = {"factor": factor, "horizon": horizon}
            for segment, (start, end) in segments.items():
                metrics = segment_metrics(frame, factor, label, start, end, config.min_universe, config.quantile)
                for key, value in metrics.items():
                    row[f"{segment}_{key}"] = value

            row["valid_test_same_sign"] = (
                bool(row["valid_ic"] * row["test_ic"] > 0)
                if not (math.isnan(row["valid_ic"]) or math.isnan(row["test_ic"]))
                else False
            )
            row["all_segments_same_sign"] = (
                bool(row["train_ic"] * row["valid_ic"] > 0 and row["valid_ic"] * row["test_ic"] > 0)
                if not (math.isnan(row["train_ic"]) or math.isnan(row["valid_ic"]) or math.isnan(row["test_ic"]))
                else False
            )
            row["abs_test_ic"] = abs(row["test_ic"]) if not math.isnan(row["test_ic"]) else float("nan")
            if include_abs_valid_ic:
                row["abs_valid_ic"] = abs(row["valid_ic"]) if not math.isnan(row["valid_ic"]) else float("nan")
            rows.append(row)

    sort_columns = ["valid_test_same_sign", "abs_test_ic", "abs_valid_ic"] if include_abs_valid_ic else ["abs_test_ic", "factor", "horizon"]
    sort_descending = [True, True, True] if include_abs_valid_ic else [True, False, False]
    return pl.DataFrame(rows).sort(sort_columns, descending=sort_descending)


def run_first_pass_panel_factor_scan(config: ASharePanelScanConfig) -> AShareFirstPassScanResult:
    frame, factor_names, label_names = load_first_pass_scan_frame(config)
    scores = score_factor_frame(frame, factor_names, label_names, config, include_abs_valid_ic=True)
    return AShareFirstPassScanResult(
        scores=scores,
        row_count=frame.height,
        eligible_row_count=frame.filter(pl.col("eligible")).height,
        factor_names=factor_names,
        label_names=label_names,
    )


def load_candidate_combo_frame(config: ASharePanelScanConfig) -> pl.DataFrame:
    label_names, label_exprs = label_expressions(config.horizons, config.entry_lag)
    segments = _segments(config)
    min_start = min(start for start, _ in segments.values())
    max_end = max(end for _, end in segments.values())
    columns: list[str] = [
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
        "fi_netprofit_yoy",
    ]
    return (
        pl.scan_parquet(panel_glob(config.panel_path))
        .select(columns)
        .filter((pl.col("trade_date") >= min_start) & (pl.col("trade_date") <= max_end))
        .filter(pl.col(config.component_column).fill_null(False))
        .sort(["vt_symbol", "trade_date"])
        .with_columns(
            pl.col("is_limit_up").shift(-config.entry_lag).over("vt_symbol").fill_null(False).alias("entry_limit_up"),
            pl.col("is_limit_down").shift(-config.entry_lag).over("vt_symbol").fill_null(False).alias("entry_limit_down"),
        )
        .with_columns(candidate_factor_expressions() + label_exprs)
        .with_columns(scan_eligibility_expr(config).alias("eligible"))
        .select("trade_date", "vt_symbol", "eligible", *CANDIDATE_BASE_FACTORS, *label_names)
        .collect()
    )


def add_candidate_composites(frame: pl.DataFrame) -> pl.DataFrame:
    current = frame
    for name, components in CANDIDATE_COMPOSITES.items():
        rank_columns = [f"_{name}_{component}_rank" for component in components]
        current = current.with_columns(
            [
                (pl.col(component).rank().over("trade_date") / pl.len().over("trade_date")).alias(rank_column)
                for component, rank_column in zip(components, rank_columns, strict=True)
            ]
        )
        current = current.with_columns(
            pl.mean_horizontal([pl.col(rank_column) for rank_column in rank_columns]).alias(name)
        ).drop(rank_columns)
    return current


def yearly_breakdown(frame: pl.DataFrame, factor_names: list[str], config: ASharePanelScanConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    years = (
        frame.filter(pl.col("eligible"))
        .select("trade_date")
        .with_columns(pl.col("trade_date").dt.year().alias("year"))
        .get_column("year")
        .unique()
        .sort()
        .to_list()
    )
    for factor in factor_names:
        for horizon in config.horizons:
            label = f"label_{horizon}"
            for year in years:
                metrics = segment_metrics(
                    frame,
                    factor,
                    label,
                    date(int(year), 1, 1),
                    date(int(year), 12, 31),
                    config.min_universe,
                    config.quantile,
                )
                row: dict[str, Any] = {"factor": factor, "horizon": horizon, "year": int(year)}
                row.update(metrics)
                rows.append(row)
    return pl.DataFrame(rows).sort(["factor", "horizon", "year"])


def candidate_correlation_matrix(frame: pl.DataFrame, segment: str, config: ASharePanelScanConfig) -> pl.DataFrame:
    segments = _segments(config)
    start, end = segments[segment]
    factor_names = list(CANDIDATE_BASE_FACTORS)
    ranked = (
        frame.filter(
            pl.col("eligible")
            & (pl.col("trade_date") >= start)
            & (pl.col("trade_date") <= end)
        )
        .select("trade_date", *factor_names)
        .with_columns([pl.col(factor).fill_nan(None) for factor in factor_names])
        .drop_nulls(factor_names)
        .with_columns(
            [
                (pl.col(factor).rank().over("trade_date") / pl.len().over("trade_date")).alias(factor)
                for factor in factor_names
            ]
        )
    )
    rows: list[dict[str, float | str]] = []
    for left in factor_names:
        row: dict[str, float | str] = {"factor": left}
        for right in factor_names:
            row[right] = _float_or_nan(ranked.select(pl.corr(left, right)).item())
        rows.append(row)
    return pl.DataFrame(rows)


def run_candidate_factor_combo_scan(config: ASharePanelScanConfig) -> AShareCandidateComboResult:
    frame = add_candidate_composites(load_candidate_combo_frame(config))
    factor_names = list(CANDIDATE_BASE_FACTORS) + list(CANDIDATE_COMPOSITES)
    label_names = [f"label_{horizon}" for horizon in config.horizons]
    scores = score_factor_frame(frame, factor_names, label_names, config, include_abs_valid_ic=False)
    yearly = yearly_breakdown(frame, factor_names, config)
    correlations = {
        segment: candidate_correlation_matrix(frame, segment, config)
        for segment in ("valid", "test")
    }
    return AShareCandidateComboResult(
        scores=scores,
        yearly=yearly,
        correlations=correlations,
        row_count=frame.height,
        eligible_row_count=frame.filter(pl.col("eligible")).height,
        factor_names=factor_names,
    )


def _resolve_panel_path(config: ASharePanelAlphaSmokeConfig) -> Path:
    if config.panel_path is not None:
        return Path(config.panel_path)
    return Path(config.lab_path) / "panel" / "research_panel_daily"


def build_panel_eligibility_frame(
    panel: pl.DataFrame,
    *,
    entry_lag: int,
    exclude_new_listing_days: int,
    include_current_st: bool,
    include_entry_limit: bool,
    min_turnover: float,
    min_circ_mv: float,
) -> pl.DataFrame:
    df = panel.with_columns(
        pl.col("is_limit_up").shift(-entry_lag).over("vt_symbol").fill_null(False).alias("entry_limit_up"),
        pl.col("is_limit_down").shift(-entry_lag).over("vt_symbol").fill_null(False).alias("entry_limit_down"),
    )
    eligible = (
        pl.col("has_valid_bar").fill_null(False)
        & pl.col("is_list_life_valid").fill_null(False)
        & pl.col("list_date").is_not_null()
        & pl.col("close").is_not_null()
        & (pl.col("close") > 0)
    )
    if exclude_new_listing_days > 0:
        eligible &= pl.col("datetime") >= pl.col("list_date").dt.offset_by(f"{exclude_new_listing_days}d")
    if not include_current_st:
        eligible &= ~pl.col("name").fill_null("").str.contains("ST|退")
    if not include_entry_limit:
        eligible &= ~(pl.col("entry_limit_up") | pl.col("entry_limit_down"))
    if min_turnover > 0:
        eligible &= pl.col("turnover").fill_null(0) >= min_turnover
    if min_circ_mv > 0:
        eligible &= pl.col("circ_mv").fill_null(0) >= min_circ_mv

    return df.select("datetime", "vt_symbol", eligible.fill_null(False).alias("eligible"))


def load_panel_alpha_frames(config: ASharePanelAlphaSmokeConfig) -> tuple[pl.DataFrame, pl.DataFrame, int, int]:
    panel_path = _resolve_panel_path(config)
    component_column = COMPONENT_COLUMNS[config.component]
    start_date = parse_date(config.start_date)
    end_date = parse_date(config.end_date)

    columns: list[str] = [
        "trade_date",
        "vt_symbol",
        "name",
        "list_date",
        "delist_date",
        "is_limit_up",
        "is_limit_down",
        "has_valid_bar",
        "is_list_life_valid",
    ]
    columns.extend(PANEL_FEATURE_COLUMNS)
    if component_column:
        columns.append(component_column)

    lazy_frame = (
        pl.scan_parquet(panel_glob(panel_path))
        .select(columns)
        .filter((pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date))
    )
    if component_column:
        lazy_frame = lazy_frame.filter(pl.col(component_column).fill_null(False))

    if config.symbol_limit:
        symbols = (
            lazy_frame.select("vt_symbol")
            .unique()
            .sort("vt_symbol")
            .limit(config.symbol_limit)
            .collect()
            .get_column("vt_symbol")
            .to_list()
        )
        lazy_frame = lazy_frame.filter(pl.col("vt_symbol").is_in(symbols))

    panel = (
        lazy_frame.collect()
        .with_columns(
            pl.col("trade_date").cast(pl.Datetime).alias("datetime"),
            pl.col("list_date").cast(pl.Datetime),
            pl.col("delist_date").cast(pl.Datetime),
        )
        .sort(["vt_symbol", "datetime"])
    )
    if panel.is_empty():
        raise RuntimeError(f"No panel rows loaded from {panel_path}")

    bar_df = panel.select("datetime", "vt_symbol", *PANEL_FEATURE_COLUMNS)
    eligibility_df = build_panel_eligibility_frame(
        panel,
        entry_lag=config.entry_lag,
        exclude_new_listing_days=config.exclude_new_listing_days,
        include_current_st=config.include_current_st,
        include_entry_limit=config.include_entry_limit,
        min_turnover=config.min_turnover,
        min_circ_mv=config.min_circ_mv,
    )
    symbol_count = int(panel.select(pl.col("vt_symbol").n_unique()).item())
    return bar_df, eligibility_df, panel.height, symbol_count


def run_panel_alpha_smoke(config: ASharePanelAlphaSmokeConfig) -> ASharePanelAlphaSmokeResult:
    bar_df, eligibility_df, panel_rows, symbol_count = load_panel_alpha_frames(config)
    loop = StatAlphaLoop(
        bar_df,
        train_period=config.train_period,
        valid_period=config.valid_period,
        test_period=config.test_period,
        horizons=config.horizons,
        entry_lag=config.entry_lag,
        min_universe=config.min_universe,
        quantile=config.quantile,
        eligibility_df=eligibility_df,
    )
    loop.score_batch(panel_expressions(config.windows), top=config.top)
    return ASharePanelAlphaSmokeResult(
        history=loop.history_frame(),
        panel_rows=panel_rows,
        symbol_count=symbol_count,
        eligible_row_count=eligibility_df.filter(pl.col("eligible")).height,
    )


def load_component_bars(config: AShareStatAlphaLoopConfig) -> pl.DataFrame:
    lab = AlphaLab(str(config.lab_path))
    symbols = sorted(lab.load_component_symbols(config.component, config.start_date, config.end_date))
    if config.symbol_limit:
        symbols = symbols[: config.symbol_limit]
    if not symbols:
        raise RuntimeError(f"No symbols found for component {config.component!r}")

    bar_df = lab.load_bar_df(
        symbols,
        "d",
        config.start_date,
        config.end_date,
        config.extended_days,
    )
    if bar_df is None or bar_df.is_empty():
        raise RuntimeError("No bars loaded from AlphaLab")
    return bar_df


def load_a_share_eligibility(config: AShareStatAlphaLoopConfig) -> pl.DataFrame:
    import duckdb  # type: ignore[import-not-found]

    lab_path = Path(config.lab_path)
    db_path = Path(config.db_path) if config.db_path else lab_path / "source" / "tushare_full.duckdb"
    start_date = parse_datetime(config.start_date)
    end_date = parse_datetime(config.end_date)
    load_end = end_date + timedelta(days=45)

    query = """
        SELECT
            d.ts_code,
            d.trade_date AS datetime,
            d.close AS raw_close,
            d.amount,
            b.circ_mv,
            l.up_limit,
            l.down_limit,
            s.name,
            s.list_date,
            s.delist_date
        FROM daily_raw d
        LEFT JOIN daily_basic b
            ON d.ts_code = b.ts_code AND d.trade_date = b.trade_date
        LEFT JOIN stk_limit l
            ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
        LEFT JOIN stock_basic s
            ON d.ts_code = s.ts_code
        WHERE d.trade_date BETWEEN ? AND ?
        ORDER BY d.ts_code, d.trade_date
    """
    with duckdb.connect(str(db_path), read_only=True) as con:
        source = pl.from_pandas(con.execute(query, [start_date.date(), load_end.date()]).fetchdf())

    source = source.with_columns(pl.col("ts_code").map_elements(to_vt_symbol, return_dtype=pl.String).alias("vt_symbol"))
    return build_eligibility_from_source_frame(
        source,
        start_date=start_date,
        end_date=end_date,
        entry_lag=config.entry_lag,
        exclude_new_listing_days=config.exclude_new_listing_days,
        include_current_st=config.include_current_st,
        include_entry_limit=config.include_entry_limit,
        min_amount=config.min_amount,
        min_circ_mv=config.min_circ_mv,
    )


def run_a_share_stat_alpha_loop(config: AShareStatAlphaLoopConfig) -> AShareStatAlphaLoopResult:
    bar_df = load_component_bars(config)
    eligibility_df = None if config.no_eligibility else load_a_share_eligibility(config)
    loop = StatAlphaLoop(
        bar_df,
        train_period=config.train_period,
        valid_period=config.valid_period,
        test_period=config.test_period,
        horizons=config.horizons,
        entry_lag=config.entry_lag,
        min_universe=config.min_universe,
        quantile=config.quantile,
        eligibility_df=eligibility_df,
    )
    loop.score_batch(classic_price_expressions(config.windows), top=config.top)
    return AShareStatAlphaLoopResult(
        history=loop.history_frame(),
        bar_rows=bar_df.height,
        eligibility_rows=None if eligibility_df is None else eligibility_df.height,
        eligible_row_count=None if eligibility_df is None else eligibility_df.filter(pl.col("eligible")).height,
    )
