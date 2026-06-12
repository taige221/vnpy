"""Compare L1 v1 rank scores under the clean signal-close timeline.

This script keeps the buy signal, day filter, and next-open execution rules
fixed. It only changes the T-close ranking score used to form the
``signal_close_topN`` pool, then evaluates the selected next-open topK names
with offline future labels.

Decision stages:

- signal_close_day_filter: T close day-level gate; no ``future_*`` or
  ``next_*`` fields are allowed.
- signal_close_rank_pool: T close stock-level ranking; no ``future_*`` or
  ``next_*`` fields are allowed.
- next_open_execution: T+1 open execution filter; next-open fields are allowed.
- offline_outcome: future labels are used only to score historical outcomes.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.alpha_research.tradingview.signal_core.buy_points import (
    BUY_POINT_TYPE_L1,
    add_buy_point_columns,
)


DEFAULT_SNAPSHOT_PATH = (
    "examples/alpha_research/tradingview/reports/a5_signal_snapshot/"
    "a5_signal_snapshot.parquet"
)
DEFAULT_OUTPUT_DIR = "scripts/alpha_research/tradingview/reports/rank_score_study"
DEFAULT_PANEL_PATH = "lab/a_share_research/panel/research_panel_daily"
BUY_POINT_TYPE = BUY_POINT_TYPE_L1
DAY_FILTER_COLUMN = "rank_score_core4_short70_30"
DAY_FILTER_THRESHOLD = 0.52
SIGNAL_CLOSE_TOP_N = 20
TOP_K_VALUES = "1,2,5"
OPEN_RATE = 0.0005
CLOSE_RATE = 0.0015
GAP_MIN = -0.03
GAP_MAX = 0.01
MIN_CIRC_MV = 200000.0
MIN_TURNOVER_RATE = 0.5
SIGNAL_CLOSE_FORBIDDEN_PREFIXES = ("future_", "next_")
SIGNAL_CLOSE_FORBIDDEN_COLUMNS = {
    "tradable_flag",
    "tradable_reason",
    "suggested_entry_date",
    "suggested_entry_price",
    "execution_bucket",
    "open_executable",
    "next_open_executable",
    "next_open_selected",
}
BASE_SCORE_COLUMNS = {
    "core4_short70_30": "rank_score_core4_short70_30",
    "short_tp_v1": "rank_score_short_tp_v1",
    "core4_control": "rank_score_core4_control",
    "pvcorr60_15pct": "rank_score_pvcorr60_15pct",
    "candidate_pool": "candidate_pool_score",
    "start_score": "start_score",
}
COMPUTED_SCORE_COLUMNS = {
    "l1_eod_v1": "l1_rank_score_eod_v1",
    "l1_balanced_v1": "l1_rank_score_balanced_v1",
    "l1_short_focus_v1": "l1_rank_score_short_focus_v1",
    "l1_context_safe_v1": "l1_rank_score_context_safe_v1",
    "l1_thrust_v1": "l1_rank_score_thrust_v1",
    "blend_core4_candidate_80_20": "blend_core4_candidate_80_20",
    "blend_core4_candidate_70_30": "blend_core4_candidate_70_30",
    "blend_core4_candidate_60_40": "blend_core4_candidate_60_40",
    "blend_core4_pvcorrpct_80_20": "blend_core4_pvcorrpct_80_20",
    "blend_core4_pvcorrpct_70_30": "blend_core4_pvcorrpct_70_30",
}
SINGLE_FACTOR_SCAN_FIELDS = [
    "factor_combo_core4_control",
    "factor_combo_core4_pvcorr60_15pct",
    "factor_combo_pvcorr15_broad_antitrix14_10pct",
    "factor_combo_pvcorr15_broad_top5_10pct",
    "factor_broad_top5_rank",
    "factor_core4_pct",
    "factor_pvcorr60_pct",
    "factor_antitrix14_pct",
    "factor_broad_top5_pct",
    "start_score_pct",
    "candidate_pool_score",
    "short_tp_score_v1",
    "start_score",
    "bar_range_pct",
    "real_body_pct",
    "turnover_rate",
    "close_vs_ema20",
    "close_vs_ema60",
    "ema20_vs_ema60",
    "plus_di",
    "macd_hist",
    "vol_ratio",
    "close_position",
    "rsi",
    "circ_mv",
    "industry_rs_bucket_score",
    "market_breadth_bucket_score",
    "vol_quality_bucket_score",
    "parent_l1_mfe_so_far",
    "parent_l1_mae_so_far",
    "parent_l1_return_so_far",
]
SINGLE_FACTOR_SCORE_COLUMNS = {
    **{f"sf_{field}_high": f"sf_{field}_high" for field in SINGLE_FACTOR_SCAN_FIELDS},
    **{f"sf_{field}_low": f"sf_{field}_low" for field in SINGLE_FACTOR_SCAN_FIELDS},
}
EXTRA_FACTOR_SCAN_FIELDS = [
    "factor_box_stack_lift_reversal_20",
    "factor_common_reversal_20",
    "factor_common_reversal_60",
    "factor_common_anti_limit_up_heat_20",
    "factor_box_stack_lift_reversal_30",
    "factor_double_bottom_low_similarity_60",
    "factor_common_reversal_10",
    "factor_common_anti_limit_up_heat_5",
    "factor_tv_trend_rsi_v2_down_context",
    "factor_common_value_pb_percentile_60",
    "factor_anti_ma_bias_60",
    "factor_anti_log_turnover_mean_5",
    "factor_anti_macd_dif_12_26_9",
    "factor_anti_log_turnover_mean_10",
    "factor_anti_macd_dea_12_26_9",
    "factor_anti_turnover_rate",
    "factor_anti_trix_14",
    "factor_anti_atr_pct_5",
]
EXTRA_FACTOR_SCORE_NAMES = [
    *[f"sf_{field}_high" for field in EXTRA_FACTOR_SCAN_FIELDS],
    *[f"sf_{field}_low" for field in EXTRA_FACTOR_SCAN_FIELDS],
]
CONDITIONAL_SCORE_COLUMNS = {
    "cond_panic_core4_else_candidate": "cond_panic_core4_else_candidate",
    "cond_panic_candidate_else_core4": "cond_panic_candidate_else_core4",
    "cond_both_core4_else_short": "cond_both_core4_else_short",
    "cond_grade_a_core4_else_short": "cond_grade_a_core4_else_short",
    "cond_weak_market_context_else_core4": "cond_weak_market_context_else_core4",
    "cond_weak_volume_candidate_else_core4": "cond_weak_volume_candidate_else_core4",
    "cond_strong_industry_core4_else_candidate": "cond_strong_industry_core4_else_candidate",
}
RETURN_COLUMNS = {
    "tp5_plan": "tp5_plan_return",
    "fixed_1d": "fixed_1d_return",
    "fixed_3d": "fixed_3d_return",
    "fixed_5d": "fixed_5d_return",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run L1 rank-score comparison study")
    parser.add_argument("--snapshot-path", default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-path", default=DEFAULT_PANEL_PATH)
    parser.add_argument("--day-filter-threshold", type=float, default=DAY_FILTER_THRESHOLD)
    parser.add_argument("--signal-close-top-n", type=int, default=SIGNAL_CLOSE_TOP_N)
    parser.add_argument("--top-k", default=TOP_K_VALUES)
    parser.add_argument("--gap-min", type=float, default=GAP_MIN)
    parser.add_argument("--gap-max", type=float, default=GAP_MAX)
    parser.add_argument("--min-circ-mv", type=float, default=MIN_CIRC_MV)
    parser.add_argument("--min-turnover-rate", type=float, default=MIN_TURNOVER_RATE)
    parser.add_argument(
        "--industry-cap",
        default="0.3",
        help="Max share per industry among selected names. Use 'none' to disable.",
    )
    parser.add_argument(
        "--rank-scores",
        default=",".join(
            [
                *BASE_SCORE_COLUMNS,
                *COMPUTED_SCORE_COLUMNS,
                *SINGLE_FACTOR_SCORE_COLUMNS,
                *CONDITIONAL_SCORE_COLUMNS,
            ]
        ),
        help="Comma-separated score names or existing score columns.",
    )
    parser.add_argument(
        "--write-event-selection",
        action="store_true",
        help="Write full event-level selection details. Large for broad scans.",
    )
    parser.add_argument(
        "--extra-factor-scan",
        action="store_true",
        help="Compute and scan extra T-close factor candidates from the local daily panel.",
    )
    return parser.parse_args()


def normalize_date(values: Any) -> pd.Series:
    """Normalize a date-like series."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def numeric(frame: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    """Return a numeric series, or a default-valued series when missing."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    """Return a bool series, or a default-valued series when missing."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    return frame[column].astype("boolean").fillna(default).astype(bool)


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"invalid top-k list: {value}")
    return sorted(set(values))


def parse_rank_scores(value: str) -> list[str]:
    """Parse a comma-separated score list."""
    scores = [item.strip() for item in value.split(",") if item.strip()]
    if not scores:
        raise ValueError("rank score list cannot be empty")
    return scores


def parse_industry_cap(value: str) -> float | None:
    """Parse the optional industry cap."""
    if value.lower() in {"", "none", "null", "off", "false"}:
        return None
    cap = float(value)
    if cap <= 0:
        raise ValueError("--industry-cap must be positive or 'none'")
    return cap


def assert_signal_close_columns(columns: list[str], *, stage: str) -> None:
    """Reject next-open or future fields in signal-close decisions."""
    blocked = [
        column
        for column in columns
        if column in SIGNAL_CLOSE_FORBIDDEN_COLUMNS
        or column.startswith(SIGNAL_CLOSE_FORBIDDEN_PREFIXES)
        or column.startswith("risk_")
    ]
    if blocked:
        raise ValueError(f"{stage} uses non signal-close fields: {sorted(set(blocked))}")


def drawdown(return_series: pd.Series) -> float:
    """Return maximum drawdown for a compounded return series."""
    values = pd.to_numeric(return_series, errors="coerce").dropna()
    if values.empty:
        return math.nan
    equity = (1.0 + values).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def pct_rank_by_day(frame: pd.DataFrame, column: str, *, ascending: bool = True) -> pd.Series:
    """Rank one column within each signal date."""
    values = numeric(frame, column)
    return values.groupby(frame["signal_date"]).rank(pct=True, method="average", ascending=ascending)


def fill_score(series: pd.Series, default: float = 0.5) -> pd.Series:
    """Fill score components with a neutral value and clip into [0, 1]."""
    return pd.to_numeric(series, errors="coerce").fillna(default).clip(0.0, 1.0)


def mean_components(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Average score components row-wise."""
    if not columns:
        return pd.Series(0.5, index=frame.index)
    return frame[columns].mean(axis=1, skipna=True).fillna(0.5).clip(0.0, 1.0)


def rolling_mean_by_symbol(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Rolling mean within each symbol."""
    return frame.groupby("vt_symbol", sort=False)[column].transform(
        lambda values: values.rolling(window, min_periods=window).mean()
    )


def rolling_min_by_symbol(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Rolling min within each symbol."""
    return frame.groupby("vt_symbol", sort=False)[column].transform(
        lambda values: values.rolling(window, min_periods=window).min()
    )


def rolling_max_by_symbol(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Rolling max within each symbol."""
    return frame.groupby("vt_symbol", sort=False)[column].transform(
        lambda values: values.rolling(window, min_periods=window).max()
    )


def rolling_sum_by_symbol(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Rolling sum within each symbol."""
    return frame.groupby("vt_symbol", sort=False)[column].transform(
        lambda values: values.rolling(window, min_periods=window).sum()
    )


def rolling_pct_rank_by_symbol(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Rolling percentile rank of the current value within each symbol."""
    return frame.groupby("vt_symbol", sort=False)[column].transform(
        lambda values: values.rolling(window, min_periods=window).rank(pct=True)
    )


def ewm_by_symbol(frame: pd.DataFrame, column: str, *, span: int) -> pd.Series:
    """EMA within each symbol using pandas' standard recursive EWM."""
    return frame.groupby("vt_symbol", sort=False)[column].transform(
        lambda values: values.ewm(span=span, adjust=False, min_periods=span).mean()
    )


def wilder_ewm_by_symbol(values: pd.Series, symbols: pd.Series, *, window: int) -> pd.Series:
    """Wilder-style smoothing within each symbol."""
    return values.groupby(symbols, sort=False).transform(
        lambda series: series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    )


def load_panel_factor_source(
    panel_path: str,
    *,
    signal_dates: pd.Series,
    symbols: pd.Series,
) -> pd.DataFrame:
    """Load enough local panel history to calculate signal-close factors."""
    base = Path(panel_path)
    if not base.exists():
        print(f"skip extra factor scan: panel path not found: {panel_path}")
        return pd.DataFrame(columns=["datetime", "vt_symbol"])

    dates = normalize_date(signal_dates).dropna()
    if dates.empty:
        return pd.DataFrame(columns=["datetime", "vt_symbol"])
    start = dates.min() - pd.Timedelta(days=420)
    end = dates.max()
    symbol_set = set(symbols.dropna().astype(str).unique())
    columns = [
        "trade_date",
        "vt_symbol",
        "high",
        "low",
        "close",
        "raw_close",
        "up_limit",
        "turnover",
        "turnover_rate",
        "pb",
        "is_limit_up",
    ]
    frames: list[pd.DataFrame] = []
    for file_path in sorted(base.glob("year=*/data_*.parquet")):
        frame = pd.read_parquet(file_path, columns=columns)
        frame["datetime"] = normalize_date(frame["trade_date"])
        frame = frame[
            (frame["datetime"] >= start)
            & (frame["datetime"] <= end)
            & frame["vt_symbol"].astype(str).isin(symbol_set)
        ].drop(columns=["trade_date"])
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["datetime", "vt_symbol"])

    out = pd.concat(frames, ignore_index=True)
    for column in columns:
        if column not in {"trade_date", "vt_symbol", "is_limit_up"}:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["is_limit_up"] = out["is_limit_up"].astype("boolean").fillna(False).astype(bool)
    return out.sort_values(["vt_symbol", "datetime"]).drop_duplicates(["datetime", "vt_symbol"])


def add_extra_signal_close_factors(frame: pd.DataFrame, panel_path: str) -> pd.DataFrame:
    """Add direction-adjusted extra factors for signal-close ranking scans."""
    factors = load_panel_factor_source(
        panel_path,
        signal_dates=frame["signal_date"],
        symbols=frame["vt_symbol"],
    )
    out = frame.copy()
    if factors.empty:
        for column in EXTRA_FACTOR_SCAN_FIELDS:
            out[column] = math.nan
        return out

    source = factors.copy()
    group = source.groupby("vt_symbol", sort=False)
    close = source["close"]
    high = source["high"]
    low = source["low"]
    turnover_rate = source["turnover_rate"]
    prev_close = group["close"].shift(1)

    for window in (10, 20, 60):
        source[f"factor_common_reversal_{window}"] = -1.0 * (close / group["close"].shift(window) - 1.0)

    for window in (20, 30):
        prior_resistance = rolling_max_by_symbol(source.assign(_high_lag=group["high"].shift(1)), "_high_lag", window)
        prior_support = rolling_min_by_symbol(source.assign(_low_lag=group["low"].shift(1)), "_low_lag", window)
        resistance_lift = prior_resistance / prior_resistance.groupby(source["vt_symbol"], sort=False).shift(window) - 1.0
        support_lift = prior_support / prior_support.groupby(source["vt_symbol"], sort=False).shift(window) - 1.0
        source[f"factor_box_stack_lift_reversal_{window}"] = -1.0 * (resistance_lift + support_lift) / 2.0

    for window in (5, 20):
        source[f"factor_common_anti_limit_up_heat_{window}"] = -1.0 * rolling_sum_by_symbol(
            source.assign(_limit_up=source["is_limit_up"].astype(float)),
            "_limit_up",
            window,
        )

    support_60 = rolling_min_by_symbol(source.assign(_low_lag=group["low"].shift(1)), "_low_lag", 60)
    source["factor_double_bottom_low_similarity_60"] = -1.0 * (low / (support_60 + 1e-12) - 1.0).abs()
    source["factor_common_value_pb_percentile_60"] = -1.0 * rolling_pct_rank_by_symbol(source, "pb", 60)
    source["factor_anti_ma_bias_60"] = -1.0 * (close / (rolling_mean_by_symbol(source, "close", 60) + 1e-12) - 1.0)
    source["factor_anti_log_turnover_mean_5"] = -1.0 * np.log(rolling_mean_by_symbol(source, "turnover", 5) + 1.0)
    source["factor_anti_log_turnover_mean_10"] = -1.0 * np.log(rolling_mean_by_symbol(source, "turnover", 10) + 1.0)
    source["factor_anti_turnover_rate"] = -1.0 * turnover_rate

    ema12 = ewm_by_symbol(source, "close", span=12)
    ema26 = ewm_by_symbol(source, "close", span=26)
    macd_dif = ema12 - ema26
    macd_dea = macd_dif.groupby(source["vt_symbol"], sort=False).transform(
        lambda values: values.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    source["factor_anti_macd_dif_12_26_9"] = -1.0 * macd_dif / (close + 1e-12)
    source["factor_anti_macd_dea_12_26_9"] = -1.0 * macd_dea / (close + 1e-12)

    ema1 = ewm_by_symbol(source, "close", span=14)
    source["_trix_ema2"] = ema1.groupby(source["vt_symbol"], sort=False).transform(
        lambda values: values.ewm(span=14, adjust=False, min_periods=14).mean()
    )
    source["_trix_ema3"] = source["_trix_ema2"].groupby(source["vt_symbol"], sort=False).transform(
        lambda values: values.ewm(span=14, adjust=False, min_periods=14).mean()
    )
    trix = source["_trix_ema3"] / source["_trix_ema3"].groupby(source["vt_symbol"], sort=False).shift(1) - 1.0
    source["factor_anti_trix_14"] = -1.0 * trix

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    source["factor_anti_atr_pct_5"] = -1.0 * wilder_ewm_by_symbol(
        true_range,
        source["vt_symbol"],
        window=5,
    ) / (close + 1e-12)

    keep = ["datetime", "vt_symbol", *[column for column in EXTRA_FACTOR_SCAN_FIELDS if column in source.columns]]
    source = source[keep].drop_duplicates(["datetime", "vt_symbol"])
    out = out.merge(
        source,
        left_on=["signal_date", "vt_symbol"],
        right_on=["datetime", "vt_symbol"],
        how="left",
    ).drop(columns=["datetime"], errors="ignore")
    out["factor_tv_trend_rsi_v2_down_context"] = bool_series(out, "down_context").astype(float)
    for column in EXTRA_FACTOR_SCAN_FIELDS:
        if column not in out.columns:
            out[column] = math.nan
    return out


def add_l1_rank_score_candidates(frame: pd.DataFrame, *, extra_factor_fields: list[str] | None = None) -> pd.DataFrame:
    """Add experimental T-close L1 ranking scores."""
    out = frame.copy()
    factor_fields = [*SINGLE_FACTOR_SCAN_FIELDS, *(extra_factor_fields or [])]
    source_columns = [
        "start_score",
        "bar_range_pct",
        "real_body_pct",
        "turnover_rate",
        "close_vs_ema20",
        "close_vs_ema60",
        "ema20_vs_ema60",
        "plus_di",
        "macd_hist",
        "vol_ratio",
        "close_position",
        "rsi",
        "factor_core4_pct",
        "factor_pvcorr60_pct",
        "factor_broad_top5_pct",
        "factor_antitrix14_pct",
        "candidate_pool_score",
        "short_tp_score_v1",
        "industry_rs_bucket_score",
        "market_stage_bucket_score",
        "market_breadth_bucket_score",
        "vol_quality_bucket_score",
        "circ_mv",
        "parent_l1_mfe_so_far",
        "parent_l1_mae_so_far",
        "parent_l1_return_so_far",
    ]
    for column in sorted(set([*source_columns, *factor_fields])):
        out[column] = numeric(out, column)

    rank_sources = {
        "start_score": "cmp_start_score",
        "bar_range_pct": "cmp_bar_range",
        "real_body_pct": "cmp_real_body",
        "turnover_rate": "cmp_turnover",
        "close_vs_ema20": "cmp_close_vs_ema20",
        "close_vs_ema60": "cmp_close_vs_ema60",
        "ema20_vs_ema60": "cmp_ema20_vs_ema60",
        "plus_di": "cmp_plus_di",
        "macd_hist": "cmp_macd_hist",
        "vol_ratio": "cmp_vol_ratio",
        "close_position": "cmp_close_position",
        "rsi": "cmp_rsi",
        "circ_mv": "cmp_size",
    }
    for source, target in rank_sources.items():
        out[target] = fill_score(pct_rank_by_day(out, source))

    direct_sources = {
        "factor_core4_pct": "cmp_core4",
        "factor_pvcorr60_pct": "cmp_pvcorr60",
        "factor_broad_top5_pct": "cmp_broad_top5",
        "factor_antitrix14_pct": "cmp_antitrix14",
        "candidate_pool_score": "cmp_candidate_pool",
        "short_tp_score_v1": "cmp_short_tp",
        "industry_rs_bucket_score": "cmp_industry",
        "market_breadth_bucket_score": "cmp_breadth",
        "vol_quality_bucket_score": "cmp_vol_quality",
    }
    for source, target in direct_sources.items():
        out[target] = fill_score(out[source])

    factor_score_parts: dict[str, pd.Series] = {}
    for field in factor_fields:
        factor_score_parts[f"sf_{field}_high"] = fill_score(pct_rank_by_day(out, field, ascending=True))
        factor_score_parts[f"sf_{field}_low"] = fill_score(pct_rank_by_day(out, field, ascending=False))
    if factor_score_parts:
        out = pd.concat([out, pd.DataFrame(factor_score_parts, index=out.index)], axis=1)

    out["cmp_start_type"] = (
        out["start_type"]
        .astype(str)
        .map({"panic_reversal": 1.00, "both": 0.85, "exhaustion_start": 0.60})
        .fillna(0.50)
    )
    out["cmp_thrust"] = mean_components(
        out,
        [
            "cmp_bar_range",
            "cmp_real_body",
            "cmp_turnover",
            "cmp_close_vs_ema20",
            "cmp_plus_di",
            "cmp_vol_ratio",
            "cmp_close_position",
        ],
    )
    out["cmp_trend_repair"] = mean_components(
        out,
        [
            "cmp_close_vs_ema20",
            "cmp_close_vs_ema60",
            "cmp_ema20_vs_ema60",
            "cmp_plus_di",
            "cmp_macd_hist",
            "cmp_rsi",
        ],
    )
    out["cmp_factor_safety"] = mean_components(
        out,
        ["cmp_core4", "cmp_pvcorr60", "cmp_broad_top5", "cmp_antitrix14", "cmp_size"],
    )
    out["cmp_context"] = mean_components(out, ["cmp_industry", "cmp_breadth", "cmp_vol_quality"])
    out["cmp_start_quality"] = mean_components(out, ["cmp_start_score", "cmp_start_type"])

    out["l1_rank_score_eod_v1"] = (
        0.34 * out["cmp_thrust"]
        + 0.22 * out["cmp_start_quality"]
        + 0.20 * out["cmp_factor_safety"]
        + 0.14 * out["cmp_context"]
        + 0.10 * out["cmp_short_tp"]
    ).clip(0.0, 1.0)
    out["l1_rank_score_balanced_v1"] = (
        0.35 * out["cmp_core4"]
        + 0.30 * out["cmp_short_tp"]
        + 0.20 * out["cmp_thrust"]
        + 0.15 * out["cmp_context"]
    ).clip(0.0, 1.0)
    out["l1_rank_score_short_focus_v1"] = (
        0.45 * out["cmp_short_tp"]
        + 0.25 * out["cmp_thrust"]
        + 0.20 * out["cmp_core4"]
        + 0.10 * out["cmp_context"]
    ).clip(0.0, 1.0)
    out["l1_rank_score_context_safe_v1"] = (
        0.35 * out["cmp_factor_safety"]
        + 0.25 * out["cmp_context"]
        + 0.20 * out["cmp_start_quality"]
        + 0.20 * out["cmp_vol_quality"]
    ).clip(0.0, 1.0)
    out["l1_rank_score_thrust_v1"] = (
        0.58 * out["cmp_thrust"]
        + 0.22 * out["cmp_trend_repair"]
        + 0.20 * out["cmp_start_quality"]
    ).clip(0.0, 1.0)
    core4_short = fill_score(out["rank_score_core4_short70_30"])
    candidate_pool = fill_score(out["candidate_pool_score"])
    pvcorr_pct = fill_score(out["factor_pvcorr60_pct"])
    out["blend_core4_candidate_80_20"] = (0.80 * core4_short + 0.20 * candidate_pool).clip(0.0, 1.0)
    out["blend_core4_candidate_70_30"] = (0.70 * core4_short + 0.30 * candidate_pool).clip(0.0, 1.0)
    out["blend_core4_candidate_60_40"] = (0.60 * core4_short + 0.40 * candidate_pool).clip(0.0, 1.0)
    out["blend_core4_pvcorrpct_80_20"] = (0.80 * core4_short + 0.20 * pvcorr_pct).clip(0.0, 1.0)
    out["blend_core4_pvcorrpct_70_30"] = (0.70 * core4_short + 0.30 * pvcorr_pct).clip(0.0, 1.0)
    short_tp = fill_score(out["rank_score_short_tp_v1"])
    context_safe = out["l1_rank_score_context_safe_v1"]
    panic = out["start_type"].astype(str).eq("panic_reversal")
    both = out["start_type"].astype(str).eq("both")
    grade_a = out["start_grade"].astype(str).eq("A")
    weak_market = numeric(out, "market_stage_bucket_score") <= 2
    weak_volume = numeric(out, "vol_quality_bucket_score") <= 2
    strong_industry = numeric(out, "industry_rs_bucket_score") >= 4
    out["cond_panic_core4_else_candidate"] = pd.Series(
        np.where(panic, core4_short, candidate_pool),
        index=out.index,
    ).clip(0.0, 1.0)
    out["cond_panic_candidate_else_core4"] = pd.Series(
        np.where(panic, candidate_pool, core4_short),
        index=out.index,
    ).clip(0.0, 1.0)
    out["cond_both_core4_else_short"] = pd.Series(
        np.where(both, core4_short, short_tp),
        index=out.index,
    ).clip(0.0, 1.0)
    out["cond_grade_a_core4_else_short"] = pd.Series(
        np.where(grade_a, core4_short, short_tp),
        index=out.index,
    ).clip(0.0, 1.0)
    out["cond_weak_market_context_else_core4"] = pd.Series(
        np.where(weak_market, context_safe, core4_short),
        index=out.index,
    ).clip(0.0, 1.0)
    out["cond_weak_volume_candidate_else_core4"] = pd.Series(
        np.where(weak_volume, candidate_pool, core4_short),
        index=out.index,
    ).clip(0.0, 1.0)
    out["cond_strong_industry_core4_else_candidate"] = pd.Series(
        np.where(strong_industry, core4_short, candidate_pool),
        index=out.index,
    ).clip(0.0, 1.0)
    return out


def add_offline_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add offline return labels used only for historical scoring."""
    out = frame.copy()
    entry = numeric(out, "suggested_entry_price")
    target = numeric(out, "future_tp5_price")
    hit = bool_series(out, "future_tp5_hit_5d")
    out["tp5_plan_return"] = (
        target * (1.0 - CLOSE_RATE) / (entry * (1.0 + OPEN_RATE)) - 1.0
    ).where(hit, numeric(out, "future_timeout_net_return_5d"))
    for horizon in (1, 3, 5):
        out[f"fixed_{horizon}d_return"] = numeric(out, f"future_open_to_close_net_return_{horizon}d")
    return out


def load_industry_mapping(panel_path: str, years: list[int]) -> pd.DataFrame:
    """Load SW level-1 industry names from the local research panel."""
    frames: list[pd.DataFrame] = []
    base = Path(panel_path)
    for year in sorted(set(years)):
        path = base / f"year={year}" / "data_0.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["trade_date", "vt_symbol", "sw_l1_name"])
        frame = frame.rename(columns={"trade_date": "signal_date"})
        frame["signal_date"] = normalize_date(frame["signal_date"])
        frames.append(frame.dropna(subset=["signal_date", "vt_symbol"]))
    if not frames:
        return pd.DataFrame(columns=["signal_date", "vt_symbol", "sw_l1_name"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["signal_date", "vt_symbol"])


def add_industry(pool: pd.DataFrame, panel_path: str) -> pd.DataFrame:
    """Attach SW level-1 industry names for industry-cap selection."""
    out = pool.copy()
    if "sw_l1_name" in out.columns and out["sw_l1_name"].notna().any():
        out["sw_l1_name"] = out["sw_l1_name"].fillna("UNKNOWN")
        return out
    years = sorted(pd.to_datetime(out["signal_date"]).dt.year.astype(int).unique())
    industry = load_industry_mapping(panel_path, years)
    if industry.empty:
        out["sw_l1_name"] = "UNKNOWN"
        return out
    out = out.merge(industry, on=["signal_date", "vt_symbol"], how="left")
    out["sw_l1_name"] = out["sw_l1_name"].fillna("UNKNOWN")
    return out


def load_rank_pool(
    snapshot_path: str,
    panel_path: str,
    threshold: float,
    *,
    extra_factor_scan: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare the fixed L1 pool plus day filter."""
    snapshot = add_buy_point_columns(pd.read_parquet(snapshot_path))
    snapshot["signal_date"] = normalize_date(snapshot["signal_date"])
    assert_signal_close_columns(
        [
            "signal_date",
            "buy_point_type",
            "has_valid_bar",
            "is_list_life_valid",
            "is_st",
            "is_new_stock",
            DAY_FILTER_COLUMN,
        ],
        stage="signal_close_pool_and_day_filter",
    )
    mask = (
        snapshot["buy_point_type"].eq(BUY_POINT_TYPE)
        & bool_series(snapshot, "has_valid_bar", default=True)
        & bool_series(snapshot, "is_list_life_valid", default=True)
        & ~bool_series(snapshot, "is_st")
        & ~bool_series(snapshot, "is_new_stock")
    )
    pool = snapshot[mask].copy()
    extra_factor_fields = EXTRA_FACTOR_SCAN_FIELDS if extra_factor_scan else []
    if extra_factor_scan:
        pool = add_extra_signal_close_factors(pool, panel_path)
    pool = add_industry(pool, panel_path)
    pool = add_l1_rank_score_candidates(pool, extra_factor_fields=extra_factor_fields)
    daily = (
        pool.groupby("signal_date", sort=True)
        .agg(
            signal_close_l1_events=("vt_symbol", "count"),
            signal_close_l1_symbols=("vt_symbol", "nunique"),
            signal_close_core4_mean=(DAY_FILTER_COLUMN, "mean"),
            signal_close_core4_top=(DAY_FILTER_COLUMN, "max"),
        )
        .reset_index()
    )
    daily["signal_year"] = pd.to_datetime(daily["signal_date"]).dt.year.astype(int)
    daily["signal_close_day_filter_threshold"] = threshold
    daily["signal_close_day_filter_pass"] = daily["signal_close_core4_mean"] >= threshold
    daily["day_quality"] = np.where(daily["signal_close_core4_mean"] >= 0.54, "strong_candidate", "candidate")
    pool = pool.merge(
        daily[
            [
                "signal_date",
                "signal_year",
                "signal_close_core4_mean",
                "signal_close_day_filter_threshold",
                "signal_close_day_filter_pass",
                "day_quality",
            ]
        ],
        on="signal_date",
        how="left",
    )
    return add_offline_returns(pool), daily


def add_next_open_execution_fields(
    pool: pd.DataFrame,
    *,
    gap_min: float,
    gap_max: float,
    min_circ_mv: float,
    min_turnover_rate: float,
) -> pd.DataFrame:
    """Add fixed T+1 open executable fields."""
    out = pool.copy()
    gap = numeric(out, "future_entry_gap_pct")
    out["next_open_gap_pct"] = gap
    out["next_open_has_entry"] = bool_series(out, "future_has_entry")
    out["next_open_has_valid_bar"] = bool_series(out, "next_has_valid_bar")
    out["next_open_limit_up"] = bool_series(out, "next_is_limit_up")
    out["next_open_limit_down"] = bool_series(out, "next_is_limit_down")
    out["next_open_gap_ok"] = (gap >= gap_min) & (gap < gap_max)
    out["next_open_liquidity_ok"] = (
        ~out["vt_symbol"].astype(str).str.endswith(".BSE")
        & (numeric(out, "circ_mv") >= min_circ_mv)
        & (numeric(out, "turnover_rate") >= min_turnover_rate)
    )
    out["next_open_executable"] = (
        out["next_open_has_entry"]
        & out["next_open_has_valid_bar"]
        & ~out["next_open_limit_up"]
        & ~out["next_open_limit_down"]
        & out["next_open_gap_ok"]
        & out["next_open_liquidity_ok"]
    )
    return out


def resolve_score_map(frame: pd.DataFrame, score_names: list[str]) -> dict[str, str]:
    """Resolve score aliases to existing columns."""
    score_map: dict[str, str] = {}
    aliases = {**BASE_SCORE_COLUMNS, **COMPUTED_SCORE_COLUMNS}
    for name in score_names:
        column = aliases.get(name, name)
        if column not in frame.columns:
            print(f"skip missing rank score: {name} -> {column}")
            continue
        assert_signal_close_columns([column], stage=f"rank_score:{name}")
        score_map[name] = column
    if not score_map:
        raise ValueError("no rank score columns are available")
    return score_map


def sort_daily_candidates(group: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Sort one signal date by score and stable fallback columns."""
    fallback = [
        column
        for column in (
            "start_score",
            "rank_score_short_tp_v1",
            "rank_score_core4_short70_30",
            "candidate_pool_score",
        )
        if column in group.columns and column != score_column
    ]
    ranked = (
        group.dropna(subset=[score_column])
        .sort_values([score_column, *fallback, "vt_symbol"], ascending=[False] * (1 + len(fallback)) + [True])
        .copy()
    )
    ranked["signal_close_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def select_with_industry_cap(frame: pd.DataFrame, *, top_k: int, industry_cap: float | None) -> pd.DataFrame:
    """Select top rows with an optional industry concentration cap."""
    if frame.empty:
        return frame.head(0).copy()
    if industry_cap is None:
        return frame.head(top_k).copy()
    max_per_industry = max(1, int(math.ceil(top_k * industry_cap)))
    selected_indexes: list[int] = []
    industry_counts: dict[str, int] = {}
    for index, row in frame.iterrows():
        industry = str(row.get("sw_l1_name", "UNKNOWN") or "UNKNOWN")
        if industry_counts.get(industry, 0) >= max_per_industry:
            continue
        selected_indexes.append(int(index))
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected_indexes) >= top_k:
            break
    return frame.loc[selected_indexes].copy()


def build_rank_selections(
    pool: pd.DataFrame,
    *,
    score_map: dict[str, str],
    top_n: int,
    top_k_values: list[int],
    industry_cap: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build selected event rows and daily base rows for every score."""
    event_parts: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    candidate_days = pool[pool["signal_close_day_filter_pass"]].copy()
    grouped = list(candidate_days.groupby("signal_date", sort=True))

    for score_name, score_column in score_map.items():
        for signal_date, group in grouped:
            ranked = sort_daily_candidates(group, score_column)
            if ranked.empty:
                continue
            rank_pool = ranked[ranked["signal_close_rank"] <= top_n].copy()
            executable = rank_pool[rank_pool["next_open_executable"]].sort_values("signal_close_rank").copy()
            signal_date = pd.Timestamp(signal_date).normalize()
            base = {
                "rank_score": score_name,
                "rank_score_column": score_column,
                "signal_date": signal_date,
                "signal_year": int(signal_date.year),
                "signal_close_top_n": int(top_n),
                "signal_close_l1_candidates": int(len(ranked)),
                "signal_close_rank_pool_events": int(len(rank_pool)),
                "next_open_executable_events": int(len(executable)),
                "core4_mean": float(group["signal_close_core4_mean"].iloc[0]),
                "day_quality": str(group["day_quality"].iloc[0]),
            }
            for top_k in top_k_values:
                selected = select_with_industry_cap(executable, top_k=top_k, industry_cap=industry_cap)
                selected = selected.copy()
                selected["next_open_execution_rank"] = np.arange(1, len(selected) + 1)
                selected["rank_score"] = score_name
                selected["rank_score_column"] = score_column
                selected["rank_value"] = selected[score_column]
                selected["signal_close_top_n"] = top_n
                selected["top_k"] = top_k
                selected["industry_cap"] = industry_cap if industry_cap is not None else math.nan
                if not selected.empty:
                    event_parts.append(selected)
                daily_rows.append(
                    {
                        **base,
                        "top_k": int(top_k),
                        "industry_cap": industry_cap if industry_cap is not None else math.nan,
                        "selected_events": int(len(selected)),
                        "selected_symbols": ",".join(selected["vt_symbol"].astype(str).tolist()),
                    }
                )

    events = pd.concat(event_parts, ignore_index=True) if event_parts else pool.iloc[0:0].copy()
    daily_base = pd.DataFrame(daily_rows)
    return events, daily_base


def build_daily_returns(events: pd.DataFrame, daily_base: pd.DataFrame) -> pd.DataFrame:
    """Build one return row per score/date/topK/exit rule."""
    if daily_base.empty:
        return pd.DataFrame()
    group_cols = ["rank_score", "signal_close_top_n", "top_k", "signal_date"]
    exit_frames: list[pd.DataFrame] = []
    for exit_rule, return_column in RETURN_COLUMNS.items():
        if events.empty:
            trade_daily = pd.DataFrame(columns=[*group_cols, "daily_return", "target_hit_rate", "avg_holding_bars"])
        else:
            trade_daily = (
                events.dropna(subset=[return_column])
                .groupby(group_cols, sort=True)
                .agg(
                    daily_return=(return_column, "mean"),
                    trade_events=("vt_symbol", "count"),
                    target_hit_rate=("future_tp5_hit_5d", "mean"),
                    avg_holding_bars=("future_days_to_tp5", "mean"),
                    selected_symbols=("vt_symbol", lambda values: ",".join(values.astype(str).tolist())),
                )
                .reset_index()
            )
        frame = daily_base.merge(trade_daily, on=group_cols, how="left", suffixes=("", "_trade"))
        frame["exit_rule"] = exit_rule
        frame["daily_return"] = numeric(frame, "daily_return").fillna(0.0)
        frame["trade_events"] = numeric(frame, "trade_events").fillna(0).astype(int)
        frame["had_trade"] = frame["trade_events"] > 0
        frame["target_hit_rate"] = numeric(frame, "target_hit_rate")
        frame["avg_holding_bars"] = numeric(frame, "avg_holding_bars")
        frame["selected_symbols"] = frame["selected_symbols_trade"].fillna(frame["selected_symbols"])
        frame = frame.drop(columns=["selected_symbols_trade"], errors="ignore")
        exit_frames.append(frame)
    return pd.concat(exit_frames, ignore_index=True)


def summarize_returns(returns: pd.Series) -> dict[str, float | int]:
    """Summarize a daily return series."""
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {
            "days": 0,
            "avg_daily_return": math.nan,
            "median_daily_return": math.nan,
            "p10_daily_return": math.nan,
            "p25_daily_return": math.nan,
            "positive_day_rate": math.nan,
            "total_compounded_return": math.nan,
            "max_drawdown": math.nan,
        }
    return {
        "days": int(len(clean)),
        "avg_daily_return": float(clean.mean()),
        "median_daily_return": float(clean.median()),
        "p10_daily_return": float(clean.quantile(0.10)),
        "p25_daily_return": float(clean.quantile(0.25)),
        "positive_day_rate": float((clean > 0).mean()),
        "total_compounded_return": float((1.0 + clean).prod() - 1.0),
        "max_drawdown": drawdown(clean),
    }


def summarize_daily(daily: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Summarize daily rows by the requested keys."""
    rows: list[dict[str, Any]] = []
    for keys, group in daily.groupby(group_cols, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, values, strict=False))
        trade_returns = numeric(group[group["had_trade"]], "daily_return").dropna()
        row["candidate_days"] = int(group["signal_date"].nunique())
        row["trade_days"] = int(group.loc[group["had_trade"], "signal_date"].nunique())
        row["trade_day_rate"] = row["trade_days"] / row["candidate_days"] if row["candidate_days"] else math.nan
        row["avg_selected_events"] = float(numeric(group, "selected_events").mean())
        row["avg_trade_events"] = float(numeric(group, "trade_events").mean())
        row["avg_signal_close_l1_candidates"] = float(numeric(group, "signal_close_l1_candidates").mean())
        row["avg_next_open_executable_events"] = float(numeric(group, "next_open_executable_events").mean())
        row.update(summarize_returns(group["daily_return"]))
        row["avg_trade_day_return"] = float(trade_returns.mean()) if not trade_returns.empty else math.nan
        row["p10_trade_day_return"] = float(trade_returns.quantile(0.10)) if not trade_returns.empty else math.nan
        row["target_hit_rate"] = float(numeric(group, "target_hit_rate").mean())
        row["avg_holding_bars"] = float(numeric(group, "avg_holding_bars").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_score_ic(pool: pd.DataFrame, score_map: dict[str, str]) -> pd.DataFrame:
    """Measure cross-sectional score correlation on executable candidate events."""
    executable = pool[pool["signal_close_day_filter_pass"] & pool["next_open_executable"]].copy()
    rows: list[dict[str, Any]] = []
    for score_name, score_column in score_map.items():
        for return_name, return_column in RETURN_COLUMNS.items():
            pair = executable[[score_column, return_column, "signal_date"]].dropna()
            event_ic = (
                pair[[score_column, return_column]].corr(method="spearman").iloc[0, 1]
                if len(pair) >= 3
                else math.nan
            )
            daily_ics: list[float] = []
            for _, group in pair.groupby("signal_date", sort=True):
                if len(group) < 3 or group[score_column].nunique() <= 1 or group[return_column].nunique() <= 1:
                    continue
                daily_ics.append(float(group[[score_column, return_column]].corr(method="spearman").iloc[0, 1]))
            rows.append(
                {
                    "rank_score": score_name,
                    "rank_score_column": score_column,
                    "return_name": return_name,
                    "events": int(len(pair)),
                    "daily_ic_days": int(len(daily_ics)),
                    "event_spearman": float(event_ic),
                    "daily_ic_mean": float(np.nanmean(daily_ics)) if daily_ics else math.nan,
                    "daily_ic_median": float(np.nanmedian(daily_ics)) if daily_ics else math.nan,
                    "daily_ic_positive_rate": float((np.asarray(daily_ics) > 0).mean()) if daily_ics else math.nan,
                }
            )
    return pd.DataFrame(rows)


def write_markdown_report(path: Path, overall: pd.DataFrame, score_ic: pd.DataFrame, *, top_k: int) -> None:
    """Write a compact Markdown report."""
    focus = overall[(overall["top_k"].eq(top_k)) & (overall["exit_rule"].eq("tp5_plan"))].copy()
    focus = focus.sort_values(["avg_daily_return", "max_drawdown"], ascending=[False, False])
    lines = [
        "# L1 Rank Score Study",
        "",
        f"- Fixed signal: `{BUY_POINT_TYPE}`",
        f"- Day filter: `{DAY_FILTER_COLUMN} mean >= {DAY_FILTER_THRESHOLD}`",
        f"- Signal-close pool: `top{SIGNAL_CLOSE_TOP_N}`",
        f"- Focus selection: `next-open top{top_k}`",
        "",
        "## TP5 Plan Ranking",
        "",
    ]
    if focus.empty:
        lines.append("No results.")
    else:
        lines.append("| Rank Score | Days | Trade Days | Avg | P10 | MDD | TP5 | Avg Selected |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in focus.head(12).to_dict("records"):
            lines.append(
                "| "
                f"`{row['rank_score']}` | "
                f"{int(row['candidate_days'])} | "
                f"{int(row['trade_days'])} | "
                f"{row['avg_daily_return']:.2%} | "
                f"{row['p10_daily_return']:.2%} | "
                f"{row['max_drawdown']:.2%} | "
                f"{row['target_hit_rate']:.2%} | "
                f"{row['avg_selected_events']:.2f} |"
            )
    lines.extend(["", "## Score IC", ""])
    ic_focus = score_ic[score_ic["return_name"].eq("tp5_plan")].sort_values("daily_ic_mean", ascending=False)
    if ic_focus.empty:
        lines.append("No IC results.")
    else:
        lines.append("| Rank Score | Events | Event IC | Daily IC Mean | Daily IC Median | Positive IC Days |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in ic_focus.head(12).to_dict("records"):
            lines.append(
                "| "
                f"`{row['rank_score']}` | "
                f"{int(row['events'])} | "
                f"{row['event_spearman']:.4f} | "
                f"{row['daily_ic_mean']:.4f} | "
                f"{row['daily_ic_median']:.4f} | "
                f"{row['daily_ic_positive_rate']:.2%} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the rank-score study."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_k_values = parse_int_list(args.top_k)
    industry_cap = parse_industry_cap(args.industry_cap)
    rank_scores = parse_rank_scores(args.rank_scores)
    if args.extra_factor_scan:
        rank_scores = [*rank_scores, *EXTRA_FACTOR_SCORE_NAMES]

    pool, daily_filter = load_rank_pool(
        args.snapshot_path,
        args.panel_path,
        args.day_filter_threshold,
        extra_factor_scan=args.extra_factor_scan,
    )
    pool = add_next_open_execution_fields(
        pool,
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        min_circ_mv=args.min_circ_mv,
        min_turnover_rate=args.min_turnover_rate,
    )
    score_map = resolve_score_map(pool, rank_scores)
    events, daily_base = build_rank_selections(
        pool,
        score_map=score_map,
        top_n=args.signal_close_top_n,
        top_k_values=top_k_values,
        industry_cap=industry_cap,
    )
    daily = build_daily_returns(events, daily_base)
    overall = summarize_daily(daily, ["rank_score", "signal_close_top_n", "top_k", "exit_rule"])
    by_year = summarize_daily(daily, ["rank_score", "signal_close_top_n", "top_k", "exit_rule", "signal_year"])
    score_ic = build_score_ic(pool, score_map)

    daily_filter.to_csv(output_dir / "l1_rank_score_day_filter.csv", index=False)
    if args.write_event_selection:
        events.to_csv(output_dir / "l1_rank_score_event_selection.csv", index=False)
    else:
        events.head(1000).to_csv(output_dir / "l1_rank_score_event_selection_sample.csv", index=False)
    daily.to_csv(output_dir / "l1_rank_score_daily.csv", index=False)
    overall.to_csv(output_dir / "l1_rank_score_summary.csv", index=False)
    by_year.to_csv(output_dir / "l1_rank_score_by_year.csv", index=False)
    score_ic.to_csv(output_dir / "l1_rank_score_ic.csv", index=False)
    write_markdown_report(output_dir / "l1_rank_score_report.md", overall, score_ic, top_k=2)

    focus = overall[(overall["top_k"].eq(2)) & (overall["exit_rule"].eq("tp5_plan"))]
    focus = focus.sort_values(["avg_daily_return", "max_drawdown"], ascending=[False, False])
    print(f"candidate_days={int(daily_filter['signal_close_day_filter_pass'].sum())}")
    print(f"rank_scores={','.join(score_map)}")
    print(f"wrote {output_dir}")
    print(focus.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
