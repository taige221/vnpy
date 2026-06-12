"""Backtest L1 execution with a clean signal-close to next-open timeline.

Decision stages:

- signal_close_day_filter: T close day-level gate; no ``future_*`` or
  ``next_*`` fields are allowed.
- signal_close_rank_pool: T close stock-level rank pool; no ``future_*`` or
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
DEFAULT_OUTPUT_DIR = (
    "examples/alpha_research/tradingview/reports/a5_signal_snapshot/"
    "clean_timeline_backtest"
)
BUY_POINT_TYPE = BUY_POINT_TYPE_L1
RANK_COLUMN = "rank_score_core4_short70_30"
DAY_FILTER_THRESHOLD = 0.52
SIGNAL_CLOSE_TOP_N = 20
NEXT_OPEN_TOP_K = 2
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
    "next_open_executable",
    "next_open_selected",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run clean-timeline L1 backtest")
    parser.add_argument("--snapshot-path", default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--day-filter-threshold", type=float, default=DAY_FILTER_THRESHOLD)
    parser.add_argument("--signal-close-top-n", type=int, default=SIGNAL_CLOSE_TOP_N)
    parser.add_argument("--next-open-top-k", type=int, default=NEXT_OPEN_TOP_K)
    parser.add_argument("--gap-min", type=float, default=GAP_MIN)
    parser.add_argument("--gap-max", type=float, default=GAP_MAX)
    parser.add_argument("--min-circ-mv", type=float, default=MIN_CIRC_MV)
    parser.add_argument("--min-turnover-rate", type=float, default=MIN_TURNOVER_RATE)
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


def net_target_return(entry: pd.Series, target: pd.Series) -> pd.Series:
    """Return net target exit return from next-open entry."""
    return target * (1.0 - CLOSE_RATE) / (entry * (1.0 + OPEN_RATE)) - 1.0


def signal_close_l1_pool(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Return T-close-known L1 candidates."""
    used_columns = [
        "signal_date",
        "buy_point_type",
        "has_valid_bar",
        "is_list_life_valid",
        "is_st",
        "is_new_stock",
    ]
    assert_signal_close_columns(used_columns, stage="signal_close_l1_pool")
    mask = (
        snapshot["buy_point_type"].eq(BUY_POINT_TYPE)
        & bool_series(snapshot, "has_valid_bar", default=True)
        & bool_series(snapshot, "is_list_life_valid", default=True)
        & ~bool_series(snapshot, "is_st")
        & ~bool_series(snapshot, "is_new_stock")
    )
    return snapshot[mask].copy()


def add_signal_close_day_filter(
    pool: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add T-close day-filter fields."""
    assert_signal_close_columns([RANK_COLUMN], stage="signal_close_day_filter")
    daily = (
        pool.groupby("signal_date", sort=True)
        .agg(
            signal_close_l1_events=("vt_symbol", "count"),
            signal_close_l1_symbols=("vt_symbol", "nunique"),
            signal_close_core4_mean=(RANK_COLUMN, "mean"),
            signal_close_core4_top=(RANK_COLUMN, "max"),
        )
        .reset_index()
    )
    daily["signal_close_day_filter_name"] = "signal_close_core4_mean_ge_threshold"
    daily["signal_close_day_filter_threshold"] = threshold
    daily["signal_close_day_filter_pass"] = daily["signal_close_core4_mean"] >= threshold
    out = pool.merge(
        daily[
            [
                "signal_date",
                "signal_close_core4_mean",
                "signal_close_day_filter_name",
                "signal_close_day_filter_threshold",
                "signal_close_day_filter_pass",
            ]
        ],
        on="signal_date",
        how="left",
    )
    return out, daily


def add_signal_close_rank_pool(pool: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Add T-close rank pool fields."""
    rank_columns = [RANK_COLUMN, "start_score", "rank_score_short_tp_v1", "candidate_pool_score", "vt_symbol"]
    assert_signal_close_columns(rank_columns, stage="signal_close_rank_pool")
    ranked_parts: list[pd.DataFrame] = []
    for _, group in pool.groupby("signal_date", sort=True):
        ranked = group.sort_values(
            [RANK_COLUMN, "start_score", "rank_score_short_tp_v1", "candidate_pool_score", "vt_symbol"],
            ascending=[False, False, False, False, True],
        ).copy()
        ranked["signal_close_rank"] = np.arange(1, len(ranked) + 1)
        ranked_parts.append(ranked)
    out = pd.concat(ranked_parts, ignore_index=True) if ranked_parts else pool.iloc[0:0].copy()
    out["signal_close_rank_pool_label"] = f"signal_close_top{top_n}"
    out["signal_close_rank_pool_member"] = out["signal_close_rank"] <= top_n
    return out


def add_next_open_execution_fields(
    pool: pd.DataFrame,
    *,
    gap_min: float,
    gap_max: float,
    min_circ_mv: float,
    min_turnover_rate: float,
) -> pd.DataFrame:
    """Add T+1 open executable fields."""
    out = pool.copy()
    gap = numeric(out, "future_entry_gap_pct")
    out["next_open_gap_pct"] = gap
    out["next_open_has_entry"] = bool_series(out, "future_has_entry")
    out["next_open_has_valid_bar"] = bool_series(out, "next_has_valid_bar")
    out["next_open_limit_up"] = bool_series(out, "next_is_limit_up")
    out["next_open_limit_down"] = bool_series(out, "next_is_limit_down")
    out["offline_has_full_5d"] = bool_series(out, "future_has_full_5d")
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


def select_next_open_topk(pool: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    """Select T+1 open top-K from the signal-close rank pool."""
    selected_parts: list[pd.DataFrame] = []
    for _, group in pool.groupby("signal_date", sort=True):
        selected = group[
            group["signal_close_day_filter_pass"]
            & group["signal_close_rank_pool_member"]
            & group["next_open_executable"]
        ].sort_values("signal_close_rank").head(top_k)
        if selected.empty:
            continue
        selected = selected.copy()
        selected["next_open_execution_rank"] = np.arange(1, len(selected) + 1)
        selected["next_open_selection_label"] = (
            f"next_open_select_top{top_k}_from_signal_close_top{int(group['signal_close_rank_pool_member'].sum())}"
        )
        selected["next_open_selected"] = True
        selected_parts.append(selected)
    return pd.concat(selected_parts, ignore_index=True) if selected_parts else pool.iloc[0:0].copy()


def add_offline_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add offline outcome returns from future labels."""
    out = frame.copy()
    entry = numeric(out, "suggested_entry_price")
    target = numeric(out, "future_tp5_price")
    hit = bool_series(out, "future_tp5_hit_5d")
    out["tp5_plan_return"] = net_target_return(entry, target).where(
        hit,
        numeric(out, "future_timeout_net_return_5d"),
    )
    for horizon in (1, 3, 5):
        out[f"fixed_{horizon}d_return"] = numeric(
            out,
            f"future_open_to_close_net_return_{horizon}d",
        )
    return out


def daily_equal_return(frame: pd.DataFrame, return_column: str) -> pd.DataFrame:
    """Return one equal-weight row per signal day."""
    if frame.empty:
        return pd.DataFrame(columns=["signal_date", "daily_return", "selected_events", "tp5_rate"])
    return (
        frame.groupby("signal_date", sort=True)
        .apply(
            lambda group: pd.Series(
                {
                    "daily_return": numeric(group, return_column).mean(),
                    "selected_events": int(len(group)),
                    "tp5_rate": bool_series(group, "future_tp5_hit_5d").mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def summarize_daily(daily: pd.DataFrame) -> dict[str, float | int]:
    """Summarize a daily return frame."""
    returns = pd.to_numeric(daily["daily_return"], errors="coerce").dropna()
    if returns.empty:
        return {
            "days": 0,
            "avg_daily_return": math.nan,
            "p10_daily_return": math.nan,
            "median_daily_return": math.nan,
            "positive_day_rate": math.nan,
            "tp5_rate": math.nan,
            "max_drawdown": math.nan,
            "compounded_return": math.nan,
            "avg_selected_events": math.nan,
        }
    return {
        "days": int(len(returns)),
        "avg_daily_return": float(returns.mean()),
        "p10_daily_return": float(returns.quantile(0.10)),
        "median_daily_return": float(returns.median()),
        "positive_day_rate": float((returns > 0).mean()),
        "tp5_rate": float(pd.to_numeric(daily["tp5_rate"], errors="coerce").mean()),
        "max_drawdown": drawdown(returns),
        "compounded_return": float((1.0 + returns).prod() - 1.0),
        "avg_selected_events": float(pd.to_numeric(daily["selected_events"], errors="coerce").mean()),
    }


def build_summary(full_pool: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Build comparison summary rows."""
    variants = [
        ("all_next_open_executable_l1", full_pool[full_pool["next_open_executable"]]),
        (
            "signal_close_day_filter_only",
            full_pool[full_pool["signal_close_day_filter_pass"] & full_pool["next_open_executable"]],
        ),
        ("signal_close_top20_next_open_top2", selected),
    ]
    rows: list[dict[str, Any]] = []
    for variant, frame in variants:
        frame = add_offline_returns(frame)
        for exit_rule, return_column in [
            ("tp5_plan", "tp5_plan_return"),
            ("fixed_1d", "fixed_1d_return"),
            ("fixed_3d", "fixed_3d_return"),
            ("fixed_5d", "fixed_5d_return"),
        ]:
            daily = daily_equal_return(frame, return_column)
            rows.append(
                {
                    "variant": variant,
                    "exit_rule": exit_rule,
                    "events": int(len(frame)),
                    "evaluable_events": int(pd.to_numeric(frame[return_column], errors="coerce").notna().sum())
                    if return_column in frame.columns
                    else 0,
                    **summarize_daily(daily),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the clean timeline backtest."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = add_buy_point_columns(pd.read_parquet(args.snapshot_path))
    snapshot["signal_date"] = normalize_date(snapshot["signal_date"])

    signal_pool = signal_close_l1_pool(snapshot)
    signal_pool, daily_filter = add_signal_close_day_filter(
        signal_pool,
        threshold=args.day_filter_threshold,
    )
    signal_pool = add_signal_close_rank_pool(signal_pool, top_n=args.signal_close_top_n)
    execution_pool = add_next_open_execution_fields(
        signal_pool,
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        min_circ_mv=args.min_circ_mv,
        min_turnover_rate=args.min_turnover_rate,
    )
    execution_pool["next_open_selected"] = False
    selected = select_next_open_topk(execution_pool, top_k=args.next_open_top_k)
    selected = add_offline_returns(selected)
    summary = build_summary(execution_pool, selected)

    daily_filter_path = output_dir / "l1_signal_close_day_filter.csv"
    execution_pool_path = output_dir / "l1_clean_timeline_execution_pool.parquet"
    selected_path = output_dir / "l1_clean_timeline_next_open_selected.csv"
    summary_path = output_dir / "l1_clean_timeline_summary.csv"
    daily_filter.to_csv(daily_filter_path, index=False)
    execution_pool.to_parquet(execution_pool_path, index=False)
    selected.to_csv(selected_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"signal_close_days={len(daily_filter)}")
    print(f"signal_close_day_filter_pass_days={int(daily_filter['signal_close_day_filter_pass'].sum())}")
    print(f"next_open_selected_events={len(selected)}")
    print(f"wrote {daily_filter_path}")
    print(f"wrote {execution_pool_path}")
    print(f"wrote {selected_path}")
    print(f"wrote {summary_path}")
    compact = summary[summary["exit_rule"].eq("tp5_plan")].copy()
    for column in [
        "avg_daily_return",
        "p10_daily_return",
        "median_daily_return",
        "positive_day_rate",
        "tp5_rate",
        "max_drawdown",
        "compounded_return",
    ]:
        compact[column] = compact[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value) * 100:.2f}%"
        )
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
