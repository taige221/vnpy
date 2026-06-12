"""Study a near-start pre-L1 buy point.

This research script tests whether a T-close "almost L1" state can be used as a
T+1 open early-entry candidate before the formal L1 reversal-start signal exists.

The decision columns are built only from bars up to the signal date. The L1
snapshot is used only as an offline label: did the same symbol become L1 on the
next trading day or within the next few trading days?
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
TV_ALPHA = REPO_ROOT / "examples" / "alpha_research" / "tradingview" / "vnpy_alpha"
sys.path.insert(0, str(TV_ALPHA))
sys.path.insert(0, str(REPO_ROOT))

from run_trend_rsi_event_study import TrendRsiConfig, compute_base_columns, load_daily  # noqa: E402
from scripts.alpha_build.common.symbols import add_symbol_selection_args, select_paths  # noqa: E402
from scripts.alpha_research.tradingview.signal_core.buy_points import BUY_POINT_TYPE_L1  # noqa: E402


DEFAULT_SNAPSHOT_PATH = (
    "examples/alpha_research/tradingview/reports/a5_signal_snapshot/"
    "a5_signal_snapshot.parquet"
)
DEFAULT_OUTPUT_DIR = "scripts/alpha_research/tradingview/reports/pre_l1_near_start_study"
OPEN_RATE = 0.0005
CLOSE_RATE = 0.0015
TP_LEVELS: tuple[float, ...] = (0.03, 0.05)
FIXED_HORIZONS: tuple[int, ...] = (1, 3, 5)
PATH_HORIZONS: tuple[int, ...] = (0, 1, 3)
CONFIRM_ENTRY_VARIANTS: tuple[str, ...] = (
    "confirm_open",
    "confirm_trigger",
    "confirm_close",
    "l1_next_open",
)
SUMMARY_SCOPES: tuple[str, ...] = (
    "all",
    "open_executable",
    "main_gap_-3_+3",
    "tomorrow_l1",
    "not_tomorrow_l1",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run PRE_L1_near_start research")
    parser.add_argument("--lab-path", default="lab/a_share_research")
    parser.add_argument("--snapshot-path", default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2099-12-31")
    parser.add_argument("--min-bars", type=int, default=400)
    parser.add_argument("--limit", type=int)
    add_symbol_selection_args(parser)
    parser.add_argument("--cooldown-bars", type=int, default=3)
    parser.add_argument("--open-rate", type=float, default=OPEN_RATE)
    parser.add_argument("--close-rate", type=float, default=CLOSE_RATE)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--write-events", action="store_true")
    parser.add_argument(
        "--b1-only",
        action="store_true",
        help="Only keep strong open-break-high candidates for a faster B1 study.",
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    """Resolve a path from the repository root unless already absolute."""
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def normalize_date(values: Any) -> pd.Series:
    """Normalize date-like values."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def price_return(target: float, entry: float) -> float:
    """Return simple price return."""
    if not math.isfinite(target) or not math.isfinite(entry) or entry <= 0:
        return math.nan
    return target / entry - 1.0


def net_return(entry: float, target: float, open_rate: float, close_rate: float) -> float:
    """Return net return after entry and exit costs."""
    if not math.isfinite(target) or not math.isfinite(entry) or entry <= 0:
        return math.nan
    return target * (1.0 - close_rate) / (entry * (1.0 + open_rate)) - 1.0


def drawdown(returns: pd.Series) -> float:
    """Return maximum drawdown for a compounded return series."""
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return math.nan
    equity = (1.0 + values).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def pct(value: float) -> str:
    """Format a return-like value."""
    if not math.isfinite(value):
        return "nan"
    return f"{value * 100:.2f}%"


def load_l1_label_sets(snapshot_path: Path) -> dict[str, set[pd.Timestamp]]:
    """Load L1 signal dates by symbol for offline labeling."""
    snapshot = pd.read_parquet(snapshot_path, columns=["signal_date", "vt_symbol", "buy_point_type"])
    snapshot["signal_date"] = normalize_date(snapshot["signal_date"])
    snapshot = snapshot[snapshot["buy_point_type"].eq(BUY_POINT_TYPE_L1)].copy()
    grouped: dict[str, set[pd.Timestamp]] = {}
    for vt_symbol, group in snapshot.groupby("vt_symbol", sort=False):
        grouped[str(vt_symbol)] = {pd.Timestamp(value).normalize() for value in group["signal_date"]}
    return grouped


def load_entry_panel(
    lab_path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load next-open tradability fields from the research panel."""
    panel_path = lab_path / "panel" / "research_panel_daily"
    columns = [
        "trade_date",
        "vt_symbol",
        "name",
        "list_date",
        "open",
        "up_limit",
        "down_limit",
        "turnover_rate",
        "circ_mv",
        "has_valid_bar",
        "is_list_life_valid",
    ]
    frames: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        file_path = panel_path / f"year={year}" / "data_0.parquet"
        if not file_path.exists():
            continue
        frame = pd.read_parquet(file_path, columns=columns)
        frame["entry_date"] = normalize_date(frame["trade_date"])
        frame = frame[(frame["entry_date"] >= start) & (frame["entry_date"] <= end)].copy()
        frames.append(frame.drop(columns=["trade_date"]))
    if not frames:
        return pd.DataFrame(columns=["entry_date", "vt_symbol"])
    out = pd.concat(frames, ignore_index=True)
    out["list_date"] = pd.to_datetime(out["list_date"], errors="coerce")
    out["entry_age_days"] = (out["entry_date"] - out["list_date"]).dt.days
    out["entry_is_st"] = out["name"].fillna("").astype(str).str.contains("ST|退")
    return out.drop_duplicates(["entry_date", "vt_symbol"])


def add_execution_fields(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Attach next-open tradability metadata."""
    if events.empty:
        return events.copy()
    out = events.merge(panel, on=["entry_date", "vt_symbol"], how="left", suffixes=("", "_panel"))
    out["entry_open_panel"] = pd.to_numeric(out.get("open"), errors="coerce")
    out["entry_open"] = pd.to_numeric(out["entry_open"], errors="coerce").fillna(out["entry_open_panel"])
    out["entry_gap_pct"] = out["entry_open"] / pd.to_numeric(out["signal_close"], errors="coerce") - 1.0
    for column in ("has_valid_bar", "is_list_life_valid"):
        out[column] = out[column].astype("boolean").fillna(False).astype(bool)
    out["entry_is_st"] = out["entry_is_st"].fillna(True).astype(bool)
    out["entry_age_days"] = pd.to_numeric(out["entry_age_days"], errors="coerce")
    out["entry_limit_up"] = (
        pd.to_numeric(out["up_limit"], errors="coerce").notna()
        & (out["entry_open"] >= pd.to_numeric(out["up_limit"], errors="coerce") * 0.999)
    )
    out["entry_limit_down"] = (
        pd.to_numeric(out["down_limit"], errors="coerce").notna()
        & (out["entry_open"] <= pd.to_numeric(out["down_limit"], errors="coerce") * 1.001)
    )
    out["liquidity_ok"] = (
        ~out["vt_symbol"].astype(str).str.endswith(".BSE")
        & (pd.to_numeric(out["circ_mv"], errors="coerce") >= 200000.0)
        & (pd.to_numeric(out["turnover_rate"], errors="coerce") >= 0.5)
    )
    out["open_executable"] = (
        out["has_valid_bar"]
        & out["is_list_life_valid"]
        & ~out["entry_is_st"]
        & (out["entry_age_days"] >= 60)
        & ~out["entry_limit_up"]
        & ~out["entry_limit_down"]
        & out["liquidity_ok"]
        & out["entry_open"].notna()
    )
    out["main_gap_-3_+3"] = out["open_executable"] & (out["entry_gap_pct"] >= -0.03) & (out["entry_gap_pct"] < 0.03)
    return out.drop(columns=["open"], errors="ignore")


def gap_bucket(value: float) -> str:
    """Bucket the T open gap from the pre-L1 signal close."""
    if not math.isfinite(value):
        return "unknown"
    if value < -0.03:
        return "low_open_lt_-3"
    if value < -0.01:
        return "small_low_-3_-1"
    if value < 0.01:
        return "flat_-1_+1"
    if value < 0.03:
        return "small_high_+1_+3"
    if value < 0.05:
        return "high_open_+3_+5"
    return "chase_gt_+5"


def add_open_confirm_fields(events: pd.DataFrame) -> pd.DataFrame:
    """Add T-open confirmation buckets for pre-L1 watchlist rows."""
    if events.empty:
        return events.copy()
    out = events.copy()
    entry_open = pd.to_numeric(out["entry_open"], errors="coerce")
    signal_close = pd.to_numeric(out["signal_close"], errors="coerce")
    signal_high = pd.to_numeric(out["signal_high"], errors="coerce")
    signal_low = pd.to_numeric(out["signal_low"], errors="coerce")
    out["open_vs_signal_close"] = entry_open / signal_close - 1.0
    out["open_vs_signal_high"] = entry_open / signal_high - 1.0
    out["open_vs_signal_low"] = entry_open / signal_low - 1.0
    out["open_gap_bucket"] = out["open_vs_signal_close"].map(gap_bucket)

    out["open_above_signal_close"] = entry_open > signal_close
    out["open_above_signal_high"] = entry_open > signal_high
    out["open_holds_signal_low"] = entry_open >= signal_low
    out["open_structure_fail"] = entry_open < signal_low
    out["open_high_chase"] = out["open_vs_signal_close"] >= 0.05
    out["open_low_fail"] = out["open_vs_signal_close"] < -0.03

    bucket = pd.Series("other_open", index=out.index, dtype=object)
    bucket = bucket.mask(out["entry_limit_up"].fillna(False) | out["open_high_chase"].fillna(False), "reject_chase_or_limit")
    bucket = bucket.mask(out["open_low_fail"].fillna(False) | out["open_structure_fail"].fillna(False), "fail_low_or_break_structure")
    bucket = bucket.mask(
        out["open_executable"]
        & out["open_above_signal_high"]
        & (out["open_vs_signal_close"] < 0.05),
        "strong_open_break_high",
    )
    bucket = bucket.mask(
        out["open_executable"]
        & out["open_above_signal_close"]
        & (out["open_vs_signal_close"] >= 0)
        & (out["open_vs_signal_close"] < 0.03)
        & ~out["open_above_signal_high"],
        "normal_open_above_close",
    )
    bucket = bucket.mask(
        out["open_executable"]
        & out["open_holds_signal_low"]
        & (out["open_vs_signal_close"] >= -0.03)
        & (out["open_vs_signal_close"] < 0),
        "small_low_hold_structure",
    )
    bucket = bucket.mask(
        out["open_executable"]
        & (out["open_vs_signal_close"] >= 0.03)
        & (out["open_vs_signal_close"] < 0.05)
        & ~out["open_above_signal_high"],
        "high_open_watch",
    )
    out["open_confirm_bucket"] = bucket
    out["open_confirm_pass"] = out["open_confirm_bucket"].isin(
        [
            "strong_open_break_high",
            "normal_open_above_close",
            "small_low_hold_structure",
        ]
    )
    return out


def add_forward_labels(
    row: dict[str, Any],
    data: pd.DataFrame,
    *,
    signal_idx: int,
    open_rate: float,
    close_rate: float,
) -> dict[str, Any]:
    """Add next-open entry labels and fixed/TP outcomes."""
    entry_idx = signal_idx + 1
    if entry_idx >= len(data):
        row["entry_date"] = pd.NaT
        row["entry_open"] = math.nan
        row["has_entry"] = False
        return row

    entry_open = float(data.at[entry_idx, "open"])
    row["entry_date"] = data.at[entry_idx, "datetime"]
    row["entry_open"] = entry_open
    row["has_entry"] = math.isfinite(entry_open) and entry_open > 0
    row["fixed_0d_net_return"] = net_return(
        entry_open,
        float(data.at[entry_idx, "close"]),
        open_rate,
        close_rate,
    )
    for horizon in FIXED_HORIZONS:
        exit_idx = entry_idx + horizon
        row[f"has_full_{horizon}d"] = exit_idx < len(data)
        if exit_idx >= len(data):
            row[f"fixed_{horizon}d_net_return"] = math.nan
            continue
        row[f"fixed_{horizon}d_net_return"] = net_return(
            entry_open,
            float(data.at[exit_idx, "close"]),
            open_rate,
            close_rate,
        )

    for horizon in PATH_HORIZONS:
        exit_idx = entry_idx + horizon
        row[f"has_path_{horizon}d"] = exit_idx < len(data)
        if exit_idx >= len(data):
            row[f"mfe_{horizon}d"] = math.nan
            row[f"mae_{horizon}d"] = math.nan
            for level in TP_LEVELS:
                suffix = int(round(level * 100))
                row[f"tp{suffix}_hit_{horizon}d"] = False
                row[f"days_to_tp{suffix}_{horizon}d"] = math.nan
            continue

        path = data.iloc[entry_idx : exit_idx + 1]
        row[f"mfe_{horizon}d"] = price_return(float(path["high"].max()), entry_open) if not path.empty else math.nan
        row[f"mae_{horizon}d"] = price_return(float(path["low"].min()), entry_open) if not path.empty else math.nan
        for level in TP_LEVELS:
            suffix = int(round(level * 100))
            target = entry_open * (1.0 + level)
            hit_rows = path[path["high"] >= target] if not path.empty else pd.DataFrame()
            hit = not hit_rows.empty
            row[f"tp{suffix}_hit_{horizon}d"] = hit
            row[f"days_to_tp{suffix}_{horizon}d"] = int(hit_rows.index[0]) - entry_idx if hit else math.nan

    max_exit_idx = min(entry_idx + 5, len(data) - 1)
    exitable_start = min(entry_idx + 1, max_exit_idx)
    full_path = data.iloc[entry_idx : max_exit_idx + 1]
    exitable_path = data.iloc[exitable_start : max_exit_idx + 1]
    timeout_close = float(data.at[max_exit_idx, "close"]) if max_exit_idx >= entry_idx else math.nan
    row["timeout_5d_net_return"] = net_return(entry_open, timeout_close, open_rate, close_rate)
    row["mfe_5d"] = price_return(float(full_path["high"].max()), entry_open) if not full_path.empty else math.nan
    row["mae_5d"] = price_return(float(full_path["low"].min()), entry_open) if not full_path.empty else math.nan

    for level in TP_LEVELS:
        suffix = int(round(level * 100))
        target = entry_open * (1.0 + level)
        hit_rows = exitable_path[exitable_path["high"] >= target] if not exitable_path.empty else pd.DataFrame()
        hit = not hit_rows.empty
        row[f"tp{suffix}_hit_5d"] = hit
        row[f"tp{suffix}_plan_net_return_5d"] = (
            net_return(entry_open, target, open_rate, close_rate) if hit else row["timeout_5d_net_return"]
        )
        row[f"days_to_tp{suffix}"] = int(hit_rows.index[0]) - entry_idx if hit else math.nan
    return row


def add_breakout_exit_labels(
    row: dict[str, Any],
    data: pd.DataFrame,
    *,
    signal_idx: int,
    open_rate: float,
    close_rate: float,
    horizon: int = 3,
) -> None:
    """Add breakout-specific exit labels for T+1 open breakout entries."""
    entry_idx = signal_idx + 1
    entry_open = row.get("entry_open", math.nan)
    signal_high = row.get("signal_high", math.nan)
    if (
        entry_idx >= len(data)
        or not math.isfinite(float(entry_open))
        or float(entry_open) <= 0
        or not math.isfinite(float(signal_high))
        or float(signal_high) <= 0
    ):
        for level in TP_LEVELS:
            suffix = int(round(level * 100))
            row[f"b1_close_fail_tp{suffix}_{horizon}d_net_return"] = math.nan
            row[f"b1_close_fail_tp{suffix}_{horizon}d_exit_reason"] = "no_entry"
            row[f"b1_close_fail_tp{suffix}_{horizon}d_exit_day"] = math.nan
            row[f"b1_low_fail_stop_first_tp{suffix}_{horizon}d_net_return"] = math.nan
            row[f"b1_low_fail_stop_first_tp{suffix}_{horizon}d_exit_reason"] = "no_entry"
            row[f"b1_low_fail_tp_first_tp{suffix}_{horizon}d_net_return"] = math.nan
            row[f"b1_low_fail_tp_first_tp{suffix}_{horizon}d_exit_reason"] = "no_entry"
        return

    entry_open = float(entry_open)
    signal_high = float(signal_high)
    max_exit_idx = min(entry_idx + horizon, len(data) - 1)
    path = data.iloc[entry_idx : max_exit_idx + 1]
    stop_level = signal_high * 0.995
    timeout_close = float(data.at[max_exit_idx, "close"])

    for level in TP_LEVELS:
        suffix = int(round(level * 100))
        target = entry_open * (1.0 + level)

        close_fail_price = timeout_close
        close_fail_reason = "timeout"
        close_fail_day = max_exit_idx - entry_idx
        for idx, bar in path.iterrows():
            if float(bar["high"]) >= target:
                close_fail_price = target
                close_fail_reason = f"tp{suffix}"
                close_fail_day = int(idx) - entry_idx
                break
            if float(bar["close"]) < signal_high:
                close_fail_price = float(bar["close"])
                close_fail_reason = "close_below_signal_high"
                close_fail_day = int(idx) - entry_idx
                break
        row[f"b1_close_fail_tp{suffix}_{horizon}d_net_return"] = net_return(
            entry_open,
            close_fail_price,
            open_rate,
            close_rate,
        )
        row[f"b1_close_fail_tp{suffix}_{horizon}d_exit_reason"] = close_fail_reason
        row[f"b1_close_fail_tp{suffix}_{horizon}d_exit_day"] = close_fail_day

        for mode in ("stop_first", "tp_first"):
            exit_price = timeout_close
            exit_reason = "timeout"
            exit_day = max_exit_idx - entry_idx
            for idx, bar in path.iterrows():
                hit_stop = float(bar["low"]) <= stop_level
                hit_target = float(bar["high"]) >= target
                if mode == "stop_first":
                    if hit_stop:
                        exit_price = stop_level
                        exit_reason = "low_break_signal_high"
                        exit_day = int(idx) - entry_idx
                        break
                    if hit_target:
                        exit_price = target
                        exit_reason = f"tp{suffix}"
                        exit_day = int(idx) - entry_idx
                        break
                else:
                    if hit_target:
                        exit_price = target
                        exit_reason = f"tp{suffix}"
                        exit_day = int(idx) - entry_idx
                        break
                    if hit_stop:
                        exit_price = stop_level
                        exit_reason = "low_break_signal_high"
                        exit_day = int(idx) - entry_idx
                        break
            prefix = f"b1_low_fail_{mode}_tp{suffix}_{horizon}d"
            row[f"{prefix}_net_return"] = net_return(entry_open, exit_price, open_rate, close_rate)
            row[f"{prefix}_exit_reason"] = exit_reason
            row[f"{prefix}_exit_day"] = exit_day


def finite_positive(values: list[float]) -> list[float]:
    """Return finite positive values."""
    return [value for value in values if math.isfinite(value) and value > 0]


def add_entry_variant_outcome(
    row: dict[str, Any],
    data: pd.DataFrame,
    *,
    prefix: str,
    entry_idx: int | None,
    entry_price: float,
    open_rate: float,
    close_rate: float,
) -> None:
    """Add fixed-horizon and TP outcomes for one entry variant."""
    has_entry = (
        entry_idx is not None
        and 0 <= entry_idx < len(data)
        and math.isfinite(entry_price)
        and entry_price > 0
    )
    row[f"{prefix}_has_entry"] = has_entry
    if not has_entry or entry_idx is None:
        row[f"{prefix}_entry_date"] = pd.NaT
        row[f"{prefix}_entry_price"] = math.nan
        for horizon in FIXED_HORIZONS:
            row[f"{prefix}_fixed_{horizon}d_net_return"] = math.nan
        row[f"{prefix}_mfe_5d"] = math.nan
        row[f"{prefix}_mae_5d"] = math.nan
        for level in TP_LEVELS:
            suffix = int(round(level * 100))
            row[f"{prefix}_tp{suffix}_hit_5d"] = False
            row[f"{prefix}_tp{suffix}_plan_net_return_5d"] = math.nan
        return

    row[f"{prefix}_entry_date"] = data.at[entry_idx, "datetime"]
    row[f"{prefix}_entry_price"] = entry_price
    for horizon in FIXED_HORIZONS:
        exit_idx = entry_idx + horizon
        row[f"{prefix}_has_full_{horizon}d"] = exit_idx < len(data)
        if exit_idx >= len(data):
            row[f"{prefix}_fixed_{horizon}d_net_return"] = math.nan
        else:
            row[f"{prefix}_fixed_{horizon}d_net_return"] = net_return(
                entry_price,
                float(data.at[exit_idx, "close"]),
                open_rate,
                close_rate,
            )

    max_exit_idx = min(entry_idx + 5, len(data) - 1)
    exitable_start = min(entry_idx + 1, max_exit_idx)
    full_path = data.iloc[entry_idx : max_exit_idx + 1]
    exitable_path = data.iloc[exitable_start : max_exit_idx + 1]
    timeout_close = float(data.at[max_exit_idx, "close"]) if max_exit_idx >= entry_idx else math.nan
    timeout_net = net_return(entry_price, timeout_close, open_rate, close_rate)
    row[f"{prefix}_timeout_5d_net_return"] = timeout_net
    row[f"{prefix}_mfe_5d"] = price_return(float(full_path["high"].max()), entry_price) if not full_path.empty else math.nan
    row[f"{prefix}_mae_5d"] = price_return(float(full_path["low"].min()), entry_price) if not full_path.empty else math.nan

    for level in TP_LEVELS:
        suffix = int(round(level * 100))
        target = entry_price * (1.0 + level)
        hit_rows = exitable_path[exitable_path["high"] >= target] if not exitable_path.empty else pd.DataFrame()
        hit = not hit_rows.empty
        row[f"{prefix}_tp{suffix}_hit_5d"] = hit
        row[f"{prefix}_tp{suffix}_plan_net_return_5d"] = (
            net_return(entry_price, target, open_rate, close_rate) if hit else timeout_net
        )
        row[f"{prefix}_days_to_tp{suffix}"] = int(hit_rows.index[0]) - entry_idx if hit else math.nan


def add_confirm_entry_comparison(
    row: dict[str, Any],
    data: pd.DataFrame,
    *,
    signal_idx: int,
    open_rate: float,
    close_rate: float,
) -> None:
    """Add T confirmation entry variants for tomorrow-L1 rows."""
    confirm_idx = signal_idx + 1
    l1_next_open_idx = signal_idx + 2
    if confirm_idx >= len(data):
        return

    confirm_open = float(data.at[confirm_idx, "open"])
    confirm_high = float(data.at[confirm_idx, "high"])
    confirm_close = float(data.at[confirm_idx, "close"])
    trigger_candidates = finite_positive(
        [
            float(data.at[signal_idx, "high"]),
            float(data.at[confirm_idx, "ema_fast"]),
            float(data.at[confirm_idx, "break_high_level"]),
        ]
    )
    trigger_level = max(trigger_candidates) if trigger_candidates else math.nan
    trigger_hit = math.isfinite(trigger_level) and math.isfinite(confirm_high) and confirm_high >= trigger_level
    trigger_price = max(confirm_open, trigger_level) if trigger_hit and math.isfinite(confirm_open) else math.nan

    l1_next_open_price = (
        float(data.at[l1_next_open_idx, "open"])
        if l1_next_open_idx < len(data)
        else math.nan
    )
    row["confirm_date"] = data.at[confirm_idx, "datetime"]
    row["confirm_trigger_level"] = trigger_level
    row["confirm_trigger_hit"] = trigger_hit
    row["l1_next_open_date"] = data.at[l1_next_open_idx, "datetime"] if l1_next_open_idx < len(data) else pd.NaT
    row["l1_next_open_price"] = l1_next_open_price

    add_entry_variant_outcome(
        row,
        data,
        prefix="confirm_open",
        entry_idx=confirm_idx,
        entry_price=confirm_open,
        open_rate=open_rate,
        close_rate=close_rate,
    )
    add_entry_variant_outcome(
        row,
        data,
        prefix="confirm_trigger",
        entry_idx=confirm_idx if trigger_hit else None,
        entry_price=trigger_price,
        open_rate=open_rate,
        close_rate=close_rate,
    )
    add_entry_variant_outcome(
        row,
        data,
        prefix="confirm_close",
        entry_idx=confirm_idx,
        entry_price=confirm_close,
        open_rate=open_rate,
        close_rate=close_rate,
    )
    add_entry_variant_outcome(
        row,
        data,
        prefix="l1_next_open",
        entry_idx=l1_next_open_idx if l1_next_open_idx < len(data) else None,
        entry_price=l1_next_open_price,
        open_rate=open_rate,
        close_rate=close_rate,
    )

    if math.isfinite(l1_next_open_price) and l1_next_open_price > 0:
        for prefix in CONFIRM_ENTRY_VARIANTS:
            price = row.get(f"{prefix}_entry_price", math.nan)
            row[f"{prefix}_price_adv_vs_l1_next_open"] = price_return(float(price), l1_next_open_price)


def collect_symbol_pre_l1(
    vt_symbol: str,
    data: pd.DataFrame,
    *,
    l1_dates: set[pd.Timestamp],
    config: TrendRsiConfig,
    cooldown_bars: int,
    open_rate: float,
    close_rate: float,
    b1_only: bool = False,
) -> list[dict[str, Any]]:
    """Collect pre-L1 near-start candidates for one symbol."""
    bars = compute_base_columns(data, config)
    down_context = bars["down_context"].to_numpy(dtype=bool)
    bull_div = bars["bull_div"].to_numpy(dtype=bool)
    rsi_return_now = bars["rsi_return_now"].to_numpy(dtype=bool)
    false_break_low = bars["false_break_low"].to_numpy(dtype=bool)
    recent_oversold = bars["recent_oversold"].to_numpy(dtype=bool)
    long_lower_wick = bars["long_lower_wick"].to_numpy(dtype=bool)
    strong_bull_body = bars["strong_bull_body"].to_numpy(dtype=bool)
    price_break = bars["price_break"].to_numpy(dtype=bool)
    momentum_turn = bars["momentum_turn"].to_numpy(dtype=bool)
    vol_pass = bars["vol_pass"].to_numpy(dtype=bool)

    open_values = bars["open"].to_numpy(dtype=float)
    high_values = bars["high"].to_numpy(dtype=float)
    low_values = bars["low"].to_numpy(dtype=float)
    close_values = bars["close"].to_numpy(dtype=float)
    rsi_values = bars["rsi"].to_numpy(dtype=float)
    atr_values = bars["atr"].to_numpy(dtype=float)
    ema_fast_values = bars["ema_fast"].to_numpy(dtype=float)
    ema_mid_values = bars["ema_mid"].to_numpy(dtype=float)
    ema_slow_values = bars["ema_slow"].to_numpy(dtype=float)
    macd_hist_values = bars["macd_hist"].to_numpy(dtype=float)
    break_high_values = bars["break_high_level"].to_numpy(dtype=float)
    vol_ma_values = bars["vol_ma"].to_numpy(dtype=float)
    volume_values = bars["volume"].to_numpy(dtype=float)
    datetime_values = pd.to_datetime(bars["datetime"]).dt.normalize().to_numpy()

    rows: list[dict[str, Any]] = []
    last_exhaust_bar: int | None = None
    last_pre_bar: int | None = None

    for i in range(2, len(bars) - 1):
        signal_date = pd.Timestamp(datetime_values[i]).normalize()
        next_date = pd.Timestamp(datetime_values[i + 1]).normalize()
        bar_range = max(high_values[i] - low_values[i], 0.01)
        real_body = abs(close_values[i] - open_values[i])
        close_position = (close_values[i] - low_values[i]) / bar_range
        rsi_rising = rsi_values[i] > rsi_values[i - 1]
        macd_improving = macd_hist_values[i] > macd_hist_values[i - 1]
        ema20_distance = close_values[i] / ema_fast_values[i] - 1.0 if ema_fast_values[i] > 0 else math.nan
        break_high_distance = (
            close_values[i] / break_high_values[i] - 1.0
            if math.isfinite(break_high_values[i]) and break_high_values[i] > 0
            else math.nan
        )
        vol_ratio = volume_values[i] / vol_ma_values[i] if math.isfinite(vol_ma_values[i]) and vol_ma_values[i] > 0 else math.nan

        exhaust_raw = bool(
            down_context[i]
            and (
                bull_div[i]
                or rsi_return_now[i]
                or false_break_low[i]
                or (recent_oversold[i] and long_lower_wick[i])
            )
        )
        recent_exhaust_before = last_exhaust_bar is not None and i - last_exhaust_bar <= config.start_window
        exhaust_now_or_recent = exhaust_raw or recent_exhaust_before

        panic_reversal = bool(
            down_context[i]
            and strong_bull_body[i]
            and price_break[i]
            and rsi_rising
            and (vol_pass[i] or real_body >= atr_values[i] * (config.body_atr_mult + 0.40))
        )
        exhaustion_start = bool(
            exhaust_now_or_recent
            and price_break[i]
            and rsi_values[i] >= config.rsi_start_line
            and momentum_turn[i]
            and (vol_pass[i] or strong_bull_body[i])
        )
        start_raw = exhaustion_start or panic_reversal

        near_ema20_reclaim = bool(
            math.isfinite(ema20_distance)
            and (
                -0.035 <= ema20_distance < 0.012
                or high_values[i] >= ema_fast_values[i] * 0.995
            )
        )
        near_break_high = bool(
            math.isfinite(break_high_distance)
            and (
                -0.045 <= break_high_distance < 0.012
                or high_values[i] >= break_high_values[i] * 0.995
            )
        )
        near_price_break = near_ema20_reclaim or near_break_high
        momentum_pre = bool(rsi_rising or momentum_turn[i] or macd_improving)
        rsi_repairing = bool(rsi_values[i] >= 36 and (rsi_rising or rsi_values[i] >= 40))
        vol_not_climax = not (math.isfinite(vol_ratio) and vol_ratio > 1.8 and close_position < 0.65)
        anti_chase = bool(
            not (
                close_values[i] > ema_mid_values[i]
                and ema_fast_values[i] > ema_mid_values[i]
                and ema_mid_values[i] > ema_slow_values[i]
                and rsi_values[i] >= 60
            )
        )

        strict = bool(
            exhaust_now_or_recent
            and not start_raw
            and down_context[i]
            and near_price_break
            and rsi_repairing
            and momentum_pre
            and vol_not_climax
            and anti_chase
        )
        false_break_repair = bool(
            not start_raw
            and down_context[i]
            and (false_break_low[i] or (recent_oversold[i] and long_lower_wick[i]))
            and near_price_break
            and close_position >= 0.55
            and rsi_rising
            and vol_not_climax
        )
        panic_pre = bool(
            not start_raw
            and down_context[i]
            and near_price_break
            and close_position >= 0.62
            and real_body >= atr_values[i] * 0.45
            and rsi_rising
            and vol_not_climax
        )

        parts = [
            label
            for label, flag in (
                ("near_exhaustion_strict", strict),
                ("false_break_repair", false_break_repair),
                ("panic_pre", panic_pre),
            )
            if flag
        ]
        if parts and (last_pre_bar is None or i - last_pre_bar > cooldown_bars):
            component_count = len(parts)
            score = 0
            score += 20 if exhaust_now_or_recent else 0
            score += 16 if near_price_break else 0
            score += 14 if momentum_pre else 0
            score += 12 if rsi_repairing else 0
            score += 10 if false_break_low[i] else 0
            score += 8 if close_position >= 0.62 else 0
            score += 8 if vol_not_climax else -12
            score += 6 if anti_chase else -8
            score += 6 if component_count >= 2 else 0
            score = max(0, min(100, score))

            next_open = open_values[i + 1]
            prelim_b1 = bool(
                math.isfinite(next_open)
                and next_open > high_values[i]
                and price_return(next_open, close_values[i]) < 0.05
            )
            if not b1_only or prelim_b1:
                next_l1_dates = {
                    pd.Timestamp(datetime_values[j]).normalize()
                    for j in range(i + 1, min(i + 6, len(bars)))
                    if pd.Timestamp(datetime_values[j]).normalize() in l1_dates
                }
                row: dict[str, Any] = {
                    "signal_date": signal_date,
                    "vt_symbol": vt_symbol,
                    "buy_point_type": "PRE_L1_near_start",
                    "pre_l1_type": "+".join(parts),
                    "pre_l1_score": score,
                    "component_count": component_count,
                    "signal_close": close_values[i],
                    "signal_low": low_values[i],
                    "signal_high": high_values[i],
                    "signal_rsi": rsi_values[i],
                    "signal_close_position": close_position,
                    "signal_vol_ratio": vol_ratio,
                    "ema20_distance": ema20_distance,
                    "break_high_distance": break_high_distance,
                    "exhaust_now_or_recent": exhaust_now_or_recent,
                    "near_price_break": near_price_break,
                    "momentum_pre": momentum_pre,
                    "rsi_repairing": rsi_repairing,
                    "vol_not_climax": vol_not_climax,
                    "anti_chase": anti_chase,
                    "tomorrow_l1": next_date in l1_dates,
                    "l1_within_3d": any(
                        pd.Timestamp(datetime_values[j]).normalize() in l1_dates
                        for j in range(i + 1, min(i + 4, len(bars)))
                    ),
                    "l1_within_5d": bool(next_l1_dates),
                }
                add_forward_labels(
                    row,
                    bars,
                    signal_idx=i,
                    open_rate=open_rate,
                    close_rate=close_rate,
                )
                if prelim_b1:
                    add_breakout_exit_labels(
                        row,
                        bars,
                        signal_idx=i,
                        open_rate=open_rate,
                        close_rate=close_rate,
                    )
                if row["tomorrow_l1"] and not b1_only:
                    add_confirm_entry_comparison(
                        row,
                        bars,
                        signal_idx=i,
                        open_rate=open_rate,
                        close_rate=close_rate,
                    )
                rows.append(row)
            last_pre_bar = i

        if exhaust_raw and (last_exhaust_bar is None or i - last_exhaust_bar > config.cooldown_bars):
            last_exhaust_bar = i

    return rows


def summarize_group(frame: pd.DataFrame, *, scope: str, group_name: str, group_value: str) -> dict[str, Any]:
    """Summarize a candidate subset."""
    row: dict[str, Any] = {
        "scope": scope,
        "group": group_name,
        "bucket": group_value,
        "events": int(len(frame)),
        "days": int(frame["signal_date"].nunique()) if not frame.empty else 0,
        "symbols": int(frame["vt_symbol"].nunique()) if not frame.empty else 0,
    }
    if frame.empty:
        return row

    row.update(
        {
            "tomorrow_l1_rate": float(frame["tomorrow_l1"].mean()),
            "l1_within_3d_rate": float(frame["l1_within_3d"].mean()),
            "l1_within_5d_rate": float(frame["l1_within_5d"].mean()),
            "open_executable_rate": float(frame["open_executable"].mean()) if "open_executable" in frame.columns else math.nan,
            "entry_gap_avg": float(pd.to_numeric(frame["entry_gap_pct"], errors="coerce").mean()) if "entry_gap_pct" in frame.columns else math.nan,
            "tp3_rate": float(frame["tp3_hit_5d"].mean()),
            "tp5_rate": float(frame["tp5_hit_5d"].mean()),
            "tp3_plan_net_avg": float(pd.to_numeric(frame["tp3_plan_net_return_5d"], errors="coerce").mean()),
            "tp5_plan_net_avg": float(pd.to_numeric(frame["tp5_plan_net_return_5d"], errors="coerce").mean()),
            "fixed_1d_net_avg": float(pd.to_numeric(frame["fixed_1d_net_return"], errors="coerce").mean()),
            "fixed_3d_net_avg": float(pd.to_numeric(frame["fixed_3d_net_return"], errors="coerce").mean()),
            "fixed_5d_net_avg": float(pd.to_numeric(frame["fixed_5d_net_return"], errors="coerce").mean()),
            "mfe_5d_median": float(pd.to_numeric(frame["mfe_5d"], errors="coerce").median()),
            "mae_5d_median": float(pd.to_numeric(frame["mae_5d"], errors="coerce").median()),
            "p10_tp5_plan_net": float(pd.to_numeric(frame["tp5_plan_net_return_5d"], errors="coerce").quantile(0.10)),
        }
    )
    return row


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column or an all-NaN fallback."""
    if column not in frame.columns:
        return pd.Series(math.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a bool column or an all-False fallback."""
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].astype("boolean").fillna(False).astype(bool)


def b1_gap_bucket(value: float) -> str:
    """Bucket B1 open gap from T close."""
    if not math.isfinite(value):
        return "unknown"
    if value < 0:
        return "gap_lt_0"
    if value < 0.005:
        return "gap_0_0.5"
    if value < 0.015:
        return "gap_0.5_1.5"
    if value < 0.03:
        return "gap_1.5_3"
    if value < 0.05:
        return "gap_3_5"
    return "gap_ge_5"


def b1_break_bucket(value: float) -> str:
    """Bucket B1 open breakout distance above T high."""
    if not math.isfinite(value):
        return "unknown"
    if value < 0:
        return "break_lt_0"
    if value < 0.005:
        return "break_0_0.5"
    if value < 0.015:
        return "break_0.5_1.5"
    if value < 0.03:
        return "break_1.5_3"
    if value < 0.05:
        return "break_3_5"
    return "break_ge_5"


def b1_score_bucket(value: float) -> str:
    """Bucket pre-L1 score for B1 rows."""
    if not math.isfinite(value):
        return "unknown"
    if value >= 80:
        return "score_ge_80"
    if value >= 70:
        return "score_70_80"
    if value >= 60:
        return "score_60_70"
    return "score_lt_60"


def add_b1_fields(events: pd.DataFrame) -> pd.DataFrame:
    """Add B1 open-breakout study fields."""
    if events.empty:
        return events.copy()
    out = events.copy()
    out["b1_signal"] = out["open_confirm_bucket"].eq("strong_open_break_high")
    out["b1_gap_bucket"] = numeric_column(out, "open_vs_signal_close").map(b1_gap_bucket)
    out["b1_break_bucket"] = numeric_column(out, "open_vs_signal_high").map(b1_break_bucket)
    out["b1_score_bucket"] = numeric_column(out, "pre_l1_score").map(b1_score_bucket)
    out["b1_gap_0_3"] = out["b1_signal"] & (numeric_column(out, "open_vs_signal_close") >= 0) & (
        numeric_column(out, "open_vs_signal_close") < 0.03
    )
    out["b1_break_0_3"] = out["b1_signal"] & (numeric_column(out, "open_vs_signal_high") >= 0) & (
        numeric_column(out, "open_vs_signal_high") < 0.03
    )
    out["b1_score_ge_70"] = out["b1_signal"] & (numeric_column(out, "pre_l1_score") >= 70)
    out["b1_component_ge_2"] = out["b1_signal"] & (numeric_column(out, "component_count") >= 2)
    out["b1_gap_0_3_score_ge_70"] = out["b1_gap_0_3"] & out["b1_score_ge_70"]
    out["b1_gap_0_3_break_0_3"] = out["b1_gap_0_3"] & out["b1_break_0_3"]
    return out


def summarize_b1_group(frame: pd.DataFrame, *, group: str, bucket: str) -> dict[str, Any]:
    """Summarize one B1 subset."""
    row: dict[str, Any] = {
        "group": group,
        "bucket": bucket,
        "events": int(len(frame)),
        "days": int(frame["entry_date"].nunique()) if not frame.empty else 0,
        "symbols": int(frame["vt_symbol"].nunique()) if not frame.empty else 0,
    }
    if frame.empty:
        return row

    row.update(
        {
            "tomorrow_l1_rate": float(bool_column(frame, "tomorrow_l1").mean()),
            "entry_gap_avg": float(numeric_column(frame, "open_vs_signal_close").mean()),
            "entry_gap_p50": float(numeric_column(frame, "open_vs_signal_close").median()),
            "break_high_avg": float(numeric_column(frame, "open_vs_signal_high").mean()),
            "break_high_p50": float(numeric_column(frame, "open_vs_signal_high").median()),
            "pre_l1_score_avg": float(numeric_column(frame, "pre_l1_score").mean()),
            "component_ge_2_rate": float((numeric_column(frame, "component_count") >= 2).mean()),
            "fixed_0d_net_avg": float(numeric_column(frame, "fixed_0d_net_return").mean()),
            "fixed_1d_net_avg": float(numeric_column(frame, "fixed_1d_net_return").mean()),
            "fixed_3d_net_avg": float(numeric_column(frame, "fixed_3d_net_return").mean()),
            "tp3_hit_0d_rate": float(bool_column(frame, "tp3_hit_0d").mean()),
            "tp5_hit_0d_rate": float(bool_column(frame, "tp5_hit_0d").mean()),
            "mfe_0d_median": float(numeric_column(frame, "mfe_0d").median()),
            "mae_0d_median": float(numeric_column(frame, "mae_0d").median()),
            "tp3_hit_1d_rate": float(bool_column(frame, "tp3_hit_1d").mean()),
            "tp5_hit_1d_rate": float(bool_column(frame, "tp5_hit_1d").mean()),
            "mfe_1d_median": float(numeric_column(frame, "mfe_1d").median()),
            "mae_1d_median": float(numeric_column(frame, "mae_1d").median()),
            "tp3_hit_3d_rate": float(bool_column(frame, "tp3_hit_3d").mean()),
            "tp5_hit_3d_rate": float(bool_column(frame, "tp5_hit_3d").mean()),
            "mfe_3d_median": float(numeric_column(frame, "mfe_3d").median()),
            "mae_3d_median": float(numeric_column(frame, "mae_3d").median()),
            "close_fail_tp3_3d_avg": float(numeric_column(frame, "b1_close_fail_tp3_3d_net_return").mean()),
            "close_fail_tp5_3d_avg": float(numeric_column(frame, "b1_close_fail_tp5_3d_net_return").mean()),
            "low_stop_first_tp3_3d_avg": float(numeric_column(frame, "b1_low_fail_stop_first_tp3_3d_net_return").mean()),
            "low_tp_first_tp3_3d_avg": float(numeric_column(frame, "b1_low_fail_tp_first_tp3_3d_net_return").mean()),
            "close_fail_tp3_3d_p10": float(numeric_column(frame, "b1_close_fail_tp3_3d_net_return").quantile(0.10)),
            "close_fail_tp3_3d_p50": float(numeric_column(frame, "b1_close_fail_tp3_3d_net_return").quantile(0.50)),
            "close_fail_tp3_3d_positive_rate": float(
                (numeric_column(frame, "b1_close_fail_tp3_3d_net_return") > 0).mean()
            ),
        }
    )
    return row


def b1_named_segments(events: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return named B1 subsets."""
    if events.empty:
        return []
    b1 = events[events["b1_signal"]].copy()
    return [
        ("B1_raw", b1),
        ("B1_gap_0_3", events[events["b1_gap_0_3"]].copy()),
        ("B1_break_0_3", events[events["b1_break_0_3"]].copy()),
        ("B1_gap_0_3_break_0_3", events[events["b1_gap_0_3_break_0_3"]].copy()),
        ("B1_gap_0_0.5", b1[b1["b1_gap_bucket"].eq("gap_0_0.5")].copy()),
        ("B1_gap_0.5_1.5", b1[b1["b1_gap_bucket"].eq("gap_0.5_1.5")].copy()),
        ("B1_gap_1.5_3", b1[b1["b1_gap_bucket"].eq("gap_1.5_3")].copy()),
        ("B1_gap_3_5", b1[b1["b1_gap_bucket"].eq("gap_3_5")].copy()),
        ("B1_break_0_0.5", b1[b1["b1_break_bucket"].eq("break_0_0.5")].copy()),
        ("B1_break_0.5_1.5", b1[b1["b1_break_bucket"].eq("break_0.5_1.5")].copy()),
        ("B1_break_1.5_3", b1[b1["b1_break_bucket"].eq("break_1.5_3")].copy()),
        ("B1_break_3_5", b1[b1["b1_break_bucket"].eq("break_3_5")].copy()),
        ("B1_score_ge_70", events[events["b1_score_ge_70"]].copy()),
        ("B1_component_ge_2", events[events["b1_component_ge_2"]].copy()),
        ("B1_gap_0_3_score_ge_70", events[events["b1_gap_0_3_score_ge_70"]].copy()),
    ]


def summarize_b1(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build B1 event-level, daily-level, and yearly summaries."""
    if events.empty or "b1_signal" not in events.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    b1 = events[events["b1_signal"]].copy()
    if b1.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    rows.extend(summarize_b1_group(frame, group="segment", bucket=name) for name, frame in b1_named_segments(events))
    rows.extend(
        summarize_b1_group(group_frame, group="gap_bucket", bucket=str(bucket))
        for bucket, group_frame in b1.groupby("b1_gap_bucket", sort=True)
    )
    rows.extend(
        summarize_b1_group(group_frame, group="break_bucket", bucket=str(bucket))
        for bucket, group_frame in b1.groupby("b1_break_bucket", sort=True)
    )
    rows.extend(
        summarize_b1_group(group_frame, group="score_bucket", bucket=str(bucket))
        for bucket, group_frame in b1.groupby("b1_score_bucket", sort=True)
    )
    rows.extend(
        summarize_b1_group(group_frame, group="pre_l1_type", bucket=str(bucket))
        for bucket, group_frame in b1.groupby("pre_l1_type", sort=True)
    )

    return_columns = [
        "fixed_0d_net_return",
        "fixed_1d_net_return",
        "fixed_3d_net_return",
        "b1_close_fail_tp3_3d_net_return",
        "b1_close_fail_tp5_3d_net_return",
        "b1_low_fail_stop_first_tp3_3d_net_return",
        "b1_low_fail_tp_first_tp3_3d_net_return",
    ]
    daily_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    for name, frame in b1_named_segments(events):
        if frame.empty:
            continue
        for return_name in return_columns:
            daily = frame.groupby("entry_date", sort=True)[return_name].mean().dropna()
            daily_rows.append(
                {
                    "segment": name,
                    "return_name": return_name,
                    "events": int(len(frame)),
                    "days": int(daily.size),
                    "daily_avg": float(daily.mean()) if not daily.empty else math.nan,
                    "daily_p10": float(daily.quantile(0.10)) if not daily.empty else math.nan,
                    "daily_p50": float(daily.quantile(0.50)) if not daily.empty else math.nan,
                    "positive_day_rate": float((daily > 0).mean()) if not daily.empty else math.nan,
                    "max_drawdown": drawdown(daily),
                }
            )
        for year, year_frame in frame.groupby(frame["entry_date"].dt.year, sort=True):
            row = summarize_b1_group(year_frame, group="entry_year", bucket=str(year))
            row["segment"] = name
            year_rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(daily_rows), pd.DataFrame(year_rows)


def summarize_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build overall and by-year summaries."""
    rows: list[dict[str, Any]] = []
    by_year_rows: list[dict[str, Any]] = []
    for scope in SUMMARY_SCOPES:
        if scope == "all":
            scoped = events
        elif scope == "open_executable":
            scoped = events[events["open_executable"]]
        elif scope == "main_gap_-3_+3":
            scoped = events[events["main_gap_-3_+3"]]
        elif scope == "tomorrow_l1":
            scoped = events[events["tomorrow_l1"]]
        elif scope == "not_tomorrow_l1":
            scoped = events[~events["tomorrow_l1"]]
        else:
            continue
        rows.append(summarize_group(scoped, scope=scope, group_name="all", group_value="all"))
        for bucket, group in scoped.groupby("pre_l1_type", sort=True):
            rows.append(summarize_group(group, scope=scope, group_name="pre_l1_type", group_value=str(bucket)))
        if not scoped.empty:
            for year, group in scoped.groupby(scoped["signal_date"].dt.year, sort=True):
                by_year_rows.append(summarize_group(group, scope=scope, group_name="year", group_value=str(year)))
    return pd.DataFrame(rows), pd.DataFrame(by_year_rows)


def summarize_topk(events: pd.DataFrame, top_k_values: tuple[int, ...] = (2, 5, 10)) -> pd.DataFrame:
    """Summarize daily top-K by pre-L1 score."""
    rows: list[dict[str, Any]] = []
    pool = events[events["main_gap_-3_+3"]].copy()
    if pool.empty:
        return pd.DataFrame()
    pool = pool.sort_values(["signal_date", "pre_l1_score", "component_count", "vt_symbol"], ascending=[True, False, False, True])
    for top_k in top_k_values:
        selected = pool.groupby("signal_date", sort=True).head(top_k).copy()
        for return_name in ("fixed_1d_net_return", "fixed_3d_net_return", "fixed_5d_net_return", "tp5_plan_net_return_5d"):
            daily = selected.groupby("signal_date", sort=True)[return_name].mean().dropna()
            rows.append(
                {
                    "top_k": top_k,
                    "return_name": return_name,
                    "events": int(len(selected)),
                    "days": int(daily.size),
                    "daily_avg": float(daily.mean()) if not daily.empty else math.nan,
                    "daily_p10": float(daily.quantile(0.10)) if not daily.empty else math.nan,
                    "daily_p50": float(daily.quantile(0.50)) if not daily.empty else math.nan,
                    "positive_day_rate": float((daily > 0).mean()) if not daily.empty else math.nan,
                    "max_drawdown": drawdown(daily),
                    "tomorrow_l1_rate": float(selected["tomorrow_l1"].mean()) if not selected.empty else math.nan,
                    "tp5_rate": float(selected["tp5_hit_5d"].mean()) if not selected.empty else math.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_open_confirm_group(frame: pd.DataFrame, *, group: str, bucket: str) -> dict[str, Any]:
    """Summarize one T-open confirmation group."""
    row: dict[str, Any] = {
        "group": group,
        "bucket": bucket,
        "events": int(len(frame)),
        "days": int(frame["entry_date"].nunique()) if not frame.empty else 0,
        "symbols": int(frame["vt_symbol"].nunique()) if not frame.empty else 0,
    }
    if frame.empty:
        return row
    row.update(
        {
            "open_confirm_pass_rate": float(bool_column(frame, "open_confirm_pass").mean()),
            "tomorrow_l1_rate": float(bool_column(frame, "tomorrow_l1").mean()),
            "l1_within_3d_rate": float(bool_column(frame, "l1_within_3d").mean()),
            "l1_within_5d_rate": float(bool_column(frame, "l1_within_5d").mean()),
            "entry_gap_avg": float(numeric_column(frame, "entry_gap_pct").mean()),
            "tp3_rate": float(bool_column(frame, "tp3_hit_5d").mean()),
            "tp5_rate": float(bool_column(frame, "tp5_hit_5d").mean()),
            "tp3_plan_net_avg": float(numeric_column(frame, "tp3_plan_net_return_5d").mean()),
            "tp5_plan_net_avg": float(numeric_column(frame, "tp5_plan_net_return_5d").mean()),
            "fixed_1d_net_avg": float(numeric_column(frame, "fixed_1d_net_return").mean()),
            "fixed_3d_net_avg": float(numeric_column(frame, "fixed_3d_net_return").mean()),
            "fixed_5d_net_avg": float(numeric_column(frame, "fixed_5d_net_return").mean()),
            "tp3_hit_1d_rate": float(bool_column(frame, "tp3_hit_1d").mean()),
            "tp5_hit_1d_rate": float(bool_column(frame, "tp5_hit_1d").mean()),
            "mfe_1d_median": float(numeric_column(frame, "mfe_1d").median()),
            "mae_1d_median": float(numeric_column(frame, "mae_1d").median()),
            "tp3_hit_3d_rate": float(bool_column(frame, "tp3_hit_3d").mean()),
            "tp5_hit_3d_rate": float(bool_column(frame, "tp5_hit_3d").mean()),
            "mfe_3d_median": float(numeric_column(frame, "mfe_3d").median()),
            "mae_3d_median": float(numeric_column(frame, "mae_3d").median()),
            "mfe_5d_median": float(numeric_column(frame, "mfe_5d").median()),
            "mae_5d_median": float(numeric_column(frame, "mae_5d").median()),
            "p10_tp5_plan_net": float(numeric_column(frame, "tp5_plan_net_return_5d").quantile(0.10)),
            "positive_fixed_3d_rate": float((numeric_column(frame, "fixed_3d_net_return") > 0).mean()),
        }
    )
    return row


def summarize_open_confirm(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build T-open confirmation bucket summaries."""
    executable = events[events["open_executable"]].copy()
    if executable.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = [
        summarize_open_confirm_group(executable, group="all_open_executable", bucket="all"),
        summarize_open_confirm_group(
            executable[executable["open_confirm_pass"]],
            group="open_confirm_pass",
            bucket="pass",
        ),
    ]
    for bucket, group in executable.groupby("open_confirm_bucket", sort=True):
        rows.append(summarize_open_confirm_group(group, group="open_confirm_bucket", bucket=str(bucket)))
    for bucket, group in executable.groupby("open_gap_bucket", sort=True):
        rows.append(summarize_open_confirm_group(group, group="open_gap_bucket", bucket=str(bucket)))

    by_year_rows = [
        summarize_open_confirm_group(group, group="entry_year", bucket=str(year))
        for year, group in executable.groupby(executable["entry_date"].dt.year, sort=True)
    ]
    pass_by_year_rows = [
        summarize_open_confirm_group(group, group="entry_year_confirm_pass", bucket=str(year))
        for year, group in executable[executable["open_confirm_pass"]].groupby(
            executable[executable["open_confirm_pass"]]["entry_date"].dt.year,
            sort=True,
        )
    ]
    return (
        pd.DataFrame(rows),
        pd.DataFrame(by_year_rows),
        pd.DataFrame(pass_by_year_rows),
    )


def build_open_confirm_path_summary(open_confirm_summary: pd.DataFrame) -> pd.DataFrame:
    """Return a compact open-type path summary for 1d/3d TP and MFE/MAE."""
    columns = [
        "group",
        "bucket",
        "events",
        "days",
        "symbols",
        "tomorrow_l1_rate",
        "tp3_hit_1d_rate",
        "tp5_hit_1d_rate",
        "mfe_1d_median",
        "mae_1d_median",
        "tp3_hit_3d_rate",
        "tp5_hit_3d_rate",
        "mfe_3d_median",
        "mae_3d_median",
    ]
    if open_confirm_summary.empty:
        return pd.DataFrame(columns=columns)
    return open_confirm_summary[
        open_confirm_summary["group"].isin(["open_confirm_bucket", "open_gap_bucket"])
    ][columns].copy()


def summarize_confirm_entry_group(frame: pd.DataFrame, *, group: str, bucket: str) -> pd.DataFrame:
    """Summarize T-confirmation entry variants."""
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for variant in CONFIRM_ENTRY_VARIANTS:
        has_entry_col = f"{variant}_has_entry"
        if has_entry_col not in frame.columns:
            continue
        valid = frame[frame[has_entry_col].astype("boolean").fillna(False).astype(bool)].copy()
        if valid.empty:
            continue
        rows.append(
            {
                "group": group,
                "bucket": bucket,
                "entry_variant": variant,
                "events": int(len(valid)),
                "days": int(pd.to_datetime(valid["confirm_date"]).dt.normalize().nunique())
                if "confirm_date" in valid.columns
                else 0,
                "symbols": int(valid["vt_symbol"].nunique()),
                "avg_entry_price_adv_vs_l1_next_open": float(numeric_column(valid, f"{variant}_price_adv_vs_l1_next_open").mean()),
                "p50_entry_price_adv_vs_l1_next_open": float(numeric_column(valid, f"{variant}_price_adv_vs_l1_next_open").median()),
                "p10_entry_price_adv_vs_l1_next_open": float(numeric_column(valid, f"{variant}_price_adv_vs_l1_next_open").quantile(0.10)),
                "p90_entry_price_adv_vs_l1_next_open": float(numeric_column(valid, f"{variant}_price_adv_vs_l1_next_open").quantile(0.90)),
                "fixed_1d_net_avg": float(numeric_column(valid, f"{variant}_fixed_1d_net_return").mean()),
                "fixed_3d_net_avg": float(numeric_column(valid, f"{variant}_fixed_3d_net_return").mean()),
                "fixed_5d_net_avg": float(numeric_column(valid, f"{variant}_fixed_5d_net_return").mean()),
                "tp3_rate": float(bool_column(valid, f"{variant}_tp3_hit_5d").mean()),
                "tp5_rate": float(bool_column(valid, f"{variant}_tp5_hit_5d").mean()),
                "tp5_plan_net_avg": float(numeric_column(valid, f"{variant}_tp5_plan_net_return_5d").mean()),
                "mfe_5d_median": float(numeric_column(valid, f"{variant}_mfe_5d").median()),
                "mae_5d_median": float(numeric_column(valid, f"{variant}_mae_5d").median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_confirm_entries(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build confirmation-entry comparison summaries."""
    confirmed = events[events["tomorrow_l1"]].copy()
    if confirmed.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    overall = summarize_confirm_entry_group(confirmed, group="all", bucket="all")
    by_type_parts = [
        summarize_confirm_entry_group(group, group="pre_l1_type", bucket=str(bucket))
        for bucket, group in confirmed.groupby("pre_l1_type", sort=True)
    ]
    by_year_parts = [
        summarize_confirm_entry_group(group, group="confirm_year", bucket=str(year))
        for year, group in confirmed.groupby(pd.to_datetime(confirmed["confirm_date"]).dt.year, sort=True)
    ]
    by_type = pd.concat(by_type_parts, ignore_index=True) if by_type_parts else pd.DataFrame()
    by_year = pd.concat(by_year_parts, ignore_index=True) if by_year_parts else pd.DataFrame()
    return overall, by_type, by_year


def write_report(
    path: Path,
    *,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    by_year: pd.DataFrame,
    topk: pd.DataFrame,
    confirm_summary: pd.DataFrame,
    open_confirm_summary: pd.DataFrame,
    open_confirm_pass_by_year: pd.DataFrame,
    b1_summary: pd.DataFrame,
    b1_daily: pd.DataFrame,
) -> None:
    """Write a compact Markdown report."""
    lines = [
        "# PRE_L1_near_start Study",
        "",
        "Decision timing: T close. Entry timing: T+1 open. L1 labels are offline only.",
        "",
        f"- Events: `{len(events)}`",
        f"- Days: `{events['signal_date'].nunique() if not events.empty else 0}`",
        f"- Symbols: `{events['vt_symbol'].nunique() if not events.empty else 0}`",
        "",
        "## Overall",
        "",
    ]
    if summary.empty:
        lines.append("No events.")
    else:
        focus = summary[(summary["group"].eq("all")) & (summary["scope"].isin(["all", "main_gap_-3_+3", "tomorrow_l1", "not_tomorrow_l1"]))]
        lines.append("| Scope | Events | Tomorrow L1 | TP5 | TP5 Plan | MFE P50 | MAE P50 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in focus.to_dict("records"):
            lines.append(
                "| "
                f"{row['scope']} | {int(row['events'])} | {pct(float(row.get('tomorrow_l1_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_rate', math.nan)))} | {pct(float(row.get('tp5_plan_net_avg', math.nan)))} | "
                f"{pct(float(row.get('mfe_5d_median', math.nan)))} | {pct(float(row.get('mae_5d_median', math.nan)))} |"
            )
    lines.extend(["", "## Top-K Main Gap", ""])
    if topk.empty:
        lines.append("No top-K rows.")
    else:
        focus = topk[topk["return_name"].eq("tp5_plan_net_return_5d")]
        lines.append("| TopK | Days | Daily Avg | P10 | P50 | Positive Days | MDD | Tomorrow L1 | TP5 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in focus.to_dict("records"):
            lines.append(
                "| "
                f"{int(row['top_k'])} | {int(row['days'])} | {pct(float(row['daily_avg']))} | "
                f"{pct(float(row['daily_p10']))} | {pct(float(row['daily_p50']))} | "
                f"{pct(float(row['positive_day_rate']))} | {pct(float(row['max_drawdown']))} | "
                f"{pct(float(row['tomorrow_l1_rate']))} | {pct(float(row['tp5_rate']))} |"
            )
    lines.extend(["", "## By Year Main Gap", ""])
    if not by_year.empty:
        focus = by_year[(by_year["scope"].eq("main_gap_-3_+3")) & (by_year["group"].eq("year"))]
        lines.append("| Year | Events | Tomorrow L1 | TP5 | TP5 Plan | Fixed 3D |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in focus.to_dict("records"):
            lines.append(
                "| "
                f"{row['bucket']} | {int(row['events'])} | {pct(float(row.get('tomorrow_l1_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_rate', math.nan)))} | {pct(float(row.get('tp5_plan_net_avg', math.nan)))} | "
                f"{pct(float(row.get('fixed_3d_net_avg', math.nan)))} |"
            )
    lines.extend(["", "## T Open Confirmation", ""])
    if open_confirm_summary.empty:
        lines.append("No open-confirm rows.")
    else:
        focus = open_confirm_summary[
            open_confirm_summary["group"].isin(
                ["all_open_executable", "open_confirm_pass", "open_confirm_bucket", "open_gap_bucket"]
            )
        ]
        lines.append(
            "| Group | Bucket | Events | Tomorrow L1 | 1D TP3 | 1D TP5 | 1D MFE P50 | 1D MAE P50 | "
            "3D TP3 | 3D TP5 | 3D MFE P50 | 3D MAE P50 |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in focus.to_dict("records"):
            lines.append(
                "| "
                f"{row['group']} | {row['bucket']} | {int(row['events'])} | "
                f"{pct(float(row.get('tomorrow_l1_rate', math.nan)))} | "
                f"{pct(float(row.get('tp3_hit_1d_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_hit_1d_rate', math.nan)))} | "
                f"{pct(float(row.get('mfe_1d_median', math.nan)))} | "
                f"{pct(float(row.get('mae_1d_median', math.nan)))} | "
                f"{pct(float(row.get('tp3_hit_3d_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_hit_3d_rate', math.nan)))} | "
                f"{pct(float(row.get('mfe_3d_median', math.nan)))} | "
                f"{pct(float(row.get('mae_3d_median', math.nan)))} |"
            )
    lines.extend(["", "## T Open Confirm Pass By Year", ""])
    if not open_confirm_pass_by_year.empty:
        lines.append("| Year | Events | Tomorrow L1 | TP5 | TP5 Plan | Fixed 3D |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in open_confirm_pass_by_year.to_dict("records"):
            lines.append(
                "| "
                f"{row['bucket']} | {int(row['events'])} | "
                f"{pct(float(row.get('tomorrow_l1_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_plan_net_avg', math.nan)))} | "
                f"{pct(float(row.get('fixed_3d_net_avg', math.nan)))} |"
            )
    lines.extend(["", "## B1 Strong Open Break High", ""])
    if b1_summary.empty:
        lines.append("No B1 rows.")
    else:
        focus = b1_summary[b1_summary["group"].eq("segment")]
        lines.append(
            "| Segment | Events | Days | Gap P50 | Break P50 | 0D TP3 | 0D TP5 | 0D MFE P50 | 0D MAE P50 | "
            "3D TP3 | 3D TP5 | 3D MFE P50 | 3D MAE P50 | CloseFail TP3 | StopFirst TP3 |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in focus.to_dict("records"):
            lines.append(
                "| "
                f"{row['bucket']} | {int(row['events'])} | {int(row['days'])} | "
                f"{pct(float(row.get('entry_gap_p50', math.nan)))} | "
                f"{pct(float(row.get('break_high_p50', math.nan)))} | "
                f"{pct(float(row.get('tp3_hit_0d_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_hit_0d_rate', math.nan)))} | "
                f"{pct(float(row.get('mfe_0d_median', math.nan)))} | "
                f"{pct(float(row.get('mae_0d_median', math.nan)))} | "
                f"{pct(float(row.get('tp3_hit_3d_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_hit_3d_rate', math.nan)))} | "
                f"{pct(float(row.get('mfe_3d_median', math.nan)))} | "
                f"{pct(float(row.get('mae_3d_median', math.nan)))} | "
                f"{pct(float(row.get('close_fail_tp3_3d_avg', math.nan)))} | "
                f"{pct(float(row.get('low_stop_first_tp3_3d_avg', math.nan)))} |"
            )
    lines.extend(["", "## B1 Daily Returns", ""])
    if not b1_daily.empty:
        focus = b1_daily[
            b1_daily["return_name"].isin(
                [
                    "fixed_0d_net_return",
                    "fixed_3d_net_return",
                    "b1_close_fail_tp3_3d_net_return",
                    "b1_low_fail_stop_first_tp3_3d_net_return",
                ]
            )
        ]
        lines.append("| Segment | Return | Days | Daily Avg | P10 | P50 | Positive Days | MDD |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in focus.to_dict("records"):
            lines.append(
                "| "
                f"{row['segment']} | {row['return_name']} | {int(row['days'])} | "
                f"{pct(float(row['daily_avg']))} | {pct(float(row['daily_p10']))} | "
                f"{pct(float(row['daily_p50']))} | {pct(float(row['positive_day_rate']))} | "
                f"{pct(float(row['max_drawdown']))} |"
            )
    lines.extend(["", "## Confirmed Entry Comparison", ""])
    if confirm_summary.empty:
        lines.append("No tomorrow-L1 confirmation rows.")
    else:
        lines.append("| Entry | Events | Price vs L1 Next Open | P50 Price vs L1 | TP5 | TP5 Plan | Fixed 3D | MFE P50 | MAE P50 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in confirm_summary.to_dict("records"):
            lines.append(
                "| "
                f"{row['entry_variant']} | {int(row['events'])} | "
                f"{pct(float(row.get('avg_entry_price_adv_vs_l1_next_open', math.nan)))} | "
                f"{pct(float(row.get('p50_entry_price_adv_vs_l1_next_open', math.nan)))} | "
                f"{pct(float(row.get('tp5_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_plan_net_avg', math.nan)))} | "
                f"{pct(float(row.get('fixed_3d_net_avg', math.nan)))} | "
                f"{pct(float(row.get('mfe_5d_median', math.nan)))} | "
                f"{pct(float(row.get('mae_5d_median', math.nan)))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_b1_only_report(
    path: Path,
    *,
    b1_summary: pd.DataFrame,
    b1_daily: pd.DataFrame,
    b1_by_year: pd.DataFrame,
) -> None:
    """Write the standalone B1 report."""
    lines = [
        "# PRE_L1 B1 Strong Open Break High Study",
        "",
        "Decision timing: T close pre-L1 watchlist. Entry timing: T+1 open when open > T high.",
        "",
        "## Segments",
        "",
    ]
    if b1_summary.empty:
        lines.append("No B1 rows.")
    else:
        focus = b1_summary[b1_summary["group"].eq("segment")]
        lines.append(
            "| Segment | Events | Days | Gap P50 | Break P50 | 0D TP3 | 0D TP5 | "
            "3D TP3 | 3D TP5 | 3D MFE P50 | 3D MAE P50 | CloseFail TP3 | StopFirst TP3 |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in focus.to_dict("records"):
            lines.append(
                "| "
                f"{row['bucket']} | {int(row['events'])} | {int(row['days'])} | "
                f"{pct(float(row.get('entry_gap_p50', math.nan)))} | "
                f"{pct(float(row.get('break_high_p50', math.nan)))} | "
                f"{pct(float(row.get('tp3_hit_0d_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_hit_0d_rate', math.nan)))} | "
                f"{pct(float(row.get('tp3_hit_3d_rate', math.nan)))} | "
                f"{pct(float(row.get('tp5_hit_3d_rate', math.nan)))} | "
                f"{pct(float(row.get('mfe_3d_median', math.nan)))} | "
                f"{pct(float(row.get('mae_3d_median', math.nan)))} | "
                f"{pct(float(row.get('close_fail_tp3_3d_avg', math.nan)))} | "
                f"{pct(float(row.get('low_stop_first_tp3_3d_avg', math.nan)))} |"
            )

    lines.extend(["", "## Daily Returns", ""])
    if not b1_daily.empty:
        focus = b1_daily[
            b1_daily["return_name"].isin(
                [
                    "fixed_0d_net_return",
                    "fixed_3d_net_return",
                    "b1_close_fail_tp3_3d_net_return",
                    "b1_low_fail_stop_first_tp3_3d_net_return",
                ]
            )
        ]
        lines.append("| Segment | Return | Days | Daily Avg | P10 | P50 | Positive Days | MDD |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in focus.to_dict("records"):
            lines.append(
                "| "
                f"{row['segment']} | {row['return_name']} | {int(row['days'])} | "
                f"{pct(float(row['daily_avg']))} | {pct(float(row['daily_p10']))} | "
                f"{pct(float(row['daily_p50']))} | {pct(float(row['positive_day_rate']))} | "
                f"{pct(float(row['max_drawdown']))} |"
            )

    lines.extend(["", "## B1 Gap 0-3 By Year", ""])
    if not b1_by_year.empty:
        focus = b1_by_year[b1_by_year["segment"].eq("B1_gap_0_3")]
        lines.append("| Year | Events | 0D TP3 | 3D TP3 | 3D MFE P50 | 3D MAE P50 | CloseFail TP3 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in focus.to_dict("records"):
            lines.append(
                "| "
                f"{row['bucket']} | {int(row['events'])} | "
                f"{pct(float(row.get('tp3_hit_0d_rate', math.nan)))} | "
                f"{pct(float(row.get('tp3_hit_3d_rate', math.nan)))} | "
                f"{pct(float(row.get('mfe_3d_median', math.nan)))} | "
                f"{pct(float(row.get('mae_3d_median', math.nan)))} | "
                f"{pct(float(row.get('close_fail_tp3_3d_avg', math.nan)))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the PRE_L1_near_start study."""
    args = parse_args()
    lab_path = repo_path(args.lab_path)
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    config = TrendRsiConfig()
    l1_dates_by_symbol = load_l1_label_sets(repo_path(args.snapshot_path))

    rows: list[dict[str, Any]] = []
    files = select_paths(
        sorted((lab_path / "daily").glob("*.parquet")),
        symbol_list=args.symbol_list,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    if args.limit:
        files = files[: args.limit]
    for count, file_path in enumerate(files, start=1):
        vt_symbol = file_path.stem
        data = load_daily(file_path, start, end)
        if len(data) < args.min_bars:
            continue
        rows.extend(
            collect_symbol_pre_l1(
                vt_symbol,
                data,
                l1_dates=l1_dates_by_symbol.get(vt_symbol, set()),
                config=config,
                cooldown_bars=args.cooldown_bars,
                open_rate=args.open_rate,
                close_rate=args.close_rate,
                b1_only=args.b1_only,
            )
        )
        if args.progress_every and count % args.progress_every == 0:
            print(f"progress symbols={count}/{len(files)} events={len(rows)}", flush=True)

    events = pd.DataFrame(rows)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if events.empty:
        print("no PRE_L1_near_start events")
        return

    events["signal_date"] = normalize_date(events["signal_date"])
    events["entry_date"] = normalize_date(events["entry_date"])
    entry_panel = load_entry_panel(
        lab_path,
        start=events["entry_date"].min(),
        end=events["entry_date"].max(),
    )
    events = add_execution_fields(events, entry_panel)
    events = add_open_confirm_fields(events)
    events = add_b1_fields(events)
    summary, by_year = summarize_events(events)
    topk = summarize_topk(events)
    if args.b1_only:
        confirm_summary = pd.DataFrame()
        confirm_by_type = pd.DataFrame()
        confirm_by_year = pd.DataFrame()
    else:
        confirm_summary, confirm_by_type, confirm_by_year = summarize_confirm_entries(events)
    open_confirm_summary, open_confirm_by_year, open_confirm_pass_by_year = summarize_open_confirm(events)
    open_confirm_path_summary = build_open_confirm_path_summary(open_confirm_summary)
    b1_summary, b1_daily, b1_by_year = summarize_b1(events)

    if args.b1_only:
        b1_summary.to_csv(output_dir / "pre_l1_b1_strong_open_break_high_summary.csv", index=False)
        b1_daily.to_csv(output_dir / "pre_l1_b1_strong_open_break_high_daily.csv", index=False)
        b1_by_year.to_csv(output_dir / "pre_l1_b1_strong_open_break_high_by_year.csv", index=False)
        if args.write_events:
            events.to_parquet(output_dir / "pre_l1_b1_strong_open_break_high_events.parquet", index=False)
        events.head(2000).to_csv(output_dir / "pre_l1_b1_strong_open_break_high_events_sample.csv", index=False)
        write_b1_only_report(
            output_dir / "pre_l1_b1_strong_open_break_high_report.md",
            b1_summary=b1_summary,
            b1_daily=b1_daily,
            b1_by_year=b1_by_year,
        )
        if not b1_summary.empty:
            b1_focus = b1_summary[(b1_summary["group"].eq("segment")) & (b1_summary["bucket"].eq("B1_gap_0_3"))]
            if not b1_focus.empty:
                row = b1_focus.iloc[0]
                print(
                    "PRE_L1_B1 "
                    f"segment={row['bucket']} events={int(row['events'])} "
                    f"tp3_0d={pct(float(row['tp3_hit_0d_rate']))} "
                    f"tp5_0d={pct(float(row['tp5_hit_0d_rate']))} "
                    f"close_fail_tp3_3d={pct(float(row['close_fail_tp3_3d_avg']))} "
                    f"stop_first_tp3_3d={pct(float(row['low_stop_first_tp3_3d_avg']))}",
                    flush=True,
                )
        print(f"wrote {output_dir}")
        return

    summary.to_csv(output_dir / "pre_l1_near_start_summary.csv", index=False)
    by_year.to_csv(output_dir / "pre_l1_near_start_by_year.csv", index=False)
    topk.to_csv(output_dir / "pre_l1_near_start_topk.csv", index=False)
    confirm_summary.to_csv(output_dir / "pre_l1_confirm_entry_summary.csv", index=False)
    confirm_by_type.to_csv(output_dir / "pre_l1_confirm_entry_by_type.csv", index=False)
    confirm_by_year.to_csv(output_dir / "pre_l1_confirm_entry_by_year.csv", index=False)
    open_confirm_summary.to_csv(output_dir / "pre_l1_open_confirm_summary.csv", index=False)
    open_confirm_by_year.to_csv(output_dir / "pre_l1_open_confirm_by_year.csv", index=False)
    open_confirm_pass_by_year.to_csv(output_dir / "pre_l1_open_confirm_pass_by_year.csv", index=False)
    open_confirm_path_summary.to_csv(output_dir / "pre_l1_open_confirm_path_summary.csv", index=False)
    b1_summary.to_csv(output_dir / "pre_l1_b1_strong_open_break_high_summary.csv", index=False)
    b1_daily.to_csv(output_dir / "pre_l1_b1_strong_open_break_high_daily.csv", index=False)
    b1_by_year.to_csv(output_dir / "pre_l1_b1_strong_open_break_high_by_year.csv", index=False)
    if args.write_events:
        events.to_parquet(output_dir / "pre_l1_near_start_events.parquet", index=False)
        events.head(2000).to_csv(output_dir / "pre_l1_near_start_events_sample.csv", index=False)
    else:
        events.head(2000).to_csv(output_dir / "pre_l1_near_start_events_sample.csv", index=False)
    write_report(
        output_dir / "pre_l1_near_start_report.md",
        events=events,
        summary=summary,
        by_year=by_year,
        topk=topk,
        confirm_summary=confirm_summary,
        open_confirm_summary=open_confirm_summary,
        open_confirm_pass_by_year=open_confirm_pass_by_year,
        b1_summary=b1_summary,
        b1_daily=b1_daily,
    )

    focus = summary[(summary["scope"].eq("main_gap_-3_+3")) & (summary["group"].eq("all"))]
    if not focus.empty:
        row = focus.iloc[0]
        print(
            "PRE_L1_near_start "
            f"events={len(events)} main_gap_events={int(row['events'])} "
            f"tomorrow_l1={pct(float(row['tomorrow_l1_rate']))} "
            f"tp5={pct(float(row['tp5_rate']))} "
            f"tp5_plan={pct(float(row['tp5_plan_net_avg']))}",
            flush=True,
        )
    if not confirm_summary.empty:
        best = confirm_summary.sort_values("fixed_3d_net_avg", ascending=False).iloc[0]
        print(
            "PRE_L1_confirm_entry "
            f"best={best['entry_variant']} "
            f"price_adv={pct(float(best['avg_entry_price_adv_vs_l1_next_open']))} "
            f"fixed3={pct(float(best['fixed_3d_net_avg']))} "
            f"tp5_plan={pct(float(best['tp5_plan_net_avg']))}",
            flush=True,
        )
    if not open_confirm_summary.empty:
        pass_rows = open_confirm_summary[open_confirm_summary["group"].eq("open_confirm_pass")]
        if not pass_rows.empty:
            row = pass_rows.iloc[0]
            print(
                "PRE_L1_open_confirm "
                f"events={int(row['events'])} "
                f"tomorrow_l1={pct(float(row['tomorrow_l1_rate']))} "
                f"fixed3={pct(float(row['fixed_3d_net_avg']))} "
                f"tp5_plan={pct(float(row['tp5_plan_net_avg']))}",
                flush=True,
            )
    if not b1_summary.empty:
        b1_focus = b1_summary[(b1_summary["group"].eq("segment")) & (b1_summary["bucket"].eq("B1_gap_0_3"))]
        if not b1_focus.empty:
            row = b1_focus.iloc[0]
            print(
                "PRE_L1_B1 "
                f"segment={row['bucket']} events={int(row['events'])} "
                f"tp3_0d={pct(float(row['tp3_hit_0d_rate']))} "
                f"tp5_0d={pct(float(row['tp5_hit_0d_rate']))} "
                f"close_fail_tp3_3d={pct(float(row['close_fail_tp3_3d_avg']))} "
                f"stop_first_tp3_3d={pct(float(row['low_stop_first_tp3_3d_avg']))}",
                flush=True,
            )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
