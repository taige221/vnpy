"""Generate daily L1 execution-v1 candidate lists.

The script implements the frozen research-v1 workflow:

- after close: L1 + core4_mean >= 0.52, ranked by core4_short70_30;
- signal-close ranked pool: top20;
- after open: apply gap, limit, liquidity, and industry controls, then pick
  executable top1/top2 from the signal-close top20 while preserving that rank
  order.

Historical snapshots include next-open fields, so the script can also produce
a next-open selection for past dates. In live use, if next-open fields are not
available yet, the script still produces the signal-close ranked pool and marks
execution as pending.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.alpha_research.tradingview.signal_core.buy_points import add_buy_point_columns


DEFAULT_CONFIG_PATH = "scripts/alpha_research/tradingview/configs/l1_execution_v1.json"
DEFAULT_OUTPUT_DIR = (
    "scripts/alpha_research/tradingview/reports/"
    "daily_candidates"
)
SIGNAL_CLOSE_FORBIDDEN_PREFIXES = ("future_", "next_")
SIGNAL_CLOSE_FORBIDDEN_COLUMNS = {
    "tradable_flag",
    "tradable_reason",
    "suggested_entry_date",
    "suggested_entry_price",
    "execution_bucket",
    "open_executable",
    "next_open_executable",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate L1 v1 daily candidates")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--snapshot-kind",
        choices=["live", "research"],
        default="live",
        help="Use the live-safe snapshot by default; use research for historical next-open fields.",
    )
    parser.add_argument("--snapshot-path", default=None)
    parser.add_argument(
        "--daily-filter-path",
        default=None,
        help="Deprecated. Signal-close day filter is computed from the snapshot.",
    )
    parser.add_argument("--panel-path", default="lab/a_share_research/panel/research_panel_daily")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--signal-date", default=None, help="YYYY-MM-DD. Defaults to latest eligible day.")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--execution-top-k", type=int, default=None)
    parser.add_argument(
        "--live-open-source",
        choices=["none", "tencent"],
        default="none",
        help="Runtime-only T+1 open quote overlay. Data is not written back to snapshots.",
    )
    parser.add_argument("--include-below-threshold", action="store_true")
    parser.add_argument("--write-all-candidates", action="store_true")
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    """Return a boolean series from a possibly missing column."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    return frame[column].astype("boolean").fillna(default).astype(bool)


def normalize_date(values: Any) -> pd.Series:
    """Normalize date-like values to midnight timestamps."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


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


def pct(value: Any) -> str:
    """Format a decimal as percent."""
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value) * 100:.2f}%"


def parse_float(value: str) -> float:
    """Parse Tencent numeric fields."""
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def load_config(path: str) -> dict[str, Any]:
    """Load the frozen v1 config."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


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


def load_snapshot(path: str) -> pd.DataFrame:
    """Load and normalize the signal snapshot."""
    snapshot = pd.read_parquet(path)
    snapshot["signal_date"] = normalize_date(snapshot["signal_date"])
    return add_buy_point_columns(snapshot)


def resolve_snapshot_path(args: argparse.Namespace, config: dict[str, Any]) -> str:
    """Resolve the configured snapshot path."""
    if args.snapshot_path:
        return str(args.snapshot_path)
    if args.snapshot_kind == "live":
        return str(config.get("live_signal_snapshot") or config["source_snapshot"])
    return str(config.get("research_snapshot") or config["source_snapshot"])


def validate_live_snapshot_columns(snapshot: pd.DataFrame, *, path: str) -> None:
    """Reject post-signal fields from live snapshots."""
    blocked = [
        column
        for column in snapshot.columns
        if column in SIGNAL_CLOSE_FORBIDDEN_COLUMNS
        or column.startswith(SIGNAL_CLOSE_FORBIDDEN_PREFIXES)
        or column.startswith("risk_")
    ]
    if blocked:
        raise ValueError(
            f"live snapshot contains post-signal columns from {path}: {sorted(blocked)}. "
            "Use --snapshot-kind research for historical studies."
        )


def signal_close_l1_mask(snapshot: pd.DataFrame, *, buy_point_type: str) -> pd.Series:
    """Return the T-close-known L1 candidate mask."""
    used_columns = [
        "signal_date",
        "buy_point_type",
        "has_valid_bar",
        "is_list_life_valid",
        "is_st",
        "is_new_stock",
    ]
    assert_signal_close_columns(used_columns, stage="signal_close_day_filter")
    return (
        snapshot["buy_point_type"].eq(buy_point_type)
        & bool_series(snapshot, "has_valid_bar", default=True)
        & bool_series(snapshot, "is_list_life_valid", default=True)
        & ~bool_series(snapshot, "is_st")
        & ~bool_series(snapshot, "is_new_stock")
    )


def build_signal_close_day_filter(
    snapshot: pd.DataFrame,
    *,
    buy_point_type: str,
    rank_column: str,
) -> pd.DataFrame:
    """Build clean T-close day filter metrics without next/future fields."""
    assert_signal_close_columns([rank_column], stage="signal_close_day_filter_score")
    pool = snapshot[signal_close_l1_mask(snapshot, buy_point_type=buy_point_type)].copy()
    daily = (
        pool.groupby("signal_date", sort=True)
        .agg(
            signal_year=("signal_date", lambda values: int(pd.Timestamp(values.iloc[0]).year)),
            core4_mean=(rank_column, "mean"),
            signal_close_l1_count=("vt_symbol", "count"),
            signal_close_symbol_count=("vt_symbol", "nunique"),
        )
        .reset_index()
    )
    daily["core4_source"] = f"signal_close_l1_{rank_column}_mean"
    return daily.dropna(subset=["signal_date"])


def resolve_signal_date(
    daily: pd.DataFrame,
    *,
    signal_date: str | None,
    threshold: float,
    include_below_threshold: bool,
) -> pd.Timestamp:
    """Resolve the target signal date."""
    if signal_date:
        return pd.Timestamp(signal_date).normalize()
    frame = daily if include_below_threshold else daily[daily["core4_mean"] >= threshold]
    if frame.empty:
        raise ValueError("no eligible signal date found")
    return pd.Timestamp(frame["signal_date"].max()).normalize()


def load_snapshot_day(
    snapshot: pd.DataFrame,
    signal_date: pd.Timestamp,
    *,
    buy_point_type: str,
) -> pd.DataFrame:
    """Load one signal day's T-close-known L1 snapshot rows."""
    mask = snapshot["signal_date"].eq(signal_date) & signal_close_l1_mask(
        snapshot,
        buy_point_type=buy_point_type,
    )
    day = snapshot[mask].copy()
    for column in day.columns:
        if column.endswith("_score") or column in {
            "rank_score_core4_short70_30",
            "rank_score_short_tp_v1",
            "rank_score_core4_control",
            "rank_score_pvcorr60_15pct",
            "candidate_pool_score",
            "start_score",
            "circ_mv",
            "turnover_rate",
            "future_entry_gap_pct",
            "vol_ratio",
            "close_position",
            "rsi",
        }:
            day[column] = numeric(day[column])
    for column in ("next_is_limit_up", "next_is_limit_down", "next_has_valid_bar"):
        if column in day.columns:
            day[column] = day[column].astype("boolean").fillna(False).astype(bool)
        else:
            day[column] = False
    return day


def add_industry(day: pd.DataFrame, panel_path: str) -> pd.DataFrame:
    """Join SW level-1 industry names."""
    if day.empty:
        day["sw_l1_name"] = pd.NA
        return day
    if "sw_l1_name" in day.columns and day["sw_l1_name"].notna().any():
        out = day.copy()
        out["sw_l1_name"] = out["sw_l1_name"].fillna("UNKNOWN")
        return out
    years = sorted(pd.to_datetime(day["signal_date"]).dt.year.astype(int).unique())
    industry = load_industry_mapping(panel_path, years)
    if industry.empty:
        out = day.copy()
        out["sw_l1_name"] = "UNKNOWN"
        return out
    out = day.merge(industry, on=["signal_date", "vt_symbol"], how="left")
    out["sw_l1_name"] = out["sw_l1_name"].fillna("UNKNOWN")
    return out


def classify_day_quality(core4_mean: float, threshold: float, strong_threshold: float) -> str:
    """Classify the day quality bucket."""
    if not math.isfinite(core4_mean):
        return "unknown"
    if core4_mean >= strong_threshold:
        return "strong_candidate"
    if core4_mean >= threshold:
        return "candidate"
    return "below_threshold"


def rank_candidates(
    day: pd.DataFrame,
    *,
    rank_column: str,
    top_n: int,
    write_all: bool,
) -> pd.DataFrame:
    """Rank candidates by the frozen rank score."""
    if day.empty:
        return day.copy()
    fallback = [
        column
        for column in ("start_score", "rank_score_short_tp_v1", "candidate_pool_score")
        if column in day.columns and column != rank_column
    ]
    assert_signal_close_columns([rank_column, *fallback], stage="signal_close_rank_pool")
    ranked = (
        day.dropna(subset=[rank_column])
        .sort_values([rank_column] + fallback + ["vt_symbol"], ascending=[False] * (1 + len(fallback)) + [True])
        .copy()
    )
    ranked["signal_close_rank"] = np.arange(1, len(ranked) + 1)
    ranked["overnight_rank"] = ranked["signal_close_rank"]
    ranked["signal_close_pool_label"] = f"signal_close_top{top_n}"
    ranked["signal_close_pool_member"] = ranked["signal_close_rank"] <= top_n
    ranked["signal_close_rank_pool_member"] = ranked["signal_close_pool_member"]
    ranked["selection_stage"] = "signal_close_rank"
    if not write_all:
        ranked = ranked[ranked["signal_close_pool_member"]].copy()
    return ranked


def to_tencent_symbol(vt_symbol: str) -> str:
    """Convert a vt_symbol to a Tencent quote symbol."""
    code = str(vt_symbol).split(".")[0]
    exchange = str(vt_symbol).split(".")[-1] if "." in str(vt_symbol) else ""
    if exchange == "SSE":
        return f"sh{code}"
    if exchange == "SZSE":
        return f"sz{code}"
    if exchange == "BSE":
        return f"bj{code}"
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith("8"):
        return f"bj{code}"
    return f"sz{code}"


def fetch_tencent_quotes(vt_symbols: list[str]) -> pd.DataFrame:
    """Fetch runtime Tencent quotes for the selected symbols."""
    symbol_map = {to_tencent_symbol(vt_symbol): vt_symbol for vt_symbol in vt_symbols}
    if not symbol_map:
        return pd.DataFrame(columns=["vt_symbol"])
    url = "http://qt.gtimg.cn/q=" + ",".join(symbol_map)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        raw = response.read().decode("gbk", errors="ignore")

    rows: list[dict[str, Any]] = []
    for line in raw.strip().split(";"):
        if not line.strip() or "=" not in line or '"' not in line:
            continue
        quote_symbol = line.split("=")[0].split("_")[-1]
        vt_symbol = symbol_map.get(quote_symbol)
        if not vt_symbol:
            continue
        values = line.split('"')[1].split("~")
        if len(values) < 53:
            continue
        last_close = parse_float(values[4])
        open_price = parse_float(values[5])
        limit_up = parse_float(values[47])
        limit_down = parse_float(values[48])
        gap_pct = (
            open_price / last_close - 1
            if math.isfinite(open_price) and math.isfinite(last_close) and last_close > 0
            else math.nan
        )
        rows.append(
            {
                "vt_symbol": vt_symbol,
                "live_quote_symbol": quote_symbol,
                "live_quote_time": values[30] if len(values) > 30 else "",
                "live_name": values[1],
                "live_open_price": open_price,
                "live_last_close": last_close,
                "live_current_price": parse_float(values[3]),
                "live_change_pct": parse_float(values[32]),
                "live_high": parse_float(values[33]),
                "live_low": parse_float(values[34]),
                "live_amount_wan": parse_float(values[37]),
                "turnover_rate": parse_float(values[38]),
                "live_pe_ttm": parse_float(values[39]),
                "live_amplitude_pct": parse_float(values[43]),
                "live_mcap_yi": parse_float(values[44]),
                "live_float_mcap_yi": parse_float(values[45]),
                "live_pb": parse_float(values[46]),
                "live_limit_up_price": limit_up,
                "live_limit_down_price": limit_down,
                "vol_ratio": parse_float(values[49]),
                "next_open_gap_pct": gap_pct,
                "next_has_valid_bar": math.isfinite(open_price) and open_price > 0 and last_close > 0,
                "next_is_limit_up": (
                    math.isfinite(open_price)
                    and math.isfinite(limit_up)
                    and limit_up > 0
                    and open_price >= limit_up * 0.999
                ),
                "next_is_limit_down": (
                    math.isfinite(open_price)
                    and math.isfinite(limit_down)
                    and limit_down > 0
                    and open_price <= limit_down * 1.001
                ),
                "circ_mv": parse_float(values[45]) * 10000,
                "execution_data_source": "live_tencent_quote",
            }
        )
    return pd.DataFrame(rows)


def apply_live_open_overlay(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Overlay runtime T+1 open fields without mutating persistent snapshots."""
    out = frame.copy()
    if source == "none" or out.empty:
        if "execution_data_source" not in out.columns:
            has_snapshot_open_fields = (
                ("next_open_gap_pct" in out.columns and out["next_open_gap_pct"].notna().any())
                or ("future_entry_gap_pct" in out.columns and out["future_entry_gap_pct"].notna().any())
            )
            out["execution_data_source"] = (
                "snapshot_next_open_fields" if has_snapshot_open_fields else "pending_no_open_overlay"
            )
        return out
    if source != "tencent":
        raise ValueError(f"unsupported live open source: {source}")

    quotes = fetch_tencent_quotes(out["vt_symbol"].astype(str).tolist())
    if quotes.empty:
        out["execution_data_source"] = "live_tencent_quote_missing"
        return out
    out = out.merge(quotes, on="vt_symbol", how="left", suffixes=("", "_live"))
    overlay_columns = [
        "circ_mv",
        "turnover_rate",
        "vol_ratio",
        "next_open_gap_pct",
        "next_has_valid_bar",
        "next_is_limit_up",
        "next_is_limit_down",
    ]
    for column in overlay_columns:
        live_column = f"{column}_live"
        if live_column in out.columns:
            out[column] = out[live_column]
            out = out.drop(columns=[live_column])
    if "execution_data_source_live" in out.columns:
        out["execution_data_source"] = out["execution_data_source_live"]
        out = out.drop(columns=["execution_data_source_live"])
    for column in ("next_has_valid_bar", "next_is_limit_up", "next_is_limit_down"):
        out[column] = out[column].astype("boolean").fillna(False).astype(bool)
    out["execution_data_source"] = out["execution_data_source"].fillna("live_tencent_quote_missing")
    return out


def add_execution_status(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add next-open execution status fields."""
    out = frame.copy()
    entry = config["entry_execution"]
    tradable = config["tradability_filters"]
    if "next_open_gap_pct" in out.columns and out["next_open_gap_pct"].notna().any():
        gap = numeric(out["next_open_gap_pct"])
        out["execution_gap_field"] = "next_open_gap_pct"
    else:
        gap = numeric(out.get(entry["main_gap_field"], pd.Series(index=out.index, dtype=float)))
        out["execution_gap_field"] = entry["main_gap_field"]
    out["next_open_gap_pct"] = gap
    circ_mv = numeric(out.get("circ_mv", pd.Series(index=out.index, dtype=float)))
    turnover = numeric(out.get("turnover_rate", pd.Series(index=out.index, dtype=float)))
    out["is_bse"] = out["vt_symbol"].astype(str).str.endswith(".BSE")
    out["main_gap_ok"] = (gap >= entry["main_gap_min_inclusive"]) & (gap < entry["main_gap_max_exclusive"])
    out["watch_gap_ok"] = (gap >= entry["watch_gap_min_inclusive"]) & (gap < entry["watch_gap_max_exclusive"])
    out["liquidity_ok"] = (
        ~out["is_bse"]
        & (circ_mv >= float(tradable["min_circ_mv"]))
        & (turnover >= float(tradable["min_turnover_rate"]))
    )
    out["limit_state_ok"] = out["next_has_valid_bar"] & ~out["next_is_limit_up"] & ~out["next_is_limit_down"]
    out["execution_bucket"] = np.select(
        [
            gap.isna(),
            ~out["next_has_valid_bar"],
            out["next_is_limit_up"],
            out["next_is_limit_down"],
            out["is_bse"],
            ~out["liquidity_ok"],
            out["main_gap_ok"] & out["limit_state_ok"] & out["liquidity_ok"],
            out["watch_gap_ok"] & out["limit_state_ok"] & out["liquidity_ok"],
            gap >= entry["reject_gap_min_inclusive"],
            gap < entry["main_gap_min_inclusive"],
        ],
        [
            "pending_next_open",
            "no_next_bar",
            "next_limit_up",
            "next_limit_down",
            "bse_excluded",
            "liquidity_fail",
            "main_execute",
            "watch_only",
            "reject_high_gap",
            "reject_low_gap",
        ],
        default="reject_gap_other",
    )
    out["open_executable"] = out["execution_bucket"].eq("main_execute")
    out["next_open_executable"] = out["open_executable"]
    return out


def select_executable(
    frame: pd.DataFrame,
    *,
    top_k: int,
    industry_cap: float,
    source_pool_label: str,
) -> pd.DataFrame:
    """Select top executable rows while preserving signal-close rank order."""
    executable = frame[frame["open_executable"]].sort_values("signal_close_rank").copy()
    max_per_industry = max(1, int(math.ceil(top_k * industry_cap)))
    selected_indexes: list[int] = []
    industry_counts: dict[str, int] = {}
    for index, row in executable.iterrows():
        industry = str(row.get("sw_l1_name", "UNKNOWN") or "UNKNOWN")
        if industry_counts.get(industry, 0) >= max_per_industry:
            continue
        selected_indexes.append(int(index))
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected_indexes) >= top_k:
            break
    selected = executable.loc[selected_indexes].copy()
    selected["next_open_execution_rank"] = np.arange(1, len(selected) + 1)
    selected["execution_rank"] = selected["next_open_execution_rank"]
    selected["next_open_selection_label"] = f"next_open_select_top{top_k}_from_{source_pool_label}"
    selected["next_open_selected"] = True
    selected["selection_stage"] = "next_open_execution_select"
    return selected


def add_reason_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add compact reader-facing reason/risk strings."""
    out = frame.copy()
    out["rank_reason"] = out.apply(
        lambda row: "; ".join(
            str(item)
            for item in [
                f"signal_close_rank={int(row['signal_close_rank'])}"
                if pd.notna(row.get("signal_close_rank"))
                else "",
                f"score={row.get('rank_score_core4_short70_30', math.nan):.4f}"
                if pd.notna(row.get("rank_score_core4_short70_30"))
                else "",
                str(row.get("buy_point_label", "")),
                str(row.get("buy_point_subtype", "")),
                str(row.get("start_type", "")),
                f"grade={row.get('start_grade', '')}",
                str(row.get("vol_quality_bucket", "")),
                str(row.get("market_stage_bucket", "")),
                str(row.get("industry_rs_bucket", "")),
            ]
            if item
        ),
        axis=1,
    )
    out["risk_tags"] = out.apply(build_risk_tags, axis=1)
    return out


def build_risk_tags(row: pd.Series) -> str:
    """Build a compact risk tag list."""
    tags: list[str] = []
    if row.get("execution_bucket") != "main_execute":
        tags.append(str(row.get("execution_bucket", "")))
    if str(row.get("market_stage_bucket", "")).startswith(("Q1", "Q2")):
        tags.append("weak_market")
    if str(row.get("industry_rs_bucket", "")).startswith(("Q0", "Q1", "Q2")):
        tags.append("weak_industry")
    if str(row.get("vol_quality_bucket", "")).startswith("C_"):
        tags.append("weak_volume_quality")
    if pd.notna(row.get("next_open_gap_pct")) and abs(float(row["next_open_gap_pct"])) >= 0.03:
        tags.append("large_open_gap")
    return ",".join(tag for tag in tags if tag)


def output_columns(frame: pd.DataFrame) -> list[str]:
    """Return stable output columns that exist in frame."""
    columns = [
        "signal_date",
        "signal_close_rank",
        "overnight_rank",
        "signal_close_pool_label",
        "signal_close_pool_member",
        "signal_close_rank_pool_member",
        "next_open_selection_label",
        "next_open_executable",
        "next_open_selected",
        "next_open_execution_rank",
        "execution_rank",
        "vt_symbol",
        "name",
        "sw_l1_name",
        "buy_point_type",
        "buy_point_label",
        "buy_point_subtype",
        "rank_score_core4_short70_30",
        "start_score",
        "start_grade",
        "start_type",
        "day_quality",
        "day_filter_name",
        "day_filter_pass",
        "signal_close_day_filter_pass",
        "core4_mean",
        "core4_source",
        "candidate_scope",
        "execution_data_source",
        "execution_gap_field",
        "live_quote_time",
        "live_open_price",
        "live_last_close",
        "live_current_price",
        "live_limit_up_price",
        "live_limit_down_price",
        "live_amount_wan",
        "live_mcap_yi",
        "live_float_mcap_yi",
        "execution_bucket",
        "next_open_gap_pct",
        "future_entry_gap_pct",
        "circ_mv",
        "turnover_rate",
        "vol_ratio",
        "vol_quality_bucket",
        "market_stage_bucket",
        "industry_rs_bucket",
        "rank_reason",
        "risk_tags",
    ]
    return [column for column in columns if column in frame.columns]


def write_markdown(
    path: Path,
    *,
    signal_date: pd.Timestamp,
    daily_row: pd.Series | None,
    config: dict[str, Any],
    snapshot_kind: str,
    snapshot_path: str,
    top_n: int,
    execution_top_k: int,
    watch: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    """Write a short Markdown candidate report."""
    quality = watch["day_quality"].iloc[0] if not watch.empty and "day_quality" in watch else "unknown"
    core4_mean = watch["core4_mean"].iloc[0] if not watch.empty and "core4_mean" in watch else math.nan
    lines = [
        f"# L1 Execution V1 Daily Candidates - {signal_date.date()}",
        "",
        f"- Day quality: `{quality}`",
        f"- core4_mean: `{core4_mean:.4f}`" if math.isfinite(float(core4_mean)) else "- core4_mean: `nan`",
        f"- core4_source: `{watch['core4_source'].iloc[0]}`"
        if not watch.empty and "core4_source" in watch
        else "- core4_source: `unknown`",
        f"- candidate_scope: `{watch['candidate_scope'].iloc[0]}`"
        if not watch.empty and "candidate_scope" in watch
        else "- candidate_scope: `unknown`",
        f"- day_filter_name: `{watch['day_filter_name'].iloc[0]}`"
        if not watch.empty and "day_filter_name" in watch
        else "- day_filter_name: `unknown`",
        f"- Rank: `{config['portfolio_controls']['ranking_default']}`",
        f"- Snapshot kind: `{snapshot_kind}`",
        f"- Snapshot path: `{snapshot_path}`",
        f"- Signal-close ranked pool: `top{top_n}`",
        f"- Next-open selection: `top{execution_top_k} from signal-close top{top_n}`",
        f"- Execution data source: `{watch['execution_data_source'].iloc[0]}`"
        if not watch.empty and "execution_data_source" in watch
        else "- Execution data source: `unknown`",
        "",
        "## Next-Open Selection",
        "",
    ]
    if selected.empty:
        lines.append("No executable selection under the v1 main execution rules.")
    else:
        lines.append("| Next Open | Signal Close | Symbol | Name | Industry | Score | Gap | Tags |")
        lines.append("| ---: | ---: | --- | --- | --- | ---: | ---: | --- |")
        for row in selected.to_dict("records"):
            lines.append(
                "| "
                f"{int(row.get('next_open_execution_rank', row.get('execution_rank', 0)))} | "
                f"{int(row.get('signal_close_rank', row.get('overnight_rank', 0)))} | "
                f"`{row.get('vt_symbol', '')}` | "
                f"{row.get('name', '')} | "
                f"{row.get('sw_l1_name', '')} | "
                f"{float(row.get('rank_score_core4_short70_30', math.nan)):.4f} | "
                f"{pct(row.get('next_open_gap_pct', math.nan))} | "
                f"{row.get('risk_tags', '')} |"
            )
    lines.extend(["", "## Signal-Close Ranked Pool", ""])
    if watch.empty:
        lines.append("No signal-close candidates.")
    else:
        lines.append("| Rank | Symbol | Name | Industry | Score | Start | Exec Bucket | Tags |")
        lines.append("| ---: | --- | --- | --- | ---: | --- | --- | --- |")
        for row in watch.head(top_n).to_dict("records"):
            buy_point_text = row.get("buy_point_label", "") or row.get("buy_point_type", "")
            subtype_text = row.get("buy_point_subtype", "") or row.get("start_type", "")
            lines.append(
                "| "
                f"{int(row.get('signal_close_rank', row.get('overnight_rank', 0)))} | "
                f"`{row.get('vt_symbol', '')}` | "
                f"{row.get('name', '')} | "
                f"{row.get('sw_l1_name', '')} | "
                f"{float(row.get('rank_score_core4_short70_30', math.nan)):.4f} | "
                f"{buy_point_text}/{subtype_text}/{row.get('start_grade', '')} | "
                f"{row.get('execution_bucket', '')} | "
                f"{row.get('risk_tags', '')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate the candidate files."""
    args = parse_args()
    config = load_config(args.config_path)
    snapshot_path = resolve_snapshot_path(args, config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(config["candidate_day_filter"]["value"])
    strong_threshold = 0.54
    for label in config.get("day_quality_labels", []):
        if label.get("label") == "strong_candidate":
            strong_threshold = float(str(label["condition"]).split(">=")[-1].strip())

    top_n = args.top_n or int(
        config["portfolio_controls"].get(
            "signal_close_rank_pool_size",
            config["portfolio_controls"]["overnight_pool_size"],
        )
    )
    execution_top_k = args.execution_top_k or int(
        config["portfolio_controls"].get(
            "next_open_execution_top_k",
            config["portfolio_controls"]["execution_top_k"],
        )
    )
    rank_column = config["portfolio_controls"]["ranking_column"]
    source_pool_label = f"signal_close_top{top_n}"
    buy_point_type = config["signal"]["buy_point_type"]

    snapshot = load_snapshot(snapshot_path)
    if args.snapshot_kind == "live":
        validate_live_snapshot_columns(snapshot, path=snapshot_path)
    daily = build_signal_close_day_filter(
        snapshot,
        buy_point_type=buy_point_type,
        rank_column=rank_column,
    )
    signal_date = resolve_signal_date(
        daily,
        signal_date=args.signal_date,
        threshold=threshold,
        include_below_threshold=args.include_below_threshold,
    )
    daily_match = daily[daily["signal_date"].eq(signal_date)]
    daily_row = daily_match.iloc[0] if not daily_match.empty else None
    core4_mean = float(daily_row["core4_mean"]) if daily_row is not None else math.nan
    core4_source = str(daily_row["core4_source"]) if daily_row is not None else "missing"

    day = load_snapshot_day(snapshot, signal_date, buy_point_type=buy_point_type)
    candidate_scope = "signal_close_l1_rows"
    day = add_industry(day, args.panel_path)
    if day.empty:
        raise ValueError(f"no L1 candidates for {signal_date.date()}")
    if not args.include_below_threshold and (not math.isfinite(core4_mean) or core4_mean < threshold):
        raise ValueError(f"{signal_date.date()} core4_mean={core4_mean:.4f} below v1 threshold {threshold}")
    day["core4_mean"] = core4_mean
    day["core4_source"] = core4_source
    day["candidate_scope"] = candidate_scope
    day["day_filter_name"] = "signal_close_day_filter"
    day["day_filter_pass"] = bool(math.isfinite(core4_mean) and core4_mean >= threshold)
    day["signal_close_day_filter_pass"] = day["day_filter_pass"]
    day["day_quality"] = classify_day_quality(core4_mean, threshold, strong_threshold)
    ranked = rank_candidates(day, rank_column=rank_column, top_n=top_n, write_all=args.write_all_candidates)
    ranked = apply_live_open_overlay(ranked, args.live_open_source)
    ranked = add_execution_status(ranked, config)
    ranked["next_open_selection_label"] = f"next_open_select_top{execution_top_k}_from_{source_pool_label}"
    ranked["next_open_selected"] = False
    ranked = add_reason_columns(ranked)
    selected = select_executable(
        ranked,
        top_k=execution_top_k,
        industry_cap=float(config["portfolio_controls"]["industry_cap_default"]),
        source_pool_label=source_pool_label,
    )
    if not selected.empty:
        ranked.loc[selected.index, "next_open_selected"] = True

    date_str = signal_date.strftime("%Y-%m-%d")
    live_suffix = "" if args.live_open_source == "none" else f"_live_{args.live_open_source}"
    base_name = f"l1_execution_v1_{date_str}{live_suffix}"
    watch_path = output_dir / f"{base_name}_signal_close_top{top_n}.csv"
    selected_path = output_dir / f"{base_name}_next_open_from_signal_close_top{top_n}_top{execution_top_k}.csv"
    report_path = output_dir / f"{base_name}_report.md"

    ranked[output_columns(ranked)].to_csv(watch_path, index=False)
    selected[output_columns(selected)].to_csv(selected_path, index=False)
    write_markdown(
        report_path,
        signal_date=signal_date,
        daily_row=daily_row,
        config=config,
        snapshot_kind=args.snapshot_kind,
        snapshot_path=snapshot_path,
        top_n=top_n,
        execution_top_k=execution_top_k,
        watch=ranked,
        selected=selected,
    )

    print(
        f"signal_date={date_str} core4_mean={core4_mean:.4f} "
        f"day_quality={ranked['day_quality'].iloc[0]} snapshot_kind={args.snapshot_kind}"
    )
    print(f"snapshot_path={snapshot_path}")
    print(f"watch_candidates={len(ranked)} executable={int(ranked['open_executable'].sum())} selected={len(selected)}")
    print(f"wrote {watch_path}")
    print(f"wrote {selected_path}")
    print(f"wrote {report_path}")
    if not selected.empty:
        print(
            selected[
                [
                    "next_open_execution_rank",
                    "signal_close_rank",
                    "vt_symbol",
                    "name",
                    "rank_score_core4_short70_30",
                    "execution_bucket",
                    "next_open_gap_pct",
                    "risk_tags",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
