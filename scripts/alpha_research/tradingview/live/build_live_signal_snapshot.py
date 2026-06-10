"""Build a live-safe TradingView signal snapshot from the research snapshot.

The research snapshot contains both event-time fields and offline future labels.
This script strips all post-signal fields so daily candidate generation can read
a snapshot that is safe for T-close ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CONFIG_PATH = "scripts/alpha_research/tradingview/configs/l1_execution_v1.json"
DEFAULT_OUTPUT_PATH = "scripts/alpha_research/tradingview/data/live_signal_snapshot.parquet"
FORBIDDEN_PREFIXES = ("future_", "next_", "risk_")
FORBIDDEN_COLUMNS = {
    "suggested_entry_date",
    "suggested_entry_price",
    "tradable_flag",
    "tradable_reason",
    "execution_bucket",
    "open_executable",
    "next_open_executable",
    "next_open_selected",
}
REQUIRED_COLUMNS = {
    "snapshot_version",
    "signal_source",
    "signal_date",
    "signal_year",
    "vt_symbol",
    "buy_point_type",
    "event_id",
    "has_valid_bar",
    "is_list_life_valid",
    "is_st",
    "is_new_stock",
    "rank_score_core4_short70_30",
    "rank_score_short_tp_v1",
    "candidate_pool_score",
    "start_score",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build a live-safe L1 signal snapshot")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--research-snapshot-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a CSV next to the parquet for inspection.",
    )
    return parser.parse_args()


def normalize_date(values: Any) -> pd.Series:
    """Normalize date-like values."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def load_config(path: str) -> dict[str, Any]:
    """Load v1 config."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def forbidden_columns(columns: list[str]) -> list[str]:
    """Return columns that must not exist in a live-safe snapshot."""
    return sorted(
        column
        for column in columns
        if column in FORBIDDEN_COLUMNS or column.startswith(FORBIDDEN_PREFIXES)
    )


def live_columns(columns: list[str]) -> list[str]:
    """Return research snapshot columns allowed in the live snapshot."""
    blocked = set(forbidden_columns(columns))
    return [column for column in columns if column not in blocked]


def validate_live_snapshot(frame: pd.DataFrame) -> None:
    """Validate the generated live snapshot."""
    blocked = forbidden_columns(list(frame.columns))
    if blocked:
        raise ValueError(f"live snapshot contains forbidden columns: {blocked}")

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"live snapshot missing required columns: {missing}")

    duplicate_count = int(frame.duplicated(["event_id"]).sum())
    if duplicate_count:
        raise ValueError(f"live snapshot has duplicate event_id rows: {duplicate_count}")


def build_live_snapshot(
    research_snapshot_path: Path,
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a live-safe snapshot by removing offline fields."""
    research = pd.read_parquet(research_snapshot_path)
    research = research.copy()
    research["signal_date"] = normalize_date(research["signal_date"])

    if start_date:
        research = research[research["signal_date"] >= pd.Timestamp(start_date).normalize()]
    if end_date:
        research = research[research["signal_date"] <= pd.Timestamp(end_date).normalize()]

    if research.empty:
        raise ValueError("no research snapshot rows after date filters")

    blocked = forbidden_columns(list(research.columns))
    keep = live_columns(list(research.columns))
    live = research[keep].copy()
    live = live.sort_values(["signal_date", "vt_symbol", "buy_point_type", "event_id"]).reset_index(drop=True)
    validate_live_snapshot(live)

    metadata = {
        "kind": "live_signal_snapshot",
        "source": str(research_snapshot_path),
        "rows": int(len(live)),
        "columns": int(live.shape[1]),
        "dropped_columns": blocked,
        "dropped_column_count": int(len(blocked)),
        "signal_date_min": str(live["signal_date"].min().date()),
        "signal_date_max": str(live["signal_date"].max().date()),
        "buy_point_counts": {
            str(key): int(value)
            for key, value in live.groupby("buy_point_type", sort=True).size().to_dict().items()
        },
    }
    return live, metadata


def main() -> None:
    """Build and write the live snapshot."""
    args = parse_args()
    config = load_config(args.config_path)
    research_snapshot_path = Path(
        args.research_snapshot_path
        or config.get("research_snapshot")
        or config.get("source_snapshot")
    )
    output_path = Path(
        args.output_path
        or config.get("live_signal_snapshot")
        or DEFAULT_OUTPUT_PATH
    )
    metadata_path = Path(args.metadata_path) if args.metadata_path else output_path.with_suffix(".metadata.json")

    live, metadata = build_live_snapshot(
        research_snapshot_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    live.to_parquet(output_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.write_csv:
        live.to_csv(output_path.with_suffix(".csv"), index=False)

    print(
        "wrote "
        f"{output_path} rows={len(live)} columns={live.shape[1]} "
        f"dropped={metadata['dropped_column_count']} "
        f"date_range={metadata['signal_date_min']}..{metadata['signal_date_max']}"
    )
    if metadata["dropped_column_count"]:
        sample = metadata["dropped_columns"][:10]
        suffix = "..." if len(metadata["dropped_columns"]) > len(sample) else ""
        print(f"dropped_columns_sample={sample}{suffix}")


if __name__ == "__main__":
    main()
