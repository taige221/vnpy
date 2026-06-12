"""Merge sharded v4 start-grade event outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_EVENTS_NAME = "trendRsiv4_start_grade_events.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Merge v4 start-grade part outputs")
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--events-name", default=DEFAULT_EVENTS_NAME)
    return parser.parse_args()


def main() -> None:
    """Merge event CSVs."""
    args = parse_args()
    parts_dir = Path(args.parts_dir)
    output_dir = Path(args.output_dir)
    paths = sorted(parts_dir.rglob(args.events_name))
    if not paths:
        raise RuntimeError(f"no {args.events_name} files found under {parts_dir}")

    frames = [pd.read_csv(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True)
    if "signal_date" in merged.columns:
        merged["signal_date"] = pd.to_datetime(merged["signal_date"]).dt.normalize()
    dedupe_key = [column for column in ["vt_symbol", "signal_date", "signal_idx", "start_type"] if column in merged.columns]
    if dedupe_key:
        merged = merged.drop_duplicates(dedupe_key)
    sort_key = [column for column in ["signal_date", "vt_symbol", "signal_idx", "start_type"] if column in merged.columns]
    if sort_key:
        merged = merged.sort_values(sort_key).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.events_name
    merged.to_csv(output_path, index=False)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parts_dir": str(parts_dir),
        "output_path": str(output_path),
        "part_files": [str(path) for path in paths],
        "rows": int(len(merged)),
        "symbols": int(merged["vt_symbol"].nunique()) if "vt_symbol" in merged.columns else None,
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"parts={len(paths)} rows={len(merged)}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
