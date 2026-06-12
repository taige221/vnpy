"""Print PRE_L1/B1 sharded rebuild commands."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Print PRE_L1/B1 rebuild commands")
    parser.add_argument("--run-id", default="rebuild_2015_2019")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--lab-path", default="lab/a_share_research")
    parser.add_argument("--base-dir", default="scripts/alpha_build/data")
    parser.add_argument(
        "--snapshot-path",
        default="scripts/alpha_build/data/rebuild_2015_2019/outputs/a5_snapshot/a5_signal_snapshot.parquet",
    )
    parser.add_argument("--b1-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Print shell commands."""
    args = parse_args()
    run_dir = Path(args.base_dir) / args.run_id
    parts_dir = run_dir / "parts" / "pre_l1_b1"
    output_dir = run_dir / "outputs" / "pre_l1_b1"
    b1 = " --b1-only" if args.b1_only else ""

    print("# Run PRE_L1/B1 parts")
    for shard_index in range(args.shard_count):
        part_dir = parts_dir / f"part_{shard_index:03d}"
        print(
            "rtk python3 scripts/alpha_research/tradingview/research/"
            "run_pre_l1_near_start_study.py "
            f"--lab-path {args.lab_path} --snapshot-path {args.snapshot_path} "
            f"--start {args.start} --end {args.end} --output-dir {part_dir} "
            f"--shard-count {args.shard_count} --shard-index {shard_index} "
            f"--progress-every 500 --write-events{b1}"
        )
    print()
    print("# Merge PRE_L1/B1 event parts")
    print(
        "rtk python3 scripts/alpha_build/pipelines/pre_l1_b1/merge_pre_l1_parts.py "
        f"--parts-dir {parts_dir} --output-dir {output_dir}"
    )


if __name__ == "__main__":
    main()
