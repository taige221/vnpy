"""Print L1/A5 sharded rebuild commands."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Print L1/A5 rebuild commands")
    parser.add_argument("--run-id", default="rebuild_2015_2019")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument("--years", default="2015,2016,2017,2018,2019")
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--lab-path", default="lab/a_share_research")
    parser.add_argument("--base-dir", default="scripts/alpha_build/data")
    parser.add_argument("--skip-pvcorr", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Print shell commands."""
    args = parse_args()
    run_dir = Path(args.base_dir) / args.run_id
    manifest = run_dir / "manifests" / f"symbol_shards_{args.shard_count:03d}.csv"
    v4_parts = run_dir / "parts" / "v4_start_grade"
    v4_output = run_dir / "outputs" / "v4_start_grade"
    v5_output = run_dir / "outputs" / "v5_playbook"
    a5_parts = run_dir / "parts" / "a5_snapshot"
    a5_output = run_dir / "outputs" / "a5_snapshot"

    print("# 1. Build symbol shard manifest")
    print(
        "rtk python3 scripts/alpha_build/build_symbol_shards.py "
        f"--lab-path {args.lab_path} --shard-count {args.shard_count} --output-path {manifest}"
    )
    print()
    print("# 2. Run v4 start-grade parts")
    for shard_index in range(args.shard_count):
        output_dir = v4_parts / f"part_{shard_index:03d}"
        skip = " --skip-pvcorr" if args.skip_pvcorr else ""
        print(
            "rtk python3 examples/alpha_research/tradingview/vnpy_alpha/"
            "run_trend_rsi_v4_start_grade_study.py "
            f"--lab-path {args.lab_path} --start {args.start} --end {args.end} "
            f"--output-dir {output_dir} --shard-count {args.shard_count} "
            f"--shard-index {shard_index} --progress-every 500{skip}"
        )
    print()
    print("# 3. Merge v4 parts")
    print(
        "rtk python3 scripts/alpha_build/pipelines/l1_a5/merge_v4_start_grade_parts.py "
        f"--parts-dir {v4_parts} --output-dir {v4_output}"
    )
    print()
    print("# 4. Run v5 playbook once")
    print(
        "rtk python3 examples/alpha_research/tradingview/vnpy_alpha/"
        "run_trend_rsi_v5_market_playbook_study.py "
        f"--events-path {v4_output / 'trendRsiv4_start_grade_events.csv'} "
        f"--output-dir {v5_output}"
    )
    print()
    print("# 5. Run A5 snapshot parts")
    for shard_index in range(args.shard_count):
        output_dir = a5_parts / f"part_{shard_index:03d}"
        print(
            "rtk python3 examples/alpha_research/tradingview/vnpy_alpha/"
            "build_a5_signal_snapshot.py "
            f"--lab-path {args.lab_path} "
            f"--event-path {v5_output / 'trendRsiv5_market_playbook_events.csv'} "
            f"--output-dir {output_dir} --years {args.years} "
            f"--shard-count {args.shard_count} --shard-index {shard_index} "
            "--progress-every 500"
        )
    print()
    print("# 6. Merge A5 snapshot parts")
    print(
        "rtk python3 scripts/alpha_build/pipelines/l1_a5/merge_a5_signal_snapshot_parts.py "
        f"--parts-dir {a5_parts} --output-dir {a5_output}"
    )


if __name__ == "__main__":
    main()
