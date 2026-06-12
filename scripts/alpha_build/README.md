# Alpha Build Scripts

This directory is for build-time orchestration, sharding, merge, and range-combine
utilities. It should not contain signal definitions or research conclusions.

Boundary:

- `scripts/alpha_build`: how to build, split, merge, and combine research datasets.
- `scripts/alpha_research`: mature research scripts and experiments.

The build layer calls existing research scripts instead of reimplementing signal
logic. Future labels remain offline research data and must not be used in live
ranking.

Build artifacts should be written under `scripts/alpha_build/data/<run_id>/`.
That directory is ignored by Git except for `.gitkeep`.

## Data Layout

Each build run should use one `run_id`:

```text
scripts/alpha_build/data/<run_id>/
├── manifests/
├── parts/
│   ├── v4_start_grade/
│   ├── a5_snapshot/
│   └── pre_l1_b1/
└── outputs/
    ├── v4_start_grade/
    ├── v5_playbook/
    ├── a5_snapshot/
    └── pre_l1_b1/
```

`parts/` are temporary shard outputs. `outputs/` are complete run-level outputs.
Research and validation scripts should read `outputs/`, not `parts/`.

## L1/A5 Snapshot Build

Use this flow when rebuilding the L1/L2/L3 A5 research snapshot for a date range.
For example, a historical run could use:

```text
run_id = rebuild_2015_2019
start = 2015-01-01
end = 2019-12-31
years = 2015,2016,2017,2018,2019
```

The same flow applies to any other date range.

### 1. Print Commands

Generate the concrete command list:

```bash
rtk python3 scripts/alpha_build/pipelines/l1_a5/print_rebuild_commands.py \
  --run-id <run_id> \
  --start <start> \
  --end <end> \
  --years <comma-separated-years> \
  --shard-count <N>
```

Add `--skip-pvcorr` when the date range does not have reliable combo/pvcorr
factor context.

### 2. Build Symbol Manifest

The first printed command writes:

```text
scripts/alpha_build/data/<run_id>/manifests/symbol_shards_<N>.csv
```

This file records `vt_symbol`, `ordinal`, `shard_index`, and `shard_count`.

### 3. Run v4 Start-Grade Parts

Run every printed `run_trend_rsi_v4_start_grade_study.py` command. Each shard
writes a local v4 event set under:

```text
scripts/alpha_build/data/<run_id>/parts/v4_start_grade/part_<idx>/
```

### 4. Merge v4 Parts

Run the printed `merge_v4_start_grade_parts.py` command. It writes the complete
v4 event file:

```text
scripts/alpha_build/data/<run_id>/outputs/v4_start_grade/trendRsiv4_start_grade_events.csv
```

### 5. Run v5 Playbook

Run `run_trend_rsi_v5_market_playbook_study.py` once from the merged v4 events.
This step is not sharded because it is fast and event-table based.

Output:

```text
scripts/alpha_build/data/<run_id>/outputs/v5_playbook/trendRsiv5_market_playbook_events.csv
```

### 6. Run A5 Snapshot Parts

Run every printed `build_a5_signal_snapshot.py` command. Each shard writes a
local snapshot under:

```text
scripts/alpha_build/data/<run_id>/parts/a5_snapshot/part_<idx>/
```

### 7. Merge A5 Snapshot Parts

Run the printed `merge_a5_signal_snapshot_parts.py` command. It writes the
complete run-level snapshot:

```text
scripts/alpha_build/data/<run_id>/outputs/a5_snapshot/a5_signal_snapshot.parquet
scripts/alpha_build/data/<run_id>/outputs/a5_snapshot/daily_filter_snapshot.csv
scripts/alpha_build/data/<run_id>/outputs/a5_snapshot/snapshot_schema.md
```

At this point, the L1/L2/L3 build is complete.

## Optional Pre-L1/B1 Build

Use this flow when rebuilding `PRE_L1_near_start` or `B1_strong_open_break_high`
research data. It depends on an A5 snapshot only for offline L1 labels such as
`tomorrow_l1`.

Print commands:

```bash
rtk python3 scripts/alpha_build/pipelines/pre_l1_b1/print_rebuild_commands.py \
  --run-id <run_id> \
  --start <start> \
  --end <end> \
  --shard-count <N> \
  --snapshot-path scripts/alpha_build/data/<run_id>/outputs/a5_snapshot/a5_signal_snapshot.parquet
```

Add `--b1-only` to build only the B1 open-break-high event set.

Run all printed part commands, then the printed merge command. Output is written
under:

```text
scripts/alpha_build/data/<run_id>/outputs/pre_l1_b1/
```

## Combining Date Ranges

To combine a rebuilt historical snapshot with another complete snapshot, use the
range-combine utility. Example:

```bash
rtk python3 scripts/alpha_build/pipelines/l1_a5/combine_a5_snapshot_ranges.py \
  --snapshot-path scripts/alpha_build/data/<historical_run>/outputs/a5_snapshot/a5_signal_snapshot.parquet \
  --snapshot-path <existing_snapshot_path> \
  --output-dir scripts/alpha_build/data/<combined_run>/outputs/a5_snapshot
```

The combined output is another complete snapshot directory and can be used by
downstream research scripts.

## Practical Notes

- Sharding is by baskets of symbols, not by year.
- Only shard slow symbol-scan steps. Do not shard v5 playbook or simple snapshot
  read-only studies.
- If a part fails, rerun only that part and then rerun the relevant merge step.
- Keep generated artifacts under `scripts/alpha_build/data/`; promote only
  selected final reports or snapshots elsewhere when they become a stable
  research baseline.
