# TradingView Alpha Research Scripts

This directory contains tracked entrypoints for TradingView/A-share research
workflows whose source data and generated reports live under the ignored
`examples/alpha_research/tradingview/` workspace.

## Layout

- `live/`: operational entrypoints for daily use. These scripts read the
  live-safe snapshot by default and must not use `future_*`, `next_*`, or
  `risk_*` fields for T-close decisions.
- `research/`: historical studies and offline experiments. These scripts may
  use future labels only as outcome metrics after the decision stage is fixed.
- `configs/`: frozen execution/version configs shared by live and research
  entrypoints.
- `data/` and `reports/`: generated local artifacts, ignored by Git.

## L1 Execution V1 Daily Candidates

Build or refresh the live-safe snapshot first:

```bash
rtk python3 scripts/alpha_research/tradingview/live/build_live_signal_snapshot.py
```

Run:

```bash
rtk python3 scripts/alpha_research/tradingview/live/run_l1_daily_candidates.py
```

The script reads `configs/l1_execution_v1.json` and defaults to the live-safe
snapshot at `scripts/alpha_research/tradingview/data/live_signal_snapshot.parquet`.
That file is derived from the research snapshot but strips `future_*`, `next_*`,
`risk_*`, and other post-signal fields. The daily flow is separated into
explicit stages:

- `signal_close_day_filter`: T close day-level gate, currently
  `core4_mean >= 0.52`.
- `signal_close_top20`: T close stock-level ranked pool, sorted by
  `rank_score_core4_short70_30`.
- `next_open_select_top2_from_signal_close_top20`: T+1 open executable
  selection from the signal-close top20 after gap, limit-state, liquidity, and
  industry controls.

When next-open fields are not available yet, the script still emits the
`signal_close_top20` file and marks execution as pending.

For live next-open execution checks, pass `--live-open-source tencent`. This
overlays runtime-only Tencent quote fields such as open gap, limit state,
turnover, and float market cap for the current run; it does not write live quote
data back into the historical signal snapshot.

Daily candidate files default to
`scripts/alpha_research/tradingview/reports/daily_candidates/` so operational
outputs stay separate from historical backtest artifacts under `examples/`.

For historical checks that intentionally need research-snapshot `future_*` and
`next_*` fields, pass `--snapshot-kind research`. The default `live` kind rejects
post-signal columns by design.

## L1 Clean-Timeline Backtest

Run:

```bash
rtk python3 scripts/alpha_research/tradingview/research/run_l1_clean_timeline_backtest.py
```

This is the canonical no-lookahead timing check for L1 v1. It does not use the
old `l1_daily_filter_table.csv` as a decision input. Instead it computes:

- `signal_close_day_filter_pass`: T close day-level pass/fail.
- `signal_close_rank_pool_member`: T close stock-level rank-pool membership.
- `next_open_executable`: T+1 open executable state.
- `next_open_selected`: T+1 open selected position.

The first two stages reject `future_*`, `next_*`, and other post-signal fields
by code-level guard. Future-label fields remain allowed only in the offline
outcome section.

## L1 Rank Score Study

Run:

```bash
rtk python3 scripts/alpha_research/tradingview/research/run_l1_rank_score_study.py
```

This keeps the L1 signal, `core4_mean >= 0.52` day filter, signal-close top20,
and next-open execution rules fixed, then compares ranking scores only. Outputs
default to `scripts/alpha_research/tradingview/reports/rank_score_study/`.
Use `--industry-cap none` for a no-industry-cap robustness check.

The default score set includes baseline scores, conservative blends, single
T-close factor high/low scans, and conditional score-switching templates. Full
event-level selection output is intentionally optional because broad scans are
large; pass `--write-event-selection` when event details are needed.

Pass `--extra-factor-scan` to compute an additional local-panel factor set at
T close, including reversal, box-stack lift reversal, anti-limit-up heat,
double-bottom similarity, anti-MA-bias, anti-turnover, anti-MACD, anti-TRIX,
anti-ATR, PB percentile, and the TradingView down-context flag. These fields are
used only for signal-close ranking scans; future labels remain offline outcomes.

## L1 V1 Exit Path Experiment

Run:

```bash
rtk python3 scripts/alpha_research/tradingview/research/run_v1_exit_path_experiment.py
```

This reads the frozen clean-timeline event selection from
`reports/v1_core4_top2_event_detail/l1_rank_score_event_selection.csv` and
does not re-rank or re-filter the buy points. It reloads local daily bars and
compares fixed exits, target-plus-timeout exits, wide-stop variants,
profit-protection variants, and limit-up next-open handling on the same event
set. Outputs default to
`scripts/alpha_research/tradingview/reports/v1_exit_path_experiment/`.
