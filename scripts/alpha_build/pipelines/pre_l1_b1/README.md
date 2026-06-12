# Pre-L1/B1 Build Pipeline

This pipeline is separate from the L1/A5 snapshot pipeline. It rebuilds
`PRE_L1_near_start` and `B1_strong_open_break_high` research outputs by symbol
part and then merges the part outputs.

The L1 snapshot is only used for offline labels such as `tomorrow_l1`.
