# L1/A5 Snapshot Build Pipeline

This pipeline rebuilds the TrendRSI A5 research base:

1. Run v4 start-grade parts by symbol shard.
2. Merge v4 part events.
3. Run v5 playbook once from the merged v4 events.
4. Run A5 snapshot parts by symbol shard.
5. Merge A5 snapshot parts.
6. Optionally combine rebuilt 2015-2019 output with the existing 2020-2026 snapshot.

Only part generation is sharded. Research and validation should read the merged
or combined outputs.
