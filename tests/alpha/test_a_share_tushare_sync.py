from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from vnpy.alpha.utils import a_share_tushare_sync


class FakeConnection:
    """Minimal DuckDB connection double for partition replacement tests."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, list[Any] | None]] = []

    def execute(self, sql: str, params: list[Any] | None = None) -> FakeConnection:
        """Record executed statements."""
        self.statements.append((sql, params))
        return self


def test_replace_dynamic_partition_deletes_existing_partition_when_source_empty(monkeypatch: Any) -> None:
    """Empty source partitions should clear stale rows from previous syncs."""
    monkeypatch.setattr(a_share_tushare_sync, "table_exists", lambda con, table: True)
    con = FakeConnection()

    rows = a_share_tushare_sync.replace_dynamic_partition(
        con,
        "index_weight",
        pd.DataFrame(),
        "index_code = ? AND trade_date >= ? AND trade_date <= ?",
        ["000300.SH", date(2024, 1, 1), date(2024, 1, 31)],
    )

    assert rows == 0
    assert con.statements == [
        (
            'DELETE FROM "index_weight" WHERE index_code = ? AND trade_date >= ? AND trade_date <= ?',
            ["000300.SH", date(2024, 1, 1), date(2024, 1, 31)],
        )
    ]
