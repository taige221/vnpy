from datetime import date, datetime
from pathlib import Path

import polars as pl

from vnpy.alpha.lab import AlphaLab


def test_load_interval_component_parquet(tmp_path: Path) -> None:
    """Load component universe from interval Parquet."""
    lab = AlphaLab(str(tmp_path / "lab"))

    pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
        }
    ).write_parquet(lab.component_path / "trading_dates.parquet")

    pl.DataFrame(
        {
            "universe": ["test", "test"],
            "vt_symbol": ["000001.SZSE", "000002.SZSE"],
            "valid_from": [date(2024, 1, 2), date(2024, 1, 4)],
            "valid_to": [date(2024, 1, 4), date(2024, 1, 5)],
            "interval_days": [3, 2],
            "is_listed": [True, True],
            "latest_bar_date": [date(2024, 1, 4), date(2024, 1, 5)],
            "generated_at": [datetime(2024, 1, 6), datetime(2024, 1, 6)],
        }
    ).write_parquet(lab.component_path / "test.parquet")

    symbols = lab.load_component_symbols("test", "2024-01-03", "2024-01-05")
    assert symbols == ["000001.SZSE", "000002.SZSE"]

    components = lab.load_component_data("test", "2024-01-03", "2024-01-05")
    assert components == {
        datetime(2024, 1, 3): ["000001.SZSE"],
        datetime(2024, 1, 4): ["000001.SZSE", "000002.SZSE"],
        datetime(2024, 1, 5): ["000002.SZSE"],
    }

    filters = lab.load_component_filters("test", "2024-01-03", "2024-01-05")
    assert filters == {
        "000001.SZSE": [(datetime(2024, 1, 3), datetime(2024, 1, 4))],
        "000002.SZSE": [(datetime(2024, 1, 4), datetime(2024, 1, 5))],
    }


def test_save_component_data_writes_point_parquet(tmp_path: Path) -> None:
    """Save and load component data with the point Parquet format."""
    lab = AlphaLab(str(tmp_path / "lab"))

    lab.save_component_data(
        "saved",
        {
            "2024-01-02": ["000002.SZSE", "000001.SZSE"],
            "2024-01-03": ["000002.SZSE"],
        },
    )

    assert (lab.component_path / "saved.parquet").exists()
    assert lab.load_component_symbols("saved", "2024-01-02", "2024-01-03") == [
        "000001.SZSE",
        "000002.SZSE",
    ]
    assert lab.load_component_data("saved", "2024-01-02", "2024-01-03") == {
        datetime(2024, 1, 2): ["000001.SZSE", "000002.SZSE"],
        datetime(2024, 1, 3): ["000002.SZSE"],
    }

    lab.save_component_data("saved", {"2024-01-02": ["000003.SZSE"]})
    assert lab.load_component_data("saved", "2024-01-02", "2024-01-03") == {
        datetime(2024, 1, 2): ["000003.SZSE"],
        datetime(2024, 1, 3): ["000002.SZSE"],
    }
