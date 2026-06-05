from datetime import datetime

import polars as pl

from vnpy.alpha.utils import load_external_signal, normalize_external_signal, to_vt_symbol


def test_to_vt_symbol_converts_common_a_share_formats() -> None:
    """Test common A-share symbol conversions."""
    assert to_vt_symbol("600000.SH") == "600000.SSE"
    assert to_vt_symbol("000001.SZ") == "000001.SZSE"
    assert to_vt_symbol("1") == "000001.SZSE"
    assert to_vt_symbol("688001") == "688001.SSE"
    assert to_vt_symbol("300750") == "300750.SZSE"
    assert to_vt_symbol("830799") == "830799.BSE"


def test_normalize_external_signal_with_alias_columns() -> None:
    """Test normalization from common event table column aliases."""
    source = pl.DataFrame(
        {
            "date": ["20240102", "2024-01-02", "2024-01-03"],
            "symbol": ["600000.SH", "000001.SZ", "300750"],
            "score": [0.9, 0.7, 0.8],
            "signal_type": ["box", "w_bottom", "vp"],
        }
    )

    signal = normalize_external_signal(source, extra_columns=["signal_type"])

    assert signal.columns == ["datetime", "vt_symbol", "signal", "signal_type"]
    assert signal["datetime"].to_list() == [
        datetime(2024, 1, 2),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]
    assert signal["vt_symbol"].to_list() == ["600000.SSE", "000001.SZSE", "300750.SZSE"]
    assert signal["signal"].to_list() == [0.9, 0.7, 0.8]


def test_normalize_external_signal_keeps_highest_duplicate() -> None:
    """Test duplicate symbol-date events keep the highest signal score."""
    source = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "vt_symbol": ["600000.SSE", "600000.SSE"],
            "signal": [0.2, 0.8],
        }
    )

    signal = normalize_external_signal(source)

    assert signal.height == 1
    assert signal["signal"].to_list() == [0.8]


def test_load_external_signal_preserves_csv_date_and_symbol_strings(tmp_path) -> None:
    """Test CSV import preserves numeric dates and leading-zero symbols."""
    csv_path = tmp_path.joinpath("signal.csv")
    csv_path.write_text("date,symbol,score\n20240102,000001,0.8\n", encoding="utf-8")

    signal = load_external_signal(csv_path)

    assert signal["datetime"].to_list() == [datetime(2024, 1, 2)]
    assert signal["vt_symbol"].to_list() == ["000001.SZSE"]
    assert signal["signal"].to_list() == [0.8]


def test_normalize_external_signal_parses_integer_yyyymmdd_dates() -> None:
    """Test in-memory numeric YYYYMMDD dates are parsed as calendar dates."""
    source = pl.DataFrame(
        {
            "trade_date": [20240102],
            "ts_code": ["000001.SZ"],
            "signal": [1.0],
        }
    )

    signal = normalize_external_signal(source)

    assert signal["datetime"].to_list() == [datetime(2024, 1, 2)]
    assert signal["vt_symbol"].to_list() == ["000001.SZSE"]


def test_load_external_signal_parses_parquet_integer_yyyymmdd_dates(tmp_path) -> None:
    """Test Parquet numeric YYYYMMDD dates are parsed as calendar dates."""
    parquet_path = tmp_path.joinpath("signal.parquet")
    pl.DataFrame(
        {
            "trade_date": [20240102],
            "ts_code": ["000001.SZ"],
            "signal": [1.0],
        }
    ).write_parquet(parquet_path)

    signal = load_external_signal(parquet_path)

    assert signal["datetime"].to_list() == [datetime(2024, 1, 2)]
    assert signal["vt_symbol"].to_list() == ["000001.SZSE"]

