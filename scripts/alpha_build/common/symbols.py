"""Symbol selection helpers for sharded alpha builds."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from pathlib import Path


SYMBOL_COLUMNS = ("vt_symbol", "symbol", "code", "ts_code")


def add_symbol_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add shared symbol-list and shard arguments to a parser."""
    parser.add_argument(
        "--symbol-list",
        help="Optional CSV/TXT file of symbols to process. CSV columns may include vt_symbol/symbol/code/ts_code.",
    )
    parser.add_argument("--shard-count", type=int, help="Total number of symbol shards.")
    parser.add_argument("--shard-index", type=int, help="Zero-based shard index to process.")


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    """Return sorted unique non-empty symbols."""
    return sorted({str(symbol).strip() for symbol in symbols if str(symbol).strip()})


def load_daily_symbols(lab_path: str | Path) -> list[str]:
    """Load symbols from ``lab/a_share_research/daily`` style parquet files."""
    daily_path = Path(lab_path) / "daily"
    return normalize_symbols(path.stem for path in daily_path.glob("*.parquet"))


def read_symbol_list(path: str | Path) -> list[str]:
    """Read symbols from a CSV/TXT file."""
    value = Path(path)
    if not value.exists():
        raise FileNotFoundError(f"symbol list not found: {value}")

    with value.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        if "," in sample or "\t" in sample:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            reader = csv.DictReader(file, dialect=dialect)
            if reader.fieldnames:
                field_map = {name.strip(): name for name in reader.fieldnames}
                column = next((field_map[name] for name in SYMBOL_COLUMNS if name in field_map), None)
                if column:
                    return normalize_symbols(row.get(column, "") for row in reader)

        file.seek(0)
        symbols: list[str] = []
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            symbols.append(line.split(",")[0].split("\t")[0].strip())
        return normalize_symbols(symbols)


def validate_shard_args(shard_count: int | None, shard_index: int | None) -> None:
    """Validate shard arguments."""
    if shard_count is None and shard_index is None:
        return
    if shard_count is None or shard_index is None:
        raise ValueError("--shard-count and --shard-index must be provided together")
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")


def select_symbols(
    symbols: Sequence[str],
    *,
    symbol_list: str | Path | None = None,
    shard_count: int | None = None,
    shard_index: int | None = None,
) -> list[str]:
    """Apply optional symbol-list and deterministic modulo sharding filters."""
    selected = normalize_symbols(symbols)
    if symbol_list:
        allowed = set(read_symbol_list(symbol_list))
        selected = [symbol for symbol in selected if symbol in allowed]

    validate_shard_args(shard_count, shard_index)
    if shard_count is not None and shard_index is not None:
        selected = [
            symbol
            for ordinal, symbol in enumerate(selected)
            if ordinal % shard_count == shard_index
        ]
    return selected


def select_paths(
    paths: Sequence[Path],
    *,
    symbol_list: str | Path | None = None,
    shard_count: int | None = None,
    shard_index: int | None = None,
) -> list[Path]:
    """Apply symbol-list and shard filters to parquet paths."""
    path_by_symbol = {path.stem: path for path in paths}
    selected_symbols = select_symbols(
        path_by_symbol,
        symbol_list=symbol_list,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    return [path_by_symbol[symbol] for symbol in selected_symbols]
