from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl


if TYPE_CHECKING:
    from .lab import AlphaLab


DATETIME_COLUMNS: tuple[str, ...] = ("datetime", "date", "trade_date")
SYMBOL_COLUMNS: tuple[str, ...] = ("vt_symbol", "symbol", "ts_code", "code")
SIGNAL_COLUMNS: tuple[str, ...] = ("signal", "score", "confidence", "rank_score")

EXCHANGE_SUFFIX_MAP: dict[str, str] = {
    "SH": "SSE",
    "SZ": "SZSE",
    "BJ": "BSE",
    "SSE": "SSE",
    "SZSE": "SZSE",
    "BSE": "BSE",
}


def to_vt_symbol(symbol: Any) -> str:
    """Convert common A-share symbols to vn.py vt_symbol format."""
    text: str = str(symbol).strip().upper()

    if not text:
        raise ValueError("Symbol cannot be empty")

    if "." in text:
        raw_symbol, raw_suffix = text.split(".", maxsplit=1)
        suffix: str = EXCHANGE_SUFFIX_MAP.get(raw_suffix, raw_suffix)
        return f"{raw_symbol}.{suffix}"

    if text.isdigit() and len(text) < 6:
        text = text.zfill(6)

    if len(text) == 6 and text.isdigit():
        if text.startswith(("6", "5", "9")):
            return f"{text}.SSE"
        if text.startswith(("0", "2", "3")):
            return f"{text}.SZSE"
        if text.startswith(("4", "8")):
            return f"{text}.BSE"

    return text


def _resolve_column(
    columns: list[str],
    explicit: str | None,
    candidates: tuple[str, ...],
    role: str
) -> str:
    """Resolve one input column from explicit name or common aliases."""
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Column {explicit!r} for {role} does not exist")
        return explicit

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(f"Cannot find {role} column, expected one of {candidates}")


def _datetime_expr(column: str, dtype: pl.DataType) -> pl.Expr:
    """Build a datetime normalization expression for common source formats."""
    expr: pl.Expr = pl.col(column)

    if dtype == pl.Date:
        return expr.cast(pl.Datetime).alias("datetime")

    if dtype == pl.Datetime:
        return expr.cast(pl.Datetime).alias("datetime")

    if dtype in (pl.String, pl.Utf8):
        generic_expr: pl.Expr = (
            pl.when(expr.str.contains(r"^\d{8}$"))
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(expr)
        )
        return pl.coalesce(
            expr.str.strptime(pl.Date, format="%Y%m%d", strict=False).cast(pl.Datetime),
            generic_expr.str.to_datetime(strict=False),
        ).alias("datetime")

    return expr.cast(pl.Datetime, strict=False).alias("datetime")


def normalize_external_signal(
    data: pl.DataFrame,
    datetime_col: str | None = None,
    symbol_col: str | None = None,
    signal_col: str | None = None,
    extra_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Normalize external strategy events to vn.py alpha signal format.

    The returned DataFrame always contains datetime, vt_symbol and signal. Extra
    columns are preserved for custom portfolio strategies.
    """
    columns: list[str] = data.columns
    resolved_datetime: str = _resolve_column(columns, datetime_col, DATETIME_COLUMNS, "datetime")
    resolved_symbol: str = _resolve_column(columns, symbol_col, SYMBOL_COLUMNS, "symbol")
    resolved_signal: str = _resolve_column(columns, signal_col, SIGNAL_COLUMNS, "signal")

    preserved_columns: list[str] = []
    if extra_columns:
        missing_columns: list[str] = [col for col in extra_columns if col not in columns]
        if missing_columns:
            raise ValueError(f"Extra columns do not exist: {missing_columns}")
        preserved_columns = extra_columns

    dtype: pl.DataType = data.schema[resolved_datetime]
    select_exprs: list[pl.Expr] = [
        _datetime_expr(resolved_datetime, dtype),
        pl.col(resolved_symbol).map_elements(to_vt_symbol, return_dtype=pl.Utf8).alias("vt_symbol"),
        pl.col(resolved_signal).cast(pl.Float64, strict=False).alias("signal"),
    ]

    select_exprs.extend(pl.col(col) for col in preserved_columns)

    signal_df: pl.DataFrame = (
        data.select(select_exprs)
        .drop_nulls(["datetime", "vt_symbol", "signal"])
        .sort(["datetime", "vt_symbol", "signal"], descending=[False, False, True])
        .unique(subset=["datetime", "vt_symbol"], keep="first")
        .sort(["datetime", "signal"], descending=[False, True])
    )

    return signal_df


def load_external_signal(
    path: str | Path,
    datetime_col: str | None = None,
    symbol_col: str | None = None,
    signal_col: str | None = None,
    extra_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Load and normalize external strategy events from CSV or Parquet."""
    file_path: Path = Path(path)

    if file_path.suffix == ".csv":
        data: pl.DataFrame = pl.read_csv(file_path, infer_schema_length=0)
    elif file_path.suffix in {".parquet", ".pq"}:
        data = pl.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported signal file type: {file_path.suffix}")

    return normalize_external_signal(
        data,
        datetime_col=datetime_col,
        symbol_col=symbol_col,
        signal_col=signal_col,
        extra_columns=extra_columns,
    )


def import_external_signal(
    lab: "AlphaLab",
    name: str,
    path: str | Path,
    datetime_col: str | None = None,
    symbol_col: str | None = None,
    signal_col: str | None = None,
    extra_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Load an external signal file and save it into AlphaLab."""
    signal_df: pl.DataFrame = load_external_signal(
        path,
        datetime_col=datetime_col,
        symbol_col=symbol_col,
        signal_col=signal_col,
        extra_columns=extra_columns,
    )
    lab.save_signal(name, signal_df)
    return signal_df
