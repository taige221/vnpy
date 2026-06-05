"""
Sync full-market A-share TuShare data into AlphaLab.

The script keeps two layers:

1. A source DuckDB under the lab folder for raw TuShare tables.
2. AlphaLab files under daily/component/contract.json for vnpy.alpha research.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    import duckdb  # type: ignore[import-not-found]
except ModuleNotFoundError:
    duckdb = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None  # type: ignore[assignment]

try:
    import requests  # type: ignore[import-untyped]
except ModuleNotFoundError:
    requests = None


API_URL = "https://api.tushare.pro"
DATE_FORMAT = "%Y-%m-%d"

DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
ADJ_FIELDS = "ts_code,trade_date,adj_factor"
STK_FACTOR_FIELDS = (
    "ts_code,trade_date,open,high,low,close,pre_close,vol,amount,adj_factor"
)
DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
    "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
    "free_share,total_mv,circ_mv"
)
STK_LIMIT_FIELDS = "ts_code,trade_date,up_limit,down_limit,pre_close"
STOCK_BASIC_FIELDS = (
    "ts_code,symbol,name,area,industry,market,list_date,delist_date,"
    "exchange,curr_type,list_status,is_hs"
)
TRADE_CAL_FIELDS = "exchange,cal_date,is_open,pretrade_date"
INDEX_BASIC_FIELDS = "ts_code,name,market,publisher,category,base_date,base_point,list_date"
INDEX_WEIGHT_FIELDS = "index_code,con_code,trade_date,weight"
INDEX_MEMBER_ALL_FIELDS = "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,name,in_date,out_date,is_new"
DISCLOSURE_DATE_FIELDS = "ts_code,ann_date,end_date,pre_date,actual_date,modify_date"
FUNDAMENTAL_APIS = {
    "income": "fundamental_income",
    "balancesheet": "fundamental_balancesheet",
    "cashflow": "fundamental_cashflow",
    "fina_indicator": "fundamental_fina_indicator",
}


def require_sync_dependencies() -> None:
    """Raise a clear error when optional sync dependencies are missing."""
    missing: list[str] = []
    if duckdb is None:
        missing.append("duckdb")
    if pd is None:
        missing.append("pandas")
    if requests is None:
        missing.append("requests")
    if missing:
        raise RuntimeError(
            "A-share TuShare sync requires optional dependencies: "
            + ", ".join(missing)
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Sync full A-share TuShare data into AlphaLab")
    parser.add_argument("--lab-path", default="lab/a_share_research", help="AlphaLab folder")
    parser.add_argument("--db-path", help="Source DuckDB path, defaults to <lab>/source/tushare_full.duckdb")
    parser.add_argument("--env-file", help="Optional .env file containing TUSHARE_TOKEN; defaults to <lab>/.env then ./.env")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=datetime.now().strftime(DATE_FORMAT), help="End date YYYY-MM-DD")
    parser.add_argument("--rate-limit", type=float, default=0.75, help="Minimum seconds between TuShare calls")
    parser.add_argument("--timeout", type=float, default=60, help="TuShare HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=6, help="TuShare retry attempts per call")
    parser.add_argument("--force", action="store_true", help="Refetch dates already marked ok")
    parser.add_argument("--sync-only", action="store_true", help="Only sync DuckDB, skip AlphaLab export")
    parser.add_argument("--export-only", action="store_true", help="Only export AlphaLab files from DuckDB")
    parser.add_argument("--skip-market-daily", action="store_true", help="Skip daily OHLC sync")
    parser.add_argument("--sync-priority-data", action="store_true", help="Sync industry and benchmark index source tables")
    parser.add_argument("--sync-fundamental-data", action="store_true", help="Sync disclosure dates and PIT fundamental source tables")
    parser.add_argument("--use-slow-daily-fallback", action="store_true", help="Use stk_factor if daily returns empty; this endpoint is heavily rate limited")
    parser.add_argument("--index-codes", default="000300.SH,000905.SH,000852.SH", help="Comma-separated benchmark index codes")
    parser.add_argument("--index-markets", default="SW", help="Comma-separated index_basic markets")
    parser.add_argument("--fundamental-start-date", help="Fundamental announcement start date YYYY-MM-DD, defaults to --start-date")
    parser.add_argument("--fundamental-end-date", help="Fundamental announcement end date YYYY-MM-DD, defaults to --end-date")
    parser.add_argument("--fundamental-symbol-limit", type=int, help="Limit symbols for fundamental sync smoke tests")
    parser.add_argument("--fundamental-apis", default="income,balancesheet,cashflow,fina_indicator", help="Comma-separated fundamental APIs")
    parser.add_argument("--min-factor-bars", type=int, default=1000, help="Minimum bars for static factor universe")
    parser.add_argument("--active-latest-only", action="store_true", help="Require latest date coverage for active universe")
    parser.add_argument("--build-research-panel", action="store_true", help="Build PIT interval tables and research_panel_daily Parquet")
    parser.add_argument("--panel-start-date", help="Research panel start date YYYY-MM-DD, defaults to first daily_raw date")
    parser.add_argument("--panel-end-date", help="Research panel end date YYYY-MM-DD, defaults to latest daily_raw date")
    return parser.parse_args()


def parse_date(value: str) -> date:
    """Parse YYYY-MM-DD date."""
    return datetime.strptime(value, DATE_FORMAT).date()


def load_token(env_files: list[Path]) -> str:
    """Load TuShare token from environment or .env without printing it."""
    for key in ("TUSHARE_TOKEN", "TUSHARE_API_KEY", "TUSHARE_KEY"):
        value = os.environ.get(key)
        if value:
            return value.strip()

    for env_file in env_files:
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", maxsplit=1)
            if key.strip() in {"TUSHARE_TOKEN", "TUSHARE_API_KEY", "TUSHARE_KEY"}:
                return value.strip().strip("\"'")

    raise RuntimeError("TUSHARE_TOKEN is not configured in environment or .env")


def to_date_series(series: pd.Series) -> pd.Series:
    """Convert TuShare date strings to Python dates."""
    return pd.to_datetime(series, format="%Y%m%d", errors="coerce").dt.date


def normalize_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    """Normalize numeric columns in place."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def to_vt_symbol(ts_code: str) -> str:
    """Convert TuShare ts_code to vn.py vt_symbol."""
    return (
        str(ts_code)
        .replace(".SZ", ".SZSE")
        .replace(".SH", ".SSE")
        .replace(".BJ", ".BSE")
    )


def to_vt_symbol_sql(column: str) -> str:
    """Return a SQL expression converting ts_code to vn.py vt_symbol."""
    exchange = f"split_part({column}, '.', 2)"
    return (
        f"split_part({column}, '.', 1) || '.' || "
        f"CASE {exchange} "
        "WHEN 'SZ' THEN 'SZSE' "
        "WHEN 'SH' THEN 'SSE' "
        "WHEN 'BJ' THEN 'BSE' "
        f"ELSE {exchange} END"
    )


def quote_path(path: Path) -> str:
    """Escape a path for a SQL string literal."""
    return str(path).replace("'", "''")


def quote_identifier(name: str) -> str:
    """Quote a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


class TushareClient:
    """Small TuShare Pro HTTP client with retry and rate limiting."""

    def __init__(self, token: str, rate_limit: float, timeout: float, retries: int) -> None:
        """Initialize client."""
        self.token = token
        self.rate_limit = max(float(rate_limit), 0)
        self.timeout = max(float(timeout), 5)
        self.retries = max(int(retries), 1)
        self.last_call = 0.0
        self.session = requests.Session()

    def request_payload(self, api_name: str, params: dict[str, Any] | None = None, fields: str = "") -> dict[str, Any]:
        """Call one TuShare endpoint and return its JSON payload."""
        for attempt in range(1, self.retries + 1):
            wait = self.rate_limit - (time.monotonic() - self.last_call)
            if wait > 0:
                time.sleep(wait)

            self.last_call = time.monotonic()
            try:
                response = self.session.post(
                    API_URL,
                    json={
                        "api_name": api_name,
                        "token": self.token,
                        "params": params or {},
                        "fields": fields,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception:
                self.session.close()
                self.session = requests.Session()
                if attempt == self.retries:
                    raise
                time.sleep(2 * attempt)
                continue

            code = int(payload.get("code", -1))
            if code != 0:
                message = str(payload.get("msg") or payload.get("message") or "")
                if attempt == self.retries:
                    raise RuntimeError(f"TuShare {api_name} failed code={code}: {message}")
                time.sleep(2 * attempt)
                continue

            data = payload.get("data") or {}
            return data

        return {}

    def call(self, api_name: str, params: dict[str, Any] | None = None, fields: str = "") -> pd.DataFrame:
        """Call one TuShare endpoint and return a DataFrame."""
        data = self.request_payload(api_name, params, fields)
        return pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])

    def call_all(
        self,
        api_name: str,
        params: dict[str, Any] | None = None,
        fields: str = "",
        limit: int = 3000,
    ) -> pd.DataFrame:
        """Call a TuShare endpoint until paginated results are exhausted."""
        frames: list[pd.DataFrame] = []
        offset = 0
        while True:
            page_params = dict(params or {})
            page_params["limit"] = limit
            page_params["offset"] = offset
            data = self.request_payload(api_name, page_params, fields)
            frame = pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])
            if not frame.empty:
                frames.append(frame)
            if not bool(data.get("has_more")):
                break
            offset += limit
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    """Create source tables."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_status (
            api_name VARCHAR,
            trade_date DATE,
            status VARCHAR,
            row_count BIGINT,
            error VARCHAR,
            synced_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamental_sync_status (
            api_name VARCHAR,
            ts_code VARCHAR,
            start_date DATE,
            end_date DATE,
            status VARCHAR,
            row_count BIGINT,
            error VARCHAR,
            synced_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_basic (
            ts_code VARCHAR,
            symbol VARCHAR,
            name VARCHAR,
            area VARCHAR,
            industry VARCHAR,
            market VARCHAR,
            list_date DATE,
            delist_date DATE,
            exchange VARCHAR,
            curr_type VARCHAR,
            list_status VARCHAR,
            is_hs VARCHAR,
            synced_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_calendar (
            exchange VARCHAR,
            cal_date DATE,
            is_open INTEGER,
            pretrade_date DATE,
            synced_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_raw (
            ts_code VARCHAR,
            trade_date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            pre_close DOUBLE,
            change DOUBLE,
            pct_chg DOUBLE,
            vol DOUBLE,
            amount DOUBLE,
            adj_factor DOUBLE,
            open_qfq DOUBLE,
            high_qfq DOUBLE,
            low_qfq DOUBLE,
            close_qfq DOUBLE,
            synced_at TIMESTAMP
        )
        """
    )
    con.execute("ALTER TABLE daily_raw ADD COLUMN IF NOT EXISTS change DOUBLE")
    con.execute("ALTER TABLE daily_raw ADD COLUMN IF NOT EXISTS pct_chg DOUBLE")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_basic (
            ts_code VARCHAR,
            trade_date DATE,
            close DOUBLE,
            turnover_rate DOUBLE,
            turnover_rate_f DOUBLE,
            volume_ratio DOUBLE,
            pe DOUBLE,
            pe_ttm DOUBLE,
            pb DOUBLE,
            ps DOUBLE,
            ps_ttm DOUBLE,
            dv_ratio DOUBLE,
            dv_ttm DOUBLE,
            total_share DOUBLE,
            float_share DOUBLE,
            free_share DOUBLE,
            total_mv DOUBLE,
            circ_mv DOUBLE,
            synced_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS stk_limit (
            ts_code VARCHAR,
            trade_date DATE,
            up_limit DOUBLE,
            down_limit DOUBLE,
            pre_close DOUBLE,
            synced_at TIMESTAMP
        )
        """
    )


def status_ok(con: duckdb.DuckDBPyConnection, api_name: str, trade_date: date) -> bool:
    """Return whether a daily endpoint was already synced."""
    row = con.execute(
        "SELECT status FROM sync_status WHERE api_name = ? AND trade_date = ? ORDER BY synced_at DESC LIMIT 1",
        [api_name, trade_date],
    ).fetchone()
    return bool(row and row[0] == "ok")


def mark_status(
    con: duckdb.DuckDBPyConnection,
    api_name: str,
    trade_date: date,
    status: str,
    row_count: int,
    error: str | None = None,
) -> None:
    """Persist endpoint sync status."""
    con.execute("DELETE FROM sync_status WHERE api_name = ? AND trade_date = ?", [api_name, trade_date])
    con.execute(
        "INSERT INTO sync_status VALUES (?, ?, ?, ?, ?, ?)",
        [api_name, trade_date, status, row_count, error, datetime.now()],
    )


def fundamental_status_ok(
    con: duckdb.DuckDBPyConnection,
    api_name: str,
    ts_code: str,
    start_date: date,
    end_date: date,
) -> bool:
    """Return whether a fundamental endpoint was already synced for one symbol."""
    row = con.execute(
        """
        SELECT status
        FROM fundamental_sync_status
        WHERE api_name = ?
          AND ts_code = ?
          AND start_date = ?
          AND end_date = ?
        ORDER BY synced_at DESC
        LIMIT 1
        """,
        [api_name, ts_code, start_date, end_date],
    ).fetchone()
    return bool(row and row[0] == "ok")


def mark_fundamental_status(
    con: duckdb.DuckDBPyConnection,
    api_name: str,
    ts_code: str,
    start_date: date,
    end_date: date,
    status: str,
    row_count: int,
    error: str | None = None,
) -> None:
    """Persist symbol-level fundamental sync status."""
    con.execute(
        """
        DELETE FROM fundamental_sync_status
        WHERE api_name = ?
          AND ts_code = ?
          AND start_date = ?
          AND end_date = ?
        """,
        [api_name, ts_code, start_date, end_date],
    )
    con.execute(
        "INSERT INTO fundamental_sync_status VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [api_name, ts_code, start_date, end_date, status, row_count, error, datetime.now()],
    )


def insert_dataframe(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    delete_where: str,
    delete_params: list[Any],
) -> int:
    """Replace one partition of a DuckDB table with a DataFrame."""
    con.execute(f"DELETE FROM {table} WHERE {delete_where}", delete_params)
    if df.empty:
        return 0
    con.register("_sync_df", df)
    con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _sync_df")
    con.unregister("_sync_df")
    return len(df)


SOURCE_DATE_COLUMNS = {
    "trade_date",
    "cal_date",
    "pretrade_date",
    "list_date",
    "delist_date",
    "base_date",
    "ann_date",
    "f_ann_date",
    "end_date",
    "pre_date",
    "actual_date",
    "modify_date",
    "in_date",
    "out_date",
}

SOURCE_TEXT_COLUMNS = {
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "market",
    "publisher",
    "category",
    "exchange",
    "curr_type",
    "list_status",
    "is_hs",
    "index_code",
    "con_code",
    "l1_code",
    "l1_name",
    "l2_code",
    "l2_name",
    "l3_code",
    "l3_name",
    "report_type",
    "comp_type",
    "end_type",
    "update_flag",
    "is_new",
}


def report_period_ends(start_date: date, end_date: date) -> list[date]:
    """Return quarterly report period end dates spanning the date range."""
    periods: list[date] = []
    for year in range(start_date.year - 1, end_date.year + 1):
        for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            period = date(year, month, day)
            if period <= end_date:
                periods.append(period)
    return periods


def normalize_source_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common TuShare source columns while keeping all returned fields."""
    columns: dict[str, pd.Series] = {}
    for column in df.columns:
        series = df[column]
        if column in SOURCE_DATE_COLUMNS:
            columns[column] = to_date_series(series)
        elif column in SOURCE_TEXT_COLUMNS:
            columns[column] = series
            continue
        elif pd.api.types.is_numeric_dtype(series):
            columns[column] = series.astype("float64")
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            if int(numeric.notna().sum()) == int(series.notna().sum()):
                columns[column] = numeric.astype("float64")
            else:
                columns[column] = series
    columns["synced_at"] = pd.Series([datetime.now()] * len(df), index=df.index)
    return pd.DataFrame(columns)


def should_widen_to_double(existing_type: str, incoming_type: str) -> bool:
    """Return whether an existing numeric column should be widened to DOUBLE."""
    existing = existing_type.upper()
    incoming = incoming_type.upper()
    return incoming == "DOUBLE" and existing not in {"DOUBLE", "FLOAT", "REAL"}


def ensure_dynamic_table(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    """Create a table from DataFrame schema and add newly returned columns."""
    if df.empty:
        return

    quoted_table = quote_identifier(table)
    con.register("_schema_df", df)
    try:
        con.execute(f"CREATE TABLE IF NOT EXISTS {quoted_table} AS SELECT * FROM _schema_df WHERE false")
        existing_info = {row[1]: str(row[2]) for row in con.execute(f"PRAGMA table_info({quoted_table})").fetchall()}
        schema_rows = con.execute("DESCRIBE SELECT * FROM _schema_df").fetchall()
        for column, column_type, *_ in schema_rows:
            if column not in existing_info:
                con.execute(
                    f"ALTER TABLE {quoted_table} ADD COLUMN {quote_identifier(column)} {column_type}"
                )
            elif should_widen_to_double(existing_info[column], str(column_type)):
                con.execute(
                    f"ALTER TABLE {quoted_table} ALTER COLUMN {quote_identifier(column)} TYPE DOUBLE"
                )
    finally:
        con.unregister("_schema_df")


def replace_dynamic_table(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Replace a dynamic source table with all returned columns."""
    if df.empty:
        con.execute(f"CREATE TABLE IF NOT EXISTS {quote_identifier(table)} (synced_at TIMESTAMP)")
        con.execute(f"DELETE FROM {quote_identifier(table)}")
        return 0

    normalized = normalize_source_dataframe(df)
    ensure_dynamic_table(con, table, normalized)
    quoted_table = quote_identifier(table)
    con.execute(f"DELETE FROM {quoted_table}")
    con.register("_sync_df", normalized)
    try:
        con.execute(f"INSERT INTO {quoted_table} BY NAME SELECT * FROM _sync_df")
    finally:
        con.unregister("_sync_df")
    return len(normalized)


def replace_dynamic_partition(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    delete_where: str,
    delete_params: list[Any],
) -> int:
    """Replace one partition in a dynamic source table."""
    quoted_table = quote_identifier(table)

    if df.empty:
        if table_exists(con, table):
            con.execute(f"DELETE FROM {quoted_table} WHERE {delete_where}", delete_params)
        return 0

    normalized = normalize_source_dataframe(df)
    ensure_dynamic_table(con, table, normalized)
    con.execute(f"DELETE FROM {quoted_table} WHERE {delete_where}", delete_params)
    con.register("_sync_df", normalized)
    try:
        con.execute(f"INSERT INTO {quoted_table} BY NAME SELECT * FROM _sync_df")
    finally:
        con.unregister("_sync_df")
    return len(normalized)


def sync_static_tables(
    con: duckdb.DuckDBPyConnection,
    client: TushareClient,
    start_date: date,
    end_date: date,
) -> list[date]:
    """Sync stock_basic and trade calendar, returning open dates."""
    stock_frames: list[pd.DataFrame] = []
    for status in ["L", "D", "P"]:
        df = client.call(
            "stock_basic",
            params={"exchange": "", "list_status": status},
            fields=STOCK_BASIC_FIELDS,
        )
        if df.empty:
            continue
        if "list_status" not in df.columns:
            df["list_status"] = status
        df["synced_at"] = datetime.now()
        for column in ["list_date", "delist_date"]:
            if column in df.columns:
                df[column] = to_date_series(df[column])
        stock_frames.append(df)

    stock_basic = pd.concat(stock_frames, ignore_index=True) if stock_frames else pd.DataFrame()
    con.execute("DELETE FROM stock_basic")
    if not stock_basic.empty:
        con.register("_stock_basic", stock_basic)
        con.execute("INSERT INTO stock_basic BY NAME SELECT * FROM _stock_basic")
        con.unregister("_stock_basic")

    calendar = client.call(
        "trade_cal",
        params={
            "exchange": "SSE",
            "start_date": start_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
        },
        fields=TRADE_CAL_FIELDS,
    )
    if not calendar.empty:
        calendar["cal_date"] = to_date_series(calendar["cal_date"])
        calendar["pretrade_date"] = to_date_series(calendar["pretrade_date"])
        calendar["is_open"] = pd.to_numeric(calendar["is_open"], errors="coerce").fillna(0).astype(int)
        calendar["synced_at"] = datetime.now()
        con.execute("DELETE FROM trade_calendar WHERE cal_date >= ? AND cal_date <= ?", [start_date, end_date])
        con.register("_trade_calendar", calendar)
        con.execute("INSERT INTO trade_calendar BY NAME SELECT * FROM _trade_calendar")
        con.unregister("_trade_calendar")

    rows = con.execute(
        """
        SELECT cal_date
        FROM trade_calendar
        WHERE is_open = 1 AND cal_date >= ? AND cal_date <= ?
        ORDER BY cal_date
        """,
        [start_date, end_date],
    ).fetchall()
    return [row[0] for row in rows]


def sync_daily_raw(
    con: duckdb.DuckDBPyConnection,
    client: TushareClient,
    trade_date: date,
    force: bool,
    use_slow_daily_fallback: bool,
) -> int:
    """Sync raw daily bars and adj_factor for one trade date."""
    api_name = "daily_raw"
    if not force and status_ok(con, api_name, trade_date):
        return -1

    date_text = trade_date.strftime("%Y%m%d")
    try:
        daily = client.call("daily", params={"trade_date": date_text}, fields=DAILY_FIELDS)
        if daily.empty:
            if not use_slow_daily_fallback:
                raise RuntimeError(f"TuShare daily returned no rows for open trading date {date_text}")
            merged = client.call("stk_factor", params={"trade_date": date_text}, fields=STK_FACTOR_FIELDS)
        else:
            adj = client.call("adj_factor", params={"trade_date": date_text}, fields=ADJ_FIELDS)
            merged = daily.merge(adj, on=["ts_code", "trade_date"], how="left") if not adj.empty else daily
        if merged.empty:
            raise RuntimeError(f"No daily raw rows for {date_text} from daily or stk_factor")
        if not merged.empty:
            merged["trade_date"] = to_date_series(merged["trade_date"])
            normalize_numeric(merged, ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "adj_factor"])
            merged["amount"] = merged["amount"] * 1000.0
            merged["synced_at"] = datetime.now()
        rows = insert_dataframe(con, "daily_raw", merged, "trade_date = ?", [trade_date])
        mark_status(con, api_name, trade_date, "ok", rows)
        return rows
    except Exception as exc:
        mark_status(con, api_name, trade_date, "error", 0, str(exc)[:500])
        raise


def sync_daily_basic(
    con: duckdb.DuckDBPyConnection,
    client: TushareClient,
    trade_date: date,
    force: bool,
) -> int:
    """Sync daily_basic for one trade date."""
    api_name = "daily_basic"
    if not force and status_ok(con, api_name, trade_date):
        return -1

    date_text = trade_date.strftime("%Y%m%d")
    try:
        df = client.call("daily_basic", params={"trade_date": date_text}, fields=DAILY_BASIC_FIELDS)
        if not df.empty:
            df["trade_date"] = to_date_series(df["trade_date"])
            normalize_numeric(df, [column for column in df.columns if column not in {"ts_code", "trade_date"}])
            df["synced_at"] = datetime.now()
        rows = insert_dataframe(con, "daily_basic", df, "trade_date = ?", [trade_date])
        mark_status(con, api_name, trade_date, "ok", rows)
        return rows
    except Exception as exc:
        mark_status(con, api_name, trade_date, "error", 0, str(exc)[:500])
        raise


def sync_stk_limit(
    con: duckdb.DuckDBPyConnection,
    client: TushareClient,
    trade_date: date,
    force: bool,
) -> int:
    """Sync daily limit prices for one trade date."""
    api_name = "stk_limit"
    if not force and status_ok(con, api_name, trade_date):
        return -1

    date_text = trade_date.strftime("%Y%m%d")
    try:
        df = client.call("stk_limit", params={"trade_date": date_text}, fields=STK_LIMIT_FIELDS)
        if not df.empty:
            df["trade_date"] = to_date_series(df["trade_date"])
            normalize_numeric(df, ["up_limit", "down_limit", "pre_close"])
            df["synced_at"] = datetime.now()
        rows = insert_dataframe(con, "stk_limit", df, "trade_date = ?", [trade_date])
        mark_status(con, api_name, trade_date, "ok", rows)
        return rows
    except Exception as exc:
        mark_status(con, api_name, trade_date, "error", 0, str(exc)[:500])
        raise


def recompute_qfq(con: duckdb.DuckDBPyConnection) -> None:
    """Recompute qfq prices using latest available adj_factor."""
    con.execute(
        """
        UPDATE daily_raw
        SET
            change = round(close - pre_close, 4),
            pct_chg = round((close - pre_close) / pre_close * 100, 4)
        WHERE (change IS NULL OR pct_chg IS NULL)
          AND close IS NOT NULL
          AND pre_close IS NOT NULL
          AND pre_close != 0
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE latest_adj AS
        SELECT ts_code, arg_max(adj_factor, trade_date) AS latest_adj
        FROM daily_raw
        WHERE adj_factor IS NOT NULL AND adj_factor > 0
        GROUP BY ts_code
        """
    )
    con.execute(
        """
        UPDATE daily_raw AS d
        SET
            open_qfq = round(d.open * d.adj_factor / l.latest_adj, 4),
            high_qfq = round(d.high * d.adj_factor / l.latest_adj, 4),
            low_qfq = round(d.low * d.adj_factor / l.latest_adj, 4),
            close_qfq = round(d.close * d.adj_factor / l.latest_adj, 4)
        FROM latest_adj AS l
        WHERE d.ts_code = l.ts_code
          AND d.adj_factor IS NOT NULL
          AND d.adj_factor > 0
          AND l.latest_adj IS NOT NULL
          AND l.latest_adj > 0
        """
    )


def sync_market(
    con: duckdb.DuckDBPyConnection,
    client: TushareClient,
    start_date: date,
    end_date: date,
    force: bool,
    use_slow_daily_fallback: bool,
) -> None:
    """Sync full-market daily tables into DuckDB."""
    trade_dates = sync_static_tables(con, client, start_date, end_date)
    total = len(trade_dates)
    for index, trade_date in enumerate(trade_dates, start=1):
        raw_rows = sync_daily_raw(con, client, trade_date, force, use_slow_daily_fallback)
        basic_rows = sync_daily_basic(con, client, trade_date, force)
        limit_rows = sync_stk_limit(con, client, trade_date, force)
        if index == 1 or index % 20 == 0 or index == total:
            print(
                f"[sync] {index}/{total} {trade_date} "
                f"raw={raw_rows} basic={basic_rows} limit={limit_rows}",
                flush=True,
            )
    recompute_qfq(con)
    print("[sync] qfq recomputed", flush=True)


def parse_csv_arg(value: str) -> list[str]:
    """Parse a comma-separated CLI argument."""
    return [item.strip() for item in value.split(",") if item.strip()]


def sync_first_priority_data(
    con: duckdb.DuckDBPyConnection,
    client: TushareClient,
    start_date: date,
    end_date: date,
    index_markets: list[str],
    index_codes: list[str],
) -> None:
    """Sync industry and benchmark index source tables with full returned fields."""
    index_basic_frames: list[pd.DataFrame] = []
    for market in index_markets:
        df = client.call_all("index_basic", params={"market": market})
        if not df.empty:
            index_basic_frames.append(df)
        print(f"[priority] index_basic market={market} rows={len(df)}", flush=True)
    index_basic = pd.concat(index_basic_frames, ignore_index=True) if index_basic_frames else pd.DataFrame()
    rows = replace_dynamic_table(con, "index_basic", index_basic)
    print(f"[priority] index_basic total={rows}", flush=True)

    index_member_all = client.call_all("index_member_all")
    rows = replace_dynamic_table(con, "index_member_all", index_member_all)
    print(f"[priority] index_member_all rows={rows}", flush=True)

    start_text = start_date.strftime("%Y%m%d")
    end_text = end_date.strftime("%Y%m%d")
    total_rows = 0
    for index_code in index_codes:
        df = client.call_all(
            "index_weight",
            params={
                "index_code": index_code,
                "start_date": start_text,
                "end_date": end_text,
            },
        )
        rows = replace_dynamic_partition(
            con,
            "index_weight",
            df,
            "index_code = ? AND trade_date >= ? AND trade_date <= ?",
            [index_code, start_date, end_date],
        )
        total_rows += rows
        print(f"[priority] index_weight index={index_code} rows={rows}", flush=True)
    print(f"[priority] index_weight total={total_rows}", flush=True)


def fetch_fundamental_symbols(con: duckdb.DuckDBPyConnection, limit: int | None) -> list[str]:
    """Fetch symbols for fundamental sync."""
    query = """
        SELECT DISTINCT ts_code
        FROM stock_basic
        WHERE ts_code IS NOT NULL
          AND list_status IN ('L', 'D', 'P')
        ORDER BY ts_code
    """
    if limit is not None:
        query += " LIMIT ?"
        rows = con.execute(query, [limit]).fetchall()
    else:
        rows = con.execute(query).fetchall()
    return [str(row[0]) for row in rows]


def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    """Return whether a DuckDB table exists."""
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    return bool(row and row[0])


def replace_symbol_ann_date_partition(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    ts_code: str,
    start_date: date,
    end_date: date,
) -> int:
    """Replace one symbol's announcement-date partition."""
    quoted_table = quote_identifier(table)
    if df.empty:
        if table_exists(con, table):
            con.execute(
                f"DELETE FROM {quoted_table} WHERE ts_code = ? AND ann_date >= ? AND ann_date <= ?",
                [ts_code, start_date, end_date],
            )
        return 0
    return replace_dynamic_partition(
        con,
        table,
        df,
        "ts_code = ? AND ann_date >= ? AND ann_date <= ?",
        [ts_code, start_date, end_date],
    )


def sync_disclosure_dates(
    con: duckdb.DuckDBPyConnection,
    client: TushareClient,
    start_date: date,
    end_date: date,
) -> int:
    """Sync disclosure dates by report period with full returned fields."""
    total_rows = 0
    for period in report_period_ends(start_date, end_date):
        df = client.call_all(
            "disclosure_date",
            params={"end_date": period.strftime("%Y%m%d")},
            fields=DISCLOSURE_DATE_FIELDS,
        )
        rows = replace_dynamic_partition(
            con,
            "disclosure_date",
            df,
            "end_date = ?",
            [period],
        )
        total_rows += rows
        print(f"[fundamental] disclosure_date end_date={period} rows={rows}", flush=True)
    print(f"[fundamental] disclosure_date total={total_rows}", flush=True)
    return total_rows


def sync_fundamental_data(
    con: duckdb.DuckDBPyConnection,
    client: TushareClient,
    start_date: date,
    end_date: date,
    apis: list[str],
    symbol_limit: int | None,
    force: bool,
) -> None:
    """Sync PIT fundamental source tables with full returned fields."""
    sync_disclosure_dates(con, client, start_date, end_date)

    symbols = fetch_fundamental_symbols(con, symbol_limit)
    total_symbols = len(symbols)
    start_text = start_date.strftime("%Y%m%d")
    end_text = end_date.strftime("%Y%m%d")
    valid_apis = [api for api in apis if api in FUNDAMENTAL_APIS]
    invalid_apis = sorted(set(apis) - set(valid_apis))
    if invalid_apis:
        raise ValueError(f"Unsupported fundamental APIs: {', '.join(invalid_apis)}")

    for index, ts_code in enumerate(symbols, start=1):
        symbol_rows: dict[str, int] = {}
        for api_name in valid_apis:
            table = FUNDAMENTAL_APIS[api_name]
            if not force and fundamental_status_ok(con, api_name, ts_code, start_date, end_date):
                symbol_rows[api_name] = -1
                continue

            try:
                df = client.call_all(
                    api_name,
                    params={
                        "ts_code": ts_code,
                        "start_date": start_text,
                        "end_date": end_text,
                    },
                )
                rows = replace_symbol_ann_date_partition(con, table, df, ts_code, start_date, end_date)
                mark_fundamental_status(con, api_name, ts_code, start_date, end_date, "ok", rows)
                symbol_rows[api_name] = rows
            except Exception as exc:
                mark_fundamental_status(
                    con,
                    api_name,
                    ts_code,
                    start_date,
                    end_date,
                    "error",
                    0,
                    str(exc)[:500],
                )
                raise

        if index == 1 or index % 50 == 0 or index == total_symbols:
            parts = " ".join(f"{api}={rows}" for api, rows in symbol_rows.items())
            print(f"[fundamental] {index}/{total_symbols} {ts_code} {parts}", flush=True)


def require_source_tables(con: duckdb.DuckDBPyConnection, tables: list[str]) -> None:
    """Raise if required source tables are missing."""
    missing = [table for table in tables if not table_exists(con, table)]
    if missing:
        raise RuntimeError(f"Missing required source tables: {', '.join(missing)}")


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """Return table column names."""
    return [str(row[1]) for row in con.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()]


def prefixed_select_columns(
    con: duckdb.DuckDBPyConnection,
    table: str,
    alias: str,
    prefix: str,
    exclude: set[str],
) -> str:
    """Build a SELECT list that keeps all table columns with a prefix."""
    columns = []
    for column in table_columns(con, table):
        if column in exclude:
            continue
        columns.append(f"{alias}.{quote_identifier(column)} AS {quote_identifier(prefix + column)}")
    return ",\n            ".join(columns)


def resolve_panel_dates(
    con: duckdb.DuckDBPyConnection,
    panel_start: str | None,
    panel_end: str | None,
) -> tuple[date, date]:
    """Resolve panel date bounds from args or source data."""
    min_date, max_date = con.execute(
        """
        SELECT min(trade_date), max(trade_date)
        FROM daily_raw
        WHERE close_qfq IS NOT NULL
        """
    ).fetchone()
    if min_date is None or max_date is None:
        raise RuntimeError("daily_raw has no qfq rows for research panel")

    start_date = parse_date(panel_start) if panel_start else min_date
    end_date = parse_date(panel_end) if panel_end else max_date
    if start_date > end_date:
        raise ValueError(f"panel start {start_date} is after end {end_date}")
    return start_date, end_date


def build_fundamental_pit_artifacts(con: duckdb.DuckDBPyConnection, latest_date: date) -> None:
    """Build PIT fundamental event view and interval table."""
    require_source_tables(con, ["fundamental_fina_indicator", "disclosure_date"])
    con.execute(
        """
        CREATE OR REPLACE VIEW v_fundamental_indicator_event AS
        WITH joined AS (
            SELECT
                f.*,
                f.ann_date AS financial_ann_date,
                d.ann_date AS disclosure_ann_date,
                d.pre_date AS disclosure_pre_date,
                d.actual_date AS disclosure_actual_date,
                d.modify_date AS disclosure_modify_date,
                d.synced_at AS disclosure_synced_at,
                COALESCE(d.actual_date, f.ann_date) AS pit_date,
                row_number() OVER (
                    PARTITION BY
                        f.ts_code,
                        f.end_date,
                        COALESCE(d.actual_date, f.ann_date)
                    ORDER BY
                        f.ann_date DESC NULLS LAST,
                        d.modify_date DESC NULLS LAST,
                        f.synced_at DESC NULLS LAST
                ) AS dedup_rank
            FROM fundamental_fina_indicator AS f
            LEFT JOIN disclosure_date AS d
                ON f.ts_code = d.ts_code
               AND f.end_date = d.end_date
            WHERE f.ts_code IS NOT NULL
              AND f.end_date IS NOT NULL
              AND COALESCE(d.actual_date, f.ann_date) IS NOT NULL
        )
        SELECT *
        FROM joined
        WHERE dedup_rank = 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE financial_indicator_pit_interval AS
        WITH events AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY ts_code, pit_date
                    ORDER BY end_date DESC NULLS LAST, ann_date DESC NULLS LAST, synced_at DESC NULLS LAST
                ) AS pit_snapshot_rank
            FROM v_fundamental_indicator_event
        ),
        snapshots AS (
            SELECT *
            FROM events
            WHERE pit_snapshot_rank = 1
        ),
        ordered AS (
            SELECT
                *,
                lead(pit_date) OVER (
                    PARTITION BY ts_code
                    ORDER BY pit_date, end_date
                ) AS next_pit_date
            FROM snapshots
        )
        SELECT
            *,
            pit_date AS valid_from,
            COALESCE(CAST(next_pit_date - INTERVAL 1 DAY AS DATE), DATE '{latest_date}') AS valid_to
        FROM ordered
        WHERE pit_date <= DATE '{latest_date}'
        ORDER BY ts_code, valid_from, end_date
        """
    )
    event_rows = con.execute("SELECT count(*) FROM v_fundamental_indicator_event").fetchone()[0]
    interval_rows = con.execute("SELECT count(*) FROM financial_indicator_pit_interval").fetchone()[0]
    print(f"[panel] v_fundamental_indicator_event rows={event_rows}", flush=True)
    print(f"[panel] financial_indicator_pit_interval rows={interval_rows}", flush=True)


def build_industry_index_intervals(con: duckdb.DuckDBPyConnection, latest_date: date) -> None:
    """Build industry and benchmark index membership intervals."""
    require_source_tables(con, ["index_member_all", "index_weight"])
    vt_member_expr = to_vt_symbol_sql("m.ts_code")
    vt_weight_expr = to_vt_symbol_sql("w.con_code")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE stock_industry_interval AS
        SELECT
            m.ts_code,
            {vt_member_expr} AS vt_symbol,
            m.name,
            m.l1_code AS sw_l1_code,
            m.l1_name AS sw_l1_name,
            m.l2_code AS sw_l2_code,
            m.l2_name AS sw_l2_name,
            m.l3_code AS sw_l3_code,
            m.l3_name AS sw_l3_name,
            COALESCE(m.in_date, DATE '1900-01-01') AS valid_from,
            COALESCE(CAST(m.out_date AS DATE), DATE '{latest_date}') AS valid_to,
            m.is_new,
            m.synced_at
        FROM index_member_all AS m
        WHERE m.ts_code IS NOT NULL
        QUALIFY row_number() OVER (
            PARTITION BY m.ts_code, COALESCE(m.in_date, DATE '1900-01-01'), COALESCE(CAST(m.out_date AS DATE), DATE '{latest_date}')
            ORDER BY m.synced_at DESC NULLS LAST
        ) = 1
        ORDER BY ts_code, valid_from
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE index_weight_interval AS
        WITH index_dates AS (
            SELECT
                index_code,
                trade_date AS valid_from,
                lead(trade_date) OVER (PARTITION BY index_code ORDER BY trade_date) AS next_trade_date
            FROM (
                SELECT DISTINCT index_code, trade_date
                FROM index_weight
                WHERE index_code IS NOT NULL
                  AND trade_date IS NOT NULL
            )
        )
        SELECT
            w.index_code,
            w.con_code AS ts_code,
            {vt_weight_expr} AS vt_symbol,
            w.trade_date AS weight_date,
            d.valid_from,
            COALESCE(CAST(d.next_trade_date - INTERVAL 1 DAY AS DATE), DATE '{latest_date}') AS valid_to,
            w.weight,
            w.synced_at
        FROM index_weight AS w
        JOIN index_dates AS d
            ON w.index_code = d.index_code
           AND w.trade_date = d.valid_from
        WHERE w.con_code IS NOT NULL
        ORDER BY w.index_code, w.con_code, d.valid_from
        """
    )
    industry_rows = con.execute("SELECT count(*) FROM stock_industry_interval").fetchone()[0]
    weight_rows = con.execute("SELECT count(*) FROM index_weight_interval").fetchone()[0]
    print(f"[panel] stock_industry_interval rows={industry_rows}", flush=True)
    print(f"[panel] index_weight_interval rows={weight_rows}", flush=True)


def build_research_panel(
    con: duckdb.DuckDBPyConnection,
    lab_path: Path,
    start_date: date,
    end_date: date,
) -> None:
    """Build research_panel_daily Parquet from source and PIT interval tables."""
    require_source_tables(
        con,
        [
            "daily_raw",
            "daily_basic",
            "stk_limit",
            "stock_basic",
            "financial_indicator_pit_interval",
            "stock_industry_interval",
            "index_weight_interval",
        ],
    )
    panel_dir = lab_path / "panel" / "research_panel_daily"
    if panel_dir.exists():
        shutil.rmtree(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)

    vt_daily_expr = to_vt_symbol_sql("d.ts_code")
    financial_columns = prefixed_select_columns(
        con,
        "financial_indicator_pit_interval",
        "fi",
        "fi_",
        {"ts_code"},
    )
    financial_select = f",\n            {financial_columns}" if financial_columns else ""
    panel_path = quote_path(panel_dir)
    con.execute(
        f"""
        COPY (
            WITH coverage AS (
                SELECT
                    ts_code,
                    count(*) AS bar_rows,
                    max(trade_date) AS latest_bar_date
                FROM daily_raw
                WHERE open_qfq IS NOT NULL
                  AND high_qfq IS NOT NULL
                  AND low_qfq IS NOT NULL
                  AND close_qfq IS NOT NULL
                GROUP BY ts_code
            )
            SELECT
                d.trade_date,
                year(d.trade_date) AS year,
                d.ts_code,
                {vt_daily_expr} AS vt_symbol,
                d.open_qfq AS open,
                d.high_qfq AS high,
                d.low_qfq AS low,
                d.close_qfq AS close,
                d.open AS raw_open,
                d.high AS raw_high,
                d.low AS raw_low,
                d.close AS raw_close,
                d.pre_close,
                d.change,
                d.pct_chg,
                d.vol AS volume,
                d.amount AS turnover,
                d.adj_factor,
                d.synced_at AS daily_synced_at,
                b.turnover_rate,
                b.turnover_rate_f,
                b.volume_ratio,
                b.pe,
                b.pe_ttm,
                b.pb,
                b.ps,
                b.ps_ttm,
                b.dv_ratio,
                b.dv_ttm,
                b.total_share,
                b.float_share,
                b.free_share,
                b.total_mv,
                b.circ_mv,
                l.up_limit,
                l.down_limit,
                l.pre_close AS limit_pre_close,
                s.name,
                s.area,
                s.industry,
                s.market,
                s.list_date,
                s.delist_date,
                s.exchange,
                s.curr_type,
                s.list_status,
                s.is_hs,
                cov.bar_rows,
                cov.latest_bar_date,
                cov.bar_rows >= 1000 AS in_a_share_ge1000,
                cov.bar_rows >= 1000
                    AND s.list_status = 'L'
                    AND cov.latest_bar_date = DATE '{end_date}' AS in_a_share_active_ge1000,
                d.close IS NOT NULL
                    AND l.up_limit IS NOT NULL
                    AND d.close >= l.up_limit * 0.999 AS is_limit_up,
                d.close IS NOT NULL
                    AND l.down_limit IS NOT NULL
                    AND d.close <= l.down_limit * 1.001 AS is_limit_down,
                s.list_date IS NOT NULL
                    AND d.trade_date >= s.list_date
                    AND (s.delist_date IS NULL OR d.trade_date <= s.delist_date) AS is_list_life_valid,
                d.close_qfq IS NOT NULL
                    AND d.close_qfq > 0
                    AND d.amount IS NOT NULL
                    AND d.amount > 0 AS has_valid_bar,
                ind.sw_l1_code,
                ind.sw_l1_name,
                ind.sw_l2_code,
                ind.sw_l2_name,
                ind.sw_l3_code,
                ind.sw_l3_name,
                hs300.weight IS NOT NULL AS is_hs300,
                hs300.weight AS hs300_weight,
                zz500.weight IS NOT NULL AS is_zz500,
                zz500.weight AS zz500_weight,
                zz1000.weight IS NOT NULL AS is_zz1000,
                zz1000.weight AS zz1000_weight
                {financial_select}
            FROM daily_raw AS d
            LEFT JOIN daily_basic AS b
                ON d.ts_code = b.ts_code
               AND d.trade_date = b.trade_date
            LEFT JOIN stk_limit AS l
                ON d.ts_code = l.ts_code
               AND d.trade_date = l.trade_date
            LEFT JOIN stock_basic AS s
                ON d.ts_code = s.ts_code
            LEFT JOIN coverage AS cov
                ON d.ts_code = cov.ts_code
            LEFT JOIN stock_industry_interval AS ind
                ON d.ts_code = ind.ts_code
               AND d.trade_date BETWEEN ind.valid_from AND ind.valid_to
            LEFT JOIN index_weight_interval AS hs300
                ON d.ts_code = hs300.ts_code
               AND hs300.index_code = '000300.SH'
               AND d.trade_date BETWEEN hs300.valid_from AND hs300.valid_to
            LEFT JOIN index_weight_interval AS zz500
                ON d.ts_code = zz500.ts_code
               AND zz500.index_code = '000905.SH'
               AND d.trade_date BETWEEN zz500.valid_from AND zz500.valid_to
            LEFT JOIN index_weight_interval AS zz1000
                ON d.ts_code = zz1000.ts_code
               AND zz1000.index_code = '000852.SH'
               AND d.trade_date BETWEEN zz1000.valid_from AND zz1000.valid_to
            LEFT JOIN financial_indicator_pit_interval AS fi
                ON d.ts_code = fi.ts_code
               AND d.trade_date BETWEEN fi.valid_from AND fi.valid_to
            WHERE d.trade_date >= DATE '{start_date}'
              AND d.trade_date <= DATE '{end_date}'
              AND d.close_qfq IS NOT NULL
            ORDER BY d.trade_date, d.ts_code
        ) TO '{panel_path}' (
            FORMAT PARQUET,
            PARTITION_BY (year),
            OVERWRITE_OR_IGNORE TRUE
        )
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW research_panel_daily AS
        SELECT *
        FROM read_parquet('{quote_path(panel_dir / "**" / "*.parquet")}', hive_partitioning = true)
        """
    )
    stats = con.execute(
        "SELECT count(*), count(distinct ts_code), min(trade_date), max(trade_date) FROM research_panel_daily"
    ).fetchone()
    manifest = {
        "panel": "research_panel_daily",
        "format": "parquet",
        "path": str(panel_dir),
        "source_db": str(lab_path / "source" / "tushare_full.duckdb"),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "rows": int(stats[0]),
        "symbols": int(stats[1]),
        "min_trade_date": str(stats[2]),
        "max_trade_date": str(stats[3]),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    (lab_path / "panel" / "research_panel_daily_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[panel] research_panel_daily rows={stats[0]} symbols={stats[1]} "
        f"range={stats[2]}..{stats[3]} path={panel_dir}",
        flush=True,
    )


def build_research_artifacts(
    con: duckdb.DuckDBPyConnection,
    lab_path: Path,
    panel_start: str | None,
    panel_end: str | None,
) -> None:
    """Build PIT/intermediate tables and research panel Parquet."""
    start_date, end_date = resolve_panel_dates(con, panel_start, panel_end)
    build_fundamental_pit_artifacts(con, end_date)
    build_industry_index_intervals(con, end_date)
    build_research_panel(con, lab_path, start_date, end_date)


def export_alpha_lab(con: duckdb.DuckDBPyConnection, lab_path: Path, min_factor_bars: int) -> None:
    """Export DuckDB source data to AlphaLab daily/component/contract files."""
    daily_dir = lab_path / "daily"
    component_dir = lab_path / "component"
    source_dir = lab_path / "source"
    daily_dir.mkdir(parents=True, exist_ok=True)
    component_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows = con.execute(
        """
        SELECT
            d.ts_code,
            min(d.trade_date) AS start_date,
            max(d.trade_date) AS end_date,
            count(*) AS rows,
            max(CASE WHEN b.list_status = 'L' THEN 1 ELSE 0 END) AS is_listed
        FROM daily_raw d
        LEFT JOIN stock_basic b ON d.ts_code = b.ts_code
        WHERE d.open_qfq IS NOT NULL
          AND d.high_qfq IS NOT NULL
          AND d.low_qfq IS NOT NULL
          AND d.close_qfq IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    latest_date = con.execute("SELECT max(trade_date) FROM daily_raw WHERE close_qfq IS NOT NULL").fetchone()[0]

    coverage: list[dict[str, Any]] = []
    exported_symbols: list[str] = []
    factor_symbols: list[str] = []
    active_symbols: list[str] = []
    for ts_code, start_date, end_date, rows, is_listed in coverage_rows:
        vt_symbol = to_vt_symbol(ts_code)
        out = daily_dir / f"{vt_symbol}.parquet"
        con.execute(
            f"""
            COPY (
                SELECT
                    CAST(trade_date AS TIMESTAMP) AS datetime,
                    open_qfq AS open,
                    high_qfq AS high,
                    low_qfq AS low,
                    close_qfq AS close,
                    vol AS volume,
                    amount AS turnover,
                    CAST(0 AS DOUBLE) AS open_interest
                FROM daily_raw
                WHERE ts_code = ?
                  AND open_qfq IS NOT NULL
                  AND high_qfq IS NOT NULL
                  AND low_qfq IS NOT NULL
                  AND close_qfq IS NOT NULL
                ORDER BY trade_date
            ) TO '{quote_path(out)}' (FORMAT PARQUET)
            """,
            [ts_code],
        )
        in_factor = int(rows) >= min_factor_bars
        in_active = bool(in_factor and is_listed and end_date == latest_date)
        exported_symbols.append(vt_symbol)
        if in_factor:
            factor_symbols.append(vt_symbol)
        if in_active:
            active_symbols.append(vt_symbol)
        coverage.append(
            {
                "vt_symbol": vt_symbol,
                "ts_code": ts_code,
                "rows": int(rows),
                "start": str(start_date),
                "end": str(end_date),
                "in_factor_universe": in_factor,
                "in_active_factor_universe": in_active,
                "is_listed": bool(is_listed),
            }
        )

    calendar_rows = con.execute(
        """
        SELECT cal_date
        FROM trade_calendar
        WHERE is_open = 1 AND cal_date <= ?
        ORDER BY cal_date
        """,
        [latest_date],
    ).fetchall()
    trading_dates = [row[0] for row in calendar_rows]
    if not trading_dates:
        raise RuntimeError("No trading dates found for AlphaLab export")

    con.execute(
        f"""
        COPY (
            SELECT cal_date AS trade_date
            FROM trade_calendar
            WHERE is_open = 1 AND cal_date <= DATE '{latest_date}'
            ORDER BY cal_date
        ) TO '{quote_path(component_dir / "trading_dates.parquet")}' (FORMAT PARQUET)
        """
    )

    vt_symbol_expr = to_vt_symbol_sql("d.ts_code")
    coverage_sql = f"""
        SELECT
            d.ts_code,
            {vt_symbol_expr} AS vt_symbol,
            min(d.trade_date) AS first_bar_date,
            max(d.trade_date) AS latest_bar_date,
            count(*) AS bar_rows,
            max(CASE WHEN b.list_status = 'L' THEN 1 ELSE 0 END) AS is_listed
        FROM daily_raw d
        LEFT JOIN stock_basic b ON d.ts_code = b.ts_code
        WHERE d.open_qfq IS NOT NULL
          AND d.high_qfq IS NOT NULL
          AND d.low_qfq IS NOT NULL
          AND d.close_qfq IS NOT NULL
        GROUP BY d.ts_code
    """
    min_trade_date = trading_dates[0]

    con.execute(
        f"""
        COPY (
            WITH trading_days AS (
                SELECT
                    cal_date,
                    row_number() OVER (ORDER BY cal_date) AS trade_idx
                FROM trade_calendar
                WHERE is_open = 1 AND cal_date <= DATE '{latest_date}'
            ),
            member_days AS (
                SELECT
                    d.trade_date,
                    td.trade_idx,
                    {vt_symbol_expr} AS vt_symbol,
                    CASE WHEN b.list_status = 'L' THEN true ELSE false END AS is_listed
                FROM daily_raw d
                JOIN trading_days td ON d.trade_date = td.cal_date
                LEFT JOIN stock_basic b ON d.ts_code = b.ts_code
                WHERE d.close_qfq IS NOT NULL
            ),
            grouped AS (
                SELECT
                    trade_date,
                    trade_idx,
                    vt_symbol,
                    is_listed,
                    trade_idx - row_number() OVER (PARTITION BY vt_symbol ORDER BY trade_idx) AS grp
                FROM member_days
            )
            SELECT
                'a_share_all' AS universe,
                vt_symbol,
                min(trade_date) AS valid_from,
                max(trade_date) AS valid_to,
                count(*) AS interval_days,
                max(is_listed) AS is_listed,
                max(trade_date) AS latest_bar_date,
                current_timestamp AS generated_at
            FROM grouped
            GROUP BY vt_symbol, grp
            ORDER BY vt_symbol, valid_from
        ) TO '{quote_path(component_dir / "a_share_all.parquet")}' (FORMAT PARQUET)
        """
    )

    con.execute(
        f"""
        COPY (
            WITH coverage AS ({coverage_sql})
            SELECT
                'a_share_ge1000' AS universe,
                vt_symbol,
                DATE '{min_trade_date}' AS valid_from,
                DATE '{latest_date}' AS valid_to,
                bar_rows AS interval_days,
                CAST(is_listed AS BOOLEAN) AS is_listed,
                latest_bar_date,
                current_timestamp AS generated_at
            FROM coverage
            WHERE bar_rows >= {min_factor_bars}
            ORDER BY vt_symbol
        ) TO '{quote_path(component_dir / "a_share_ge1000.parquet")}' (FORMAT PARQUET)
        """
    )

    con.execute(
        f"""
        COPY (
            WITH coverage AS ({coverage_sql})
            SELECT
                'a_share_active_ge1000' AS universe,
                vt_symbol,
                DATE '{min_trade_date}' AS valid_from,
                DATE '{latest_date}' AS valid_to,
                bar_rows AS interval_days,
                CAST(is_listed AS BOOLEAN) AS is_listed,
                latest_bar_date,
                current_timestamp AS generated_at
            FROM coverage
            WHERE bar_rows >= {min_factor_bars}
              AND is_listed = 1
              AND latest_bar_date = DATE '{latest_date}'
            ORDER BY vt_symbol
        ) TO '{quote_path(component_dir / "a_share_active_ge1000.parquet")}' (FORMAT PARQUET)
        """
    )

    contracts = {
        vt_symbol: {
            "long_rate": 0.001,
            "short_rate": 0.0015,
            "size": 1,
            "pricetick": 0.01,
        }
        for vt_symbol in exported_symbols
    }
    (lab_path / "contract.json").write_text(json.dumps(contracts, indent=4, ensure_ascii=False), encoding="utf-8")
    (lab_path / "daily_import_coverage.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    (lab_path / "universe_summary.json").write_text(
        json.dumps(
            {
                "source_db": str(source_dir / "tushare_full.duckdb"),
                "source": "tushare.pro",
                "price_adjustment": "qfq",
                "latest_date": str(latest_date),
                "trading_dates": len(trading_dates),
                "exported_symbols": len(exported_symbols),
                "factor_universe": "a_share_ge1000",
                "factor_symbols": len(factor_symbols),
                "active_universe": "a_share_active_ge1000",
                "active_factor_symbols": len(active_symbols),
                "dynamic_universe": "a_share_all",
                "min_factor_bars": min_factor_bars,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"[export] symbols={len(exported_symbols)} factor={len(factor_symbols)} "
        f"active={len(active_symbols)} dates={len(trading_dates)} latest={latest_date}",
        flush=True,
    )


def main() -> int:
    """Run sync/export workflow."""
    args = parse_args()
    require_sync_dependencies()
    lab_path = Path(args.lab_path)
    db_path = Path(args.db_path) if args.db_path else lab_path / "source" / "tushare_full.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    init_db(con)

    if not args.export_only:
        env_files = [Path(args.env_file)] if args.env_file else [lab_path / ".env", Path(".env")]
        token = load_token(env_files)
        client = TushareClient(token, args.rate_limit, args.timeout, args.retries)
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        if not args.skip_market_daily:
            sync_market(
                con,
                client,
                start_date,
                end_date,
                bool(args.force),
                bool(args.use_slow_daily_fallback),
            )
        if args.sync_priority_data:
            sync_first_priority_data(
                con,
                client,
                start_date,
                end_date,
                parse_csv_arg(str(args.index_markets)),
                parse_csv_arg(str(args.index_codes)),
            )
        if args.sync_fundamental_data:
            fundamental_start_date = parse_date(args.fundamental_start_date or args.start_date)
            fundamental_end_date = parse_date(args.fundamental_end_date or args.end_date)
            sync_fundamental_data(
                con,
                client,
                fundamental_start_date,
                fundamental_end_date,
                parse_csv_arg(str(args.fundamental_apis)),
                args.fundamental_symbol_limit,
                bool(args.force),
            )

    if not args.sync_only:
        export_alpha_lab(con, lab_path, int(args.min_factor_bars))

    if args.build_research_panel:
        build_research_artifacts(
            con,
            lab_path,
            args.panel_start_date,
            args.panel_end_date,
        )

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
