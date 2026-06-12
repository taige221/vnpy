"""Fetch Baostock 5-minute bars for PRE-L1 minute validation.

The first intended use is the PRE_L1 ``tomorrow_l1`` set: signal at T close,
formal L1 on T+1, and fetch T+1 through T+4 trading days for execution/path
checks.  The script is resumable and writes a request manifest next to the
cached parquet files.
"""

from __future__ import annotations

import argparse
import math
import signal
import socket
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVENTS_PATH = (
    "scripts/alpha_research/tradingview/reports/pre_l1_near_start_study/"
    "pre_l1_near_start_events.parquet"
)
DEFAULT_OUTPUT_DIR = "lab/a_share_research/minute/baostock_5m"
DEFAULT_FIELDS = "date,time,code,open,high,low,close,volume,amount,adjustflag"


class OperationTimeout(RuntimeError):
    """Raised when a Baostock operation exceeds the configured timeout."""


@dataclass(frozen=True)
class FetchRequest:
    """One Baostock request."""

    vt_symbol: str
    bs_code: str
    start_date: str
    end_date: str
    output_path: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Fetch Baostock 5m bars for PRE-L1 events")
    parser.add_argument("--events-path", default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--plan-path", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scope", choices=["tomorrow_l1", "pre_l1_negative_sample"], default="tomorrow_l1")
    parser.add_argument("--frequency", default="5")
    parser.add_argument("--adjustflag", default="3")
    parser.add_argument("--fields", default=DEFAULT_FIELDS)
    parser.add_argument("--storage-format", choices=["parquet", "csv", "pickle"], default="parquet")
    parser.add_argument("--hold-trading-days", type=int, default=4)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--socket-timeout", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--patch-socket-timeout", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    """Resolve a path from repository root unless already absolute."""
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def normalize_date(values: Any) -> pd.Series:
    """Normalize date-like values."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def vt_symbol_to_baostock(vt_symbol: str) -> str | None:
    """Convert local vt_symbol to Baostock code."""
    if vt_symbol.endswith(".SSE"):
        return f"sh.{vt_symbol.split('.')[0]}"
    if vt_symbol.endswith(".SZSE"):
        return f"sz.{vt_symbol.split('.')[0]}"
    return None


def load_trade_dates(events: pd.DataFrame) -> list[pd.Timestamp]:
    """Build the trading-date calendar needed for event windows."""
    dates = pd.concat(
        [
            normalize_date(events["signal_date"]),
            normalize_date(events["entry_date"]),
        ],
        ignore_index=True,
    )
    return sorted(pd.Timestamp(value).normalize() for value in dates.dropna().unique())


def expand_required_dates(
    events: pd.DataFrame,
    *,
    hold_trading_days: int,
) -> pd.DataFrame:
    """Expand selected events into unique vt_symbol + trading_date rows."""
    if hold_trading_days <= 0:
        raise ValueError("--hold-trading-days must be positive")

    trade_dates = load_trade_dates(events)
    if not trade_dates:
        return pd.DataFrame(columns=["vt_symbol", "trade_date"])
    date_index = {date: index for index, date in enumerate(trade_dates)}

    rows: list[dict[str, Any]] = []
    selected = events[["vt_symbol", "entry_date"]].dropna().drop_duplicates()
    for row in selected.itertuples(index=False):
        entry_date = pd.Timestamp(row.entry_date).normalize()
        start_index = date_index.get(entry_date)
        if start_index is None:
            continue
        for offset in range(hold_trading_days):
            index = start_index + offset
            if index >= len(trade_dates):
                break
            rows.append({"vt_symbol": str(row.vt_symbol), "trade_date": trade_dates[index]})
    out = pd.DataFrame(rows).drop_duplicates()
    if out.empty:
        return pd.DataFrame(columns=["vt_symbol", "trade_date"])
    out = out.sort_values(["vt_symbol", "trade_date"]).reset_index(drop=True)
    return out


def contiguous_ranges(dates: list[pd.Timestamp], calendar: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group required trading dates into contiguous trading-date ranges."""
    if not dates:
        return []
    date_index = {date: index for index, date in enumerate(calendar)}
    ordered = sorted(pd.Timestamp(date).normalize() for date in dates)
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = previous = ordered[0]
    previous_index = date_index[previous]
    for date in ordered[1:]:
        index = date_index[date]
        if index == previous_index + 1:
            previous = date
            previous_index = index
            continue
        ranges.append((start, previous))
        start = previous = date
        previous_index = index
    ranges.append((start, previous))
    return ranges


def build_requests(required_dates: pd.DataFrame, output_dir: Path) -> list[FetchRequest]:
    """Build Baostock request ranges."""
    if required_dates.empty:
        return []
    calendar = sorted(pd.Timestamp(value).normalize() for value in required_dates["trade_date"].unique())
    requests: list[FetchRequest] = []
    root = output_dir / "frequency=5m"
    for vt_symbol, group in required_dates.groupby("vt_symbol", sort=True):
        bs_code = vt_symbol_to_baostock(str(vt_symbol))
        if not bs_code:
            continue
        symbol_dir = root / f"vt_symbol={str(vt_symbol).replace('.', '_')}"
        ranges = contiguous_ranges(list(group["trade_date"]), calendar)
        for start_date, end_date in ranges:
            start_text = start_date.strftime("%Y-%m-%d")
            end_text = end_date.strftime("%Y-%m-%d")
            output_path = symbol_dir / f"{start_text}_{end_text}.parquet"
            requests.append(
                FetchRequest(
                    vt_symbol=str(vt_symbol),
                    bs_code=bs_code,
                    start_date=start_text,
                    end_date=end_text,
                    output_path=output_path,
                )
            )
    return requests


def load_requests_from_plan(plan_path: Path) -> list[FetchRequest]:
    """Load Baostock requests from a previously generated fetch plan CSV."""
    plan = pd.read_csv(plan_path)
    requests: list[FetchRequest] = []
    for row in plan.to_dict("records"):
        requests.append(
            FetchRequest(
                vt_symbol=str(row["vt_symbol"]),
                bs_code=str(row["bs_code"]),
                start_date=str(row["start_date"]),
                end_date=str(row["end_date"]),
                output_path=Path(str(row["output_path"])),
            )
        )
    return requests


def storage_path(path: Path, storage_format: str) -> Path:
    """Return the concrete cache path for the selected storage format."""
    suffix = {
        "parquet": ".parquet",
        "csv": ".csv",
        "pickle": ".pkl",
    }[storage_format]
    return path.with_suffix(suffix)


def write_frame(frame: pd.DataFrame, path: Path, storage_format: str) -> None:
    """Write a fetched minute frame."""
    if storage_format == "parquet":
        frame.to_parquet(path, index=False)
    elif storage_format == "csv":
        frame.to_csv(path, index=False)
    elif storage_format == "pickle":
        frame.to_pickle(path)
    else:
        raise ValueError(f"unsupported storage format: {storage_format}")


def run_with_timeout(seconds: float, func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a callable with a process-level alarm timeout."""
    if seconds <= 0:
        return func(*args, **kwargs)

    def handle_timeout(signum: int, frame: Any) -> None:
        raise OperationTimeout(f"operation timed out after {seconds:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def load_selected_events(args: argparse.Namespace) -> pd.DataFrame:
    """Load and filter event rows."""
    events = pd.read_parquet(repo_path(args.events_path))
    events["signal_date"] = normalize_date(events["signal_date"])
    events["entry_date"] = normalize_date(events["entry_date"])
    if args.start_date:
        events = events[events["entry_date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        events = events[events["entry_date"] <= pd.Timestamp(args.end_date)]
    if args.scope == "tomorrow_l1":
        events = events[events["tomorrow_l1"].fillna(False).astype(bool)]
    return events.reset_index(drop=True)


def read_baostock_result(result: Any, fields: str) -> pd.DataFrame:
    """Read Baostock ResultData without using pandas append."""
    columns = [column.strip() for column in fields.split(",")]
    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=columns)


def fetch_request(request: FetchRequest, *, fields: str, frequency: str, adjustflag: str) -> pd.DataFrame:
    """Fetch one request from Baostock."""
    import baostock as bs

    result = bs.query_history_k_data_plus(
        request.bs_code,
        fields,
        start_date=request.start_date,
        end_date=request.end_date,
        frequency=frequency,
        adjustflag=adjustflag,
    )
    if result.error_code != "0":
        raise RuntimeError(f"{request.bs_code} {request.start_date}-{request.end_date}: {result.error_msg}")
    frame = read_baostock_result(result, fields)
    if frame.empty:
        return frame
    frame.insert(0, "vt_symbol", request.vt_symbol)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    return frame


def write_manifest(output_dir: Path, rows: list[dict[str, Any]], name: str) -> Path:
    """Write a manifest CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def patch_baostock_socket_timeout(timeout: float) -> None:
    """Patch Baostock's socket helper so stalled reads do not hang forever."""
    import baostock.common.contants as cons
    import baostock.common.context as context
    import baostock.util.socketutil as socketutil

    def connect_with_timeout(self: Any) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((cons.BAOSTOCK_SERVER_IP, cons.BAOSTOCK_SERVER_PORT))
        except Exception:
            print("服务器连接失败，请稍后再试。")
            sock = None
        context.default_socket = sock

    socketutil.SocketUtil.connect = connect_with_timeout


def main() -> None:
    """Run the minute fetch."""
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    if args.plan_path:
        events_count: int | str = "from_plan"
        symbol_days_count: int | str = "from_plan"
        plan_path = repo_path(args.plan_path)
        requests = load_requests_from_plan(plan_path)
    else:
        events = load_selected_events(args)
        required_dates = expand_required_dates(events, hold_trading_days=args.hold_trading_days)
        requests = build_requests(required_dates, output_dir)
        events_count = len(events)
        symbol_days_count = len(required_dates)
        plan_path = output_dir / f"fetch_plan_{args.scope}.csv"
    requests = [
        replace(request, output_path=storage_path(request.output_path, args.storage_format))
        for request in requests
    ]
    manifest_rows = [
        {
            "vt_symbol": request.vt_symbol,
            "bs_code": request.bs_code,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "output_path": str(request.output_path),
            "exists": request.output_path.exists(),
        }
        for request in requests
    ]
    if not args.plan_path:
        plan_path = write_manifest(output_dir, manifest_rows, f"fetch_plan_{args.scope}.csv")
    pending = [request for request in requests if args.overwrite or not request.output_path.exists()]
    if args.max_requests is not None:
        pending = pending[: args.max_requests]

    print(
        "minute_fetch_plan "
        f"scope={args.scope} events={events_count} symbol_days={symbol_days_count} "
        f"requests={len(requests)} pending={len(pending)} plan={plan_path}",
        flush=True,
    )
    if args.dry_run:
        return

    import baostock as bs

    if args.patch_socket_timeout:
        patch_baostock_socket_timeout(args.socket_timeout)
    login = run_with_timeout(args.request_timeout, bs.login)
    print(f"baostock_login error_code={login.error_code} error_msg={login.error_msg}", flush=True)
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")

    status_rows: list[dict[str, Any]] = []
    try:
        for index, request in enumerate(pending, start=1):
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            status: dict[str, Any] = {
                "vt_symbol": request.vt_symbol,
                "bs_code": request.bs_code,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "output_path": str(request.output_path),
            }
            try:
                frame = run_with_timeout(
                    args.request_timeout,
                    fetch_request,
                    request,
                    fields=args.fields,
                    frequency=args.frequency,
                    adjustflag=args.adjustflag,
                )
                write_frame(frame, request.output_path, args.storage_format)
                status.update({"status": "ok", "rows": len(frame), "error": ""})
            except Exception as exc:  # noqa: BLE001
                status.update({"status": "error", "rows": math.nan, "error": str(exc)})
                try:
                    bs.logout()
                except Exception:
                    pass
                try:
                    login = run_with_timeout(args.request_timeout, bs.login)
                    print(
                        f"baostock_relogin error_code={login.error_code} "
                        f"error_msg={login.error_msg}",
                        flush=True,
                    )
                except Exception as login_exc:  # noqa: BLE001
                    print(f"baostock_relogin_error {login_exc}", flush=True)
            status_rows.append(status)
            progress_every = max(1, args.progress_every)
            if index % progress_every == 0 or index == len(pending):
                ok_count = sum(row["status"] == "ok" for row in status_rows)
                error_count = sum(row["status"] == "error" for row in status_rows)
                print(
                    f"progress requests={index}/{len(pending)} ok={ok_count} errors={error_count}",
                    flush=True,
                )
                write_manifest(output_dir, status_rows, f"fetch_status_{args.scope}.csv")
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    status_path = write_manifest(output_dir, status_rows, f"fetch_status_{args.scope}.csv")
    print(f"wrote {status_path}", flush=True)


if __name__ == "__main__":
    main()
