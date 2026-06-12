"""Canonical buy-point definitions for A5/TrendRSI snapshots.

The buy-point layer is signal semantics only. Ranking factors and portfolio
execution rules should consume these fields, not redefine them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


BUY_POINT_TYPE_L1 = "L1_reversal_start"
BUY_POINT_TYPE_L2 = "L2_reversal_confirm"
BUY_POINT_TYPE_L3 = "L3_pullback_relay"

BUY_POINT_TYPES: tuple[str, ...] = (
    BUY_POINT_TYPE_L1,
    BUY_POINT_TYPE_L2,
    BUY_POINT_TYPE_L3,
)

BUY_POINT_LABELS: dict[str, str] = {
    BUY_POINT_TYPE_L1: "反转启动",
    BUY_POINT_TYPE_L2: "反转确认",
    BUY_POINT_TYPE_L3: "回踩接力",
}

BUY_POINT_SUBTYPE_FIELDS: dict[str, tuple[str, ...]] = {
    BUY_POINT_TYPE_L1: ("start_type",),
    BUY_POINT_TYPE_L2: ("l2_confirm_type",),
    BUY_POINT_TYPE_L3: ("l3_filter",),
}


def _clean_text(value: Any) -> str:
    """Return a stable string for snapshot categorical fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def buy_point_label(buy_point_type: Any) -> str:
    """Return a reader-facing label for a buy-point type."""
    text = _clean_text(buy_point_type)
    return BUY_POINT_LABELS.get(text, text)


def resolve_buy_point_subtype(
    buy_point_type: Any,
    *,
    start_type: Any = "",
    l2_confirm_type: Any = "",
    l3_filter: Any = "",
) -> str:
    """Resolve the subtype used for same-type decomposition studies."""
    kind = _clean_text(buy_point_type)
    if kind == BUY_POINT_TYPE_L1:
        return _clean_text(start_type) or "unknown_start_type"
    if kind == BUY_POINT_TYPE_L2:
        return _clean_text(l2_confirm_type) or "confirm_raw"
    if kind == BUY_POINT_TYPE_L3:
        return _clean_text(l3_filter) or "raw"
    return "unknown_buy_point_type"


def derive_buy_point_subtype(row: Mapping[str, Any]) -> str:
    """Resolve the subtype from a snapshot-like row."""
    return resolve_buy_point_subtype(
        row.get("buy_point_type", ""),
        start_type=row.get("start_type", ""),
        l2_confirm_type=row.get("l2_confirm_type", ""),
        l3_filter=row.get("l3_filter", ""),
    )


def add_buy_point_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical buy-point label and subtype columns exist."""
    out = frame.copy()
    if "buy_point_type" not in out.columns:
        raise ValueError("snapshot missing buy_point_type")

    out["buy_point_label"] = out["buy_point_type"].map(buy_point_label)
    out["buy_point_subtype"] = out.apply(derive_buy_point_subtype, axis=1)
    return out
