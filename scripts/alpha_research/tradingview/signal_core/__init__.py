"""Shared signal semantics for TradingView A-share research."""

from .buy_points import (
    BUY_POINT_LABELS,
    BUY_POINT_SUBTYPE_FIELDS,
    BUY_POINT_TYPE_L1,
    BUY_POINT_TYPE_L2,
    BUY_POINT_TYPE_L3,
    BUY_POINT_TYPES,
    add_buy_point_columns,
    buy_point_label,
    derive_buy_point_subtype,
    resolve_buy_point_subtype,
)

__all__ = [
    "BUY_POINT_LABELS",
    "BUY_POINT_SUBTYPE_FIELDS",
    "BUY_POINT_TYPE_L1",
    "BUY_POINT_TYPE_L2",
    "BUY_POINT_TYPE_L3",
    "BUY_POINT_TYPES",
    "add_buy_point_columns",
    "buy_point_label",
    "derive_buy_point_subtype",
    "resolve_buy_point_subtype",
]
