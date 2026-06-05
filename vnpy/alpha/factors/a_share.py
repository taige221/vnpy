from __future__ import annotations

import polars as pl

from .generic import classic_price_expressions


PANEL_SCAN_WINDOWS: tuple[int, ...] = (20, 60, 120)
PANEL_SCAN_HORIZONS: tuple[int, ...] = (5, 10, 20)
PANEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "turnover_rate",
    "pb",
    "pe_ttm",
    "circ_mv",
    "fi_roe",
    "fi_netprofit_yoy",
    "fi_or_yoy",
    "fi_debt_to_assets",
    "fi_grossprofit_margin",
)
FUNDAMENTAL_EXPRESSIONS: tuple[str, ...] = (
    "cs_rank(fi_roe)",
    "cs_rank(fi_netprofit_yoy)",
    "cs_rank(fi_or_yoy)",
    "-1 * cs_rank(fi_debt_to_assets)",
    "cs_rank(fi_grossprofit_margin)",
    "-1 * cs_rank(pb)",
)
CANDIDATE_BASE_FACTORS: tuple[str, ...] = (
    "rev_20",
    "rev_60",
    "anti_turnover_surge_120",
    "neg_price_volume_corr_20",
    "growth_netprofit_yoy",
)
CANDIDATE_COMPOSITES: dict[str, tuple[str, ...]] = {
    "combo_core_4": (
        "rev_20",
        "anti_turnover_surge_120",
        "neg_price_volume_corr_20",
        "growth_netprofit_yoy",
    ),
    "combo_price_flow_3": (
        "rev_20",
        "anti_turnover_surge_120",
        "neg_price_volume_corr_20",
    ),
    "combo_reversal_flow_4": (
        "rev_20",
        "rev_60",
        "anti_turnover_surge_120",
        "neg_price_volume_corr_20",
    ),
    "combo_low_redundancy_3": (
        "rev_20",
        "neg_price_volume_corr_20",
        "growth_netprofit_yoy",
    ),
}


def first_pass_factor_expressions(windows: tuple[int, ...] = PANEL_SCAN_WINDOWS) -> tuple[list[str], list[pl.Expr]]:
    names: list[str] = []
    expressions: list[pl.Expr] = []

    for window in windows:
        momentum = pl.col("close") / pl.col("close").shift(window).over("vt_symbol") - 1.0
        ret_vol = pl.col("ret_1d").rolling_std(window_size=window).over("vt_symbol")
        turnover_ratio = pl.col("turnover") / pl.col("turnover").rolling_mean(window_size=window).over("vt_symbol") - 1.0
        pv_corr = pl.rolling_corr(pl.col("close"), pl.col("volume"), window_size=window).over("vt_symbol")

        for name, expr in (
            (f"mom_{window}", momentum),
            (f"rev_{window}", -momentum),
            (f"low_vol_{window}", -ret_vol),
            (f"ret_vol_{window}", ret_vol),
            (f"turnover_surge_{window}", turnover_ratio),
            (f"neg_price_volume_corr_{window}", -pv_corr),
        ):
            names.append(name)
            expressions.append(expr.alias(name))

    for name, expr in (
        ("value_pb", -pl.col("pb")),
        ("value_pe_ttm", -pl.col("pe_ttm")),
        ("quality_roe", pl.col("fi_roe")),
        ("growth_netprofit_yoy", pl.col("fi_netprofit_yoy")),
        ("growth_or_yoy", pl.col("fi_or_yoy")),
        ("low_debt_to_assets", -pl.col("fi_debt_to_assets")),
    ):
        names.append(name)
        expressions.append(expr.alias(name))

    return names, expressions


def candidate_factor_expressions() -> list[pl.Expr]:
    momentum_20 = pl.col("close") / pl.col("close").shift(20).over("vt_symbol") - 1.0
    momentum_60 = pl.col("close") / pl.col("close").shift(60).over("vt_symbol") - 1.0
    turnover_surge_120 = pl.col("turnover") / pl.col("turnover").rolling_mean(window_size=120).over("vt_symbol") - 1.0
    price_volume_corr_20 = pl.rolling_corr(pl.col("close"), pl.col("volume"), window_size=20).over("vt_symbol")
    return [
        (-momentum_20).alias("rev_20"),
        (-momentum_60).alias("rev_60"),
        (-turnover_surge_120).alias("anti_turnover_surge_120"),
        (-price_volume_corr_20).alias("neg_price_volume_corr_20"),
        pl.col("fi_netprofit_yoy").alias("growth_netprofit_yoy"),
    ]


def combo_price_flow3_base_exprs() -> list[pl.Expr]:
    momentum_20 = pl.col("close") / pl.col("close").shift(20).over("vt_symbol") - 1.0
    turnover_surge_120 = pl.col("turnover") / pl.col("turnover").rolling_mean(window_size=120).over("vt_symbol") - 1.0
    price_volume_corr_20 = pl.rolling_corr(pl.col("close"), pl.col("volume"), window_size=20).over("vt_symbol")
    return [
        (-momentum_20).alias("rev_20"),
        (-turnover_surge_120).alias("anti_turnover_surge_120"),
        (-price_volume_corr_20).alias("neg_price_volume_corr_20"),
    ]


def combo_price_flow3_rank_exprs() -> list[pl.Expr]:
    return [
        (pl.col(column).rank().over("trade_date") / pl.len().over("trade_date")).alias(f"{column}_rank")
        for column in ("rev_20", "anti_turnover_surge_120", "neg_price_volume_corr_20")
    ]


def combo_price_flow3_expr() -> pl.Expr:
    return pl.mean_horizontal(
        "rev_20_rank",
        "anti_turnover_surge_120_rank",
        "neg_price_volume_corr_20_rank",
    ).alias("combo_price_flow_3")


def panel_expressions(windows: tuple[int, ...]) -> list[str]:
    return classic_price_expressions(windows) + list(FUNDAMENTAL_EXPRESSIONS)
