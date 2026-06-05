from datetime import datetime

import polars as pl


def build_eligibility_from_source_frame(
    source: pl.DataFrame,
    *,
    start_date: datetime,
    end_date: datetime,
    entry_lag: int,
    exclude_new_listing_days: int,
    include_current_st: bool,
    include_entry_limit: bool,
    min_amount: float,
    min_circ_mv: float,
) -> pl.DataFrame:
    """Build a tradable A-share eligibility panel from source rows."""
    df: pl.DataFrame = (
        source.with_columns(
            pl.col("datetime").cast(pl.Datetime),
            pl.col("list_date").cast(pl.Datetime),
            pl.col("delist_date").cast(pl.Datetime),
        )
        .sort(["vt_symbol", "datetime"])
        .with_columns(
            (
                pl.col("raw_close").is_not_null()
                & pl.col("up_limit").is_not_null()
                & (pl.col("raw_close") >= pl.col("up_limit") * 0.999)
            ).alias("is_limit_up"),
            (
                pl.col("raw_close").is_not_null()
                & pl.col("down_limit").is_not_null()
                & (pl.col("raw_close") <= pl.col("down_limit") * 1.001)
            ).alias("is_limit_down"),
        )
        .with_columns(
            pl.col("is_limit_up").shift(-entry_lag).over("vt_symbol").fill_null(False).alias("entry_limit_up"),
            pl.col("is_limit_down").shift(-entry_lag).over("vt_symbol").fill_null(False).alias("entry_limit_down"),
        )
    )

    eligible: pl.Expr = (
        pl.col("datetime").is_between(start_date, end_date)
        & pl.col("raw_close").is_not_null()
        & (pl.col("raw_close") > 0)
        & pl.col("amount").is_not_null()
        & pl.col("list_date").is_not_null()
        & (
            pl.col("delist_date").is_null()
            | (pl.col("datetime") <= pl.col("delist_date"))
        )
    )

    if exclude_new_listing_days > 0:
        eligible &= pl.col("datetime") >= pl.col("list_date").dt.offset_by(f"{exclude_new_listing_days}d")
    if not include_current_st:
        eligible &= ~pl.col("name").fill_null("").str.contains("ST|退")
    if not include_entry_limit:
        eligible &= ~(pl.col("entry_limit_up") | pl.col("entry_limit_down"))
    if min_amount > 0:
        eligible &= pl.col("amount") >= min_amount
    if min_circ_mv > 0:
        eligible &= pl.col("circ_mv") >= min_circ_mv

    return df.select("datetime", "vt_symbol", eligible.fill_null(False).alias("eligible"))
