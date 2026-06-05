from __future__ import annotations

from dataclasses import dataclass


DEFAULT_DOUBLE_BOTTOM_WINDOWS: tuple[int, ...] = (40, 60, 120)
DEFAULT_DOUBLE_BOTTOM_VOLUME_WINDOWS: tuple[int, ...] = (5, 20)
DEFAULT_DOUBLE_BOTTOM_CONFIRM_WINDOWS: tuple[int, ...] = (3, 5)
DEFAULT_DOUBLE_BOTTOM_TOLERANCE_PCT: float = 0.02
DEFAULT_DOUBLE_BOTTOM_RECLAIM_PCT: float = 0.0
EPSILON: str = "1e-12"


@dataclass(frozen=True)
class DoubleBottomFactorSpec:
    """Factor extracted from double-bottom strategy structure."""

    name: str
    expression: str
    signal_type: str
    description: str
    source_rule: str


def _dedupe_specs(specs: list[DoubleBottomFactorSpec]) -> list[DoubleBottomFactorSpec]:
    seen: set[str] = set()
    result: list[DoubleBottomFactorSpec] = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        result.append(spec)
    return result


def _support(window: int) -> str:
    return f"ts_min(ts_delay(low, 1), {window})"


def _neckline(window: int) -> str:
    return f"ts_max(ts_delay(high, 1), {window})"


def _volume_ratio(window: int) -> str:
    return f"volume / (ts_mean(ts_delay(volume, 1), {window}) + {EPSILON})"


def _low_similarity(window: int) -> str:
    support = _support(window)
    return f"-1 * abs(low / (({support}) + {EPSILON}) - 1)"


def _break_below_low_quality(window: int) -> str:
    support = _support(window)
    return f"-1 * abs(ts_less(low, ({support})) / (({support}) + {EPSILON}) - 1)"


def _neckline_height(window: int) -> str:
    support = _support(window)
    neckline = _neckline(window)
    return f"(({neckline}) - ({support})) / (({support}) + {EPSILON})"


def _support_touch_density(window: int, tolerance_pct: float) -> str:
    support = _support(window)
    return (
        f"ts_sum(low <= (({support}) * {1.0 + tolerance_pct:.8f}), {window})"
        f" / {float(window):.1f}"
    )


def _prior_low_age(window: int, confirm_window: int) -> str:
    return f"ts_argmin(ts_delay(low, {confirm_window}), {window}) / {float(window):.1f}"


def _entry_above_low(window: int) -> str:
    support = _support(window)
    return f"close / (({support}) + {EPSILON}) - 1"


def _neckline_breakout_strength(window: int) -> str:
    neckline = _neckline(window)
    return f"close / (({neckline}) + {EPSILON}) - 1"


def _neckline_breakout_cross(window: int) -> str:
    neckline = _neckline(window)
    return f"(ts_delay(close, 1) <= ({neckline})) * (close > ({neckline}))"


def _second_low_confirm_mask(window: int, confirm_window: int, tolerance_pct: float, confirm_pct: float) -> str:
    support = _support(window)
    second_low_high = f"ts_max(ts_delay(high, 1), {confirm_window})"
    return (
        f"(low <= (({support}) * {1.0 + tolerance_pct:.8f}))"
        " * (close >= open)"
        f" * (close >= (({second_low_high}) * {1.0 + confirm_pct:.8f}))"
    )


def _neckline_retest_mask(window: int, tolerance_pct: float, reclaim_pct: float) -> str:
    neckline = _neckline(window)
    return (
        f"(low <= (({neckline}) * {1.0 + tolerance_pct:.8f}))"
        f" * (close >= (({neckline}) * {1.0 + reclaim_pct:.8f}))"
        " * (close >= open)"
    )


def _neckline_depth_quality(window: int) -> str:
    neckline = _neckline(window)
    return f"-1 * abs(low / (({neckline}) + {EPSILON}) - 1)"


def double_bottom_factor_specs(
    windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_WINDOWS,
    volume_windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_VOLUME_WINDOWS,
    confirm_windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_CONFIRM_WINDOWS,
    tolerance_pct: float = DEFAULT_DOUBLE_BOTTOM_TOLERANCE_PCT,
    reclaim_pct: float = DEFAULT_DOUBLE_BOTTOM_RECLAIM_PCT,
    confirm_close_above_low_high_pct: float = 0.0,
) -> list[DoubleBottomFactorSpec]:
    """Return double-bottom structure factors converted to expression DSL candidates."""
    specs: list[DoubleBottomFactorSpec] = []

    for volume_window in volume_windows:
        specs.append(
            DoubleBottomFactorSpec(
                name=f"double_bottom_volume_ratio_{volume_window}",
                expression=_volume_ratio(volume_window),
                signal_type="shared",
                description="Current volume versus trailing pre-signal average volume.",
                source_rule="double_bottom_min_volume_ratio / second_low_min_volume_ratio",
            )
        )

    primary_volume = _volume_ratio(volume_windows[0])

    for window in windows:
        low_similarity = _low_similarity(window)
        break_quality = _break_below_low_quality(window)
        neckline_height = _neckline_height(window)
        support_touch_density = _support_touch_density(window, tolerance_pct)
        entry_above_low = _entry_above_low(window)
        neckline_breakout_strength = _neckline_breakout_strength(window)
        neckline_breakout_cross = _neckline_breakout_cross(window)
        neckline_depth_quality = _neckline_depth_quality(window)
        neckline_retest_mask = _neckline_retest_mask(window, tolerance_pct, reclaim_pct)
        neckline_retest_strength = f"({neckline_retest_mask}) * ({neckline_breakout_strength})"

        specs.extend(
            [
                DoubleBottomFactorSpec(
                    name=f"double_bottom_low_similarity_{window}",
                    expression=low_similarity,
                    signal_type="shared",
                    description="Higher values mean the current low is closer to the prior support low.",
                    source_rule="double_bottom_low_diff_pct / second_low_max_low_diff_pct",
                ),
                DoubleBottomFactorSpec(
                    name=f"double_bottom_break_below_low_quality_{window}",
                    expression=break_quality,
                    signal_type="shared",
                    description="Penalizes breaking materially below the prior support low.",
                    source_rule="double_bottom_second_low_max_break_below_low_pct",
                ),
                DoubleBottomFactorSpec(
                    name=f"double_bottom_neckline_height_{window}",
                    expression=neckline_height,
                    signal_type="shared",
                    description="Neckline height above support, normalized by support.",
                    source_rule="double_bottom_min_neckline_height_pct",
                ),
                DoubleBottomFactorSpec(
                    name=f"double_bottom_support_touch_density_{window}",
                    expression=support_touch_density,
                    signal_type="shared",
                    description="Recent density of bars touching the double-bottom support zone.",
                    source_rule="first_low / second_low structure",
                ),
                DoubleBottomFactorSpec(
                    name=f"double_bottom_entry_above_low_{window}",
                    expression=entry_above_low,
                    signal_type="second_low_confirm",
                    description="Close distance above prior support low.",
                    source_rule="double_bottom_second_low_max_entry_above_low_pct",
                ),
                DoubleBottomFactorSpec(
                    name=f"double_bottom_neckline_breakout_strength_{window}",
                    expression=f"({neckline_breakout_cross}) * ({neckline_breakout_strength})",
                    signal_type="neckline_breakout",
                    description="First close crossing above neckline, weighted by close distance above neckline.",
                    source_rule="prior_close <= neckline and breakout_close >= neckline",
                ),
                DoubleBottomFactorSpec(
                    name=f"double_bottom_neckline_breakout_quality_{window}",
                    expression=(
                        f"({neckline_breakout_cross}) * ("
                        f"cs_rank({neckline_breakout_strength})"
                        f" + cs_rank({neckline_height})"
                        f" + cs_rank({low_similarity})"
                        f" + cs_rank({support_touch_density})"
                        f" + cs_rank({primary_volume})"
                        ") / 5"
                    ),
                    signal_type="neckline_breakout",
                    description="Composite neckline breakout quality from reclaim strength, height, low similarity, touches, and volume.",
                    source_rule="double_bottom breakout candidate score",
                ),
                DoubleBottomFactorSpec(
                    name=f"double_bottom_neckline_retest_strength_{window}",
                    expression=neckline_retest_strength,
                    signal_type="neckline_retest",
                    description="Reclaim of neckline after touching the neckline zone.",
                    source_rule="double_bottom_retest touched_zone / reclaimed / rebound_bar",
                ),
                DoubleBottomFactorSpec(
                    name=f"double_bottom_neckline_retest_quality_{window}",
                    expression=(
                        f"({neckline_retest_mask}) * ("
                        f"cs_rank({neckline_depth_quality})"
                        f" + cs_rank({neckline_breakout_strength})"
                        f" + cs_rank({neckline_height})"
                        f" + cs_rank({support_touch_density})"
                        f" + cs_rank({primary_volume})"
                        ") / 5"
                    ),
                    signal_type="neckline_retest",
                    description="Composite neckline retest quality from depth, reclaim strength, height, support touches, and volume.",
                    source_rule="double_bottom_retest quality components",
                ),
            ]
        )

        for confirm_window in confirm_windows:
            second_low_mask = _second_low_confirm_mask(
                window,
                confirm_window,
                tolerance_pct,
                confirm_close_above_low_high_pct,
            )
            prior_low_age = _prior_low_age(window, confirm_window)
            specs.extend(
                [
                    DoubleBottomFactorSpec(
                        name=f"double_bottom_prior_low_age_{window}_{confirm_window}",
                        expression=prior_low_age,
                        signal_type="shared",
                        description="Normalized age of the prior delayed low inside the lookback window.",
                        source_rule="double_bottom_min_separation_days / second_low_min_separation_days",
                    ),
                    DoubleBottomFactorSpec(
                        name=f"double_bottom_second_low_confirm_strength_{window}_{confirm_window}",
                        expression=f"({second_low_mask}) * ({entry_above_low})",
                        signal_type="second_low_confirm",
                        description="Second-low confirmation event weighted by close distance above support.",
                        source_rule="double_bottom_second_low_confirm close_above_low_high",
                    ),
                    DoubleBottomFactorSpec(
                        name=f"double_bottom_second_low_quality_{window}_{confirm_window}",
                        expression=(
                            f"({second_low_mask}) * ("
                            f"cs_rank({low_similarity})"
                            f" + cs_rank({break_quality})"
                            f" + cs_rank({neckline_height})"
                            f" + cs_rank({prior_low_age})"
                            f" + cs_rank({primary_volume})"
                            ") / 5"
                        ),
                        signal_type="second_low_confirm",
                        description="Composite second-low quality from low similarity, no-new-low behavior, neckline height, separation, and volume.",
                        source_rule="double_bottom_second_low quality components",
                    ),
                ]
            )

    return _dedupe_specs(specs)


def double_bottom_expressions(
    windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_WINDOWS,
    volume_windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_VOLUME_WINDOWS,
    confirm_windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_CONFIRM_WINDOWS,
    signal_type: str | None = None,
) -> list[str]:
    """Return only expression strings for statistical alpha loops."""
    specs = double_bottom_factor_specs(
        windows=windows,
        volume_windows=volume_windows,
        confirm_windows=confirm_windows,
    )
    if signal_type:
        specs = [spec for spec in specs if spec.signal_type in {signal_type, "shared"}]
    return [spec.expression for spec in specs]


def double_bottom_factor_names(
    windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_WINDOWS,
    volume_windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_VOLUME_WINDOWS,
    confirm_windows: tuple[int, ...] = DEFAULT_DOUBLE_BOTTOM_CONFIRM_WINDOWS,
    signal_type: str | None = None,
) -> list[str]:
    """Return factor names in the same order as ``double_bottom_expressions``."""
    specs = double_bottom_factor_specs(
        windows=windows,
        volume_windows=volume_windows,
        confirm_windows=confirm_windows,
    )
    if signal_type:
        specs = [spec for spec in specs if spec.signal_type in {signal_type, "shared"}]
    return [spec.name for spec in specs]


__all__ = [
    "DEFAULT_DOUBLE_BOTTOM_CONFIRM_WINDOWS",
    "DEFAULT_DOUBLE_BOTTOM_RECLAIM_PCT",
    "DEFAULT_DOUBLE_BOTTOM_TOLERANCE_PCT",
    "DEFAULT_DOUBLE_BOTTOM_VOLUME_WINDOWS",
    "DEFAULT_DOUBLE_BOTTOM_WINDOWS",
    "DoubleBottomFactorSpec",
    "double_bottom_expressions",
    "double_bottom_factor_names",
    "double_bottom_factor_specs",
]
