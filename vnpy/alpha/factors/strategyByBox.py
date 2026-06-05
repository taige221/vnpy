from __future__ import annotations

from dataclasses import dataclass


DEFAULT_BOX_WINDOWS: tuple[int, ...] = (20, 30, 60)
DEFAULT_VOLUME_WINDOWS: tuple[int, ...] = (5, 20)
DEFAULT_BREAKOUT_RETEST_WINDOWS: tuple[int, ...] = (5,)
DEFAULT_BOX_TOLERANCE_PCT: float = 0.01
DEFAULT_PULLBACK_RECLAIM_PCT: float = 0.0
DEFAULT_RANGE_RECLAIM2_MIN_BOX_HEIGHT_PCT: float = 0.08
DEFAULT_RANGE_RECLAIM2_MIN_CLOSE_ABOVE_RESISTANCE_PCT: float = 0.02
DEFAULT_TRADE_RULE_MIN_RANK_SCORE: float = 50.0
DEFAULT_TRADE_RULE_MAX_HEAT_SCORE: float = 55.0
DEFAULT_TRADE_RULE_MIN_SUPPORT_TOUCHES: int = 3
DEFAULT_TRADE_RULE_MAX_RESISTANCE_TOUCHES: int = 3
DEFAULT_TRADE_RULE_MIN_TURNOVER_RATE: float = 3.0
DEFAULT_TRADE_RULE_MAX_TURNOVER_RATE: float = 6.5
DEFAULT_TRADE_RULE_MIN_MA10_BIAS: float = 0.05
EPSILON: str = "1e-12"
PULLBACK_SIGNAL_TYPES: frozenset[str] = frozenset(
    {"pullback_bounce", "box_reclaim", "breakout_retest", "range_reclaim2"}
)


@dataclass(frozen=True)
class StrategyBoxFactorSpec:
    """Factor extracted from the A-share box strategy rule set."""

    name: str
    expression: str
    signal_type: str
    description: str
    source_rule: str
    required_columns: tuple[str, ...] = ()


def _dedupe_specs(specs: list[StrategyBoxFactorSpec]) -> list[StrategyBoxFactorSpec]:
    seen: set[str] = set()
    result: list[StrategyBoxFactorSpec] = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        result.append(spec)
    return result


def _resistance(window: int) -> str:
    return f"ts_max(ts_delay(high, 1), {window})"


def _support(window: int) -> str:
    return f"ts_min(ts_delay(low, 1), {window})"


def _box_height_pct(window: int) -> str:
    resistance = _resistance(window)
    support = _support(window)
    return f"(({resistance}) - ({support})) / (({support}) + {EPSILON})"


def _volume_ratio(window: int) -> str:
    return f"volume / (ts_mean(ts_delay(volume, 1), {window}) + {EPSILON})"


def _breakout_strength(window: int) -> str:
    resistance = _resistance(window)
    return f"close / (({resistance}) + {EPSILON}) - 1"


def _breakout_cross(window: int) -> str:
    resistance = _resistance(window)
    return f"(ts_delay(close, 1) <= ({resistance})) * (close > ({resistance}))"


def _body_pct() -> str:
    return f"close / (open + {EPSILON}) - 1"


def _upper_shadow_ratio() -> str:
    return f"(high - close) / (high - low + {EPSILON})"


def _resistance_touch_density(window: int, tolerance_pct: float) -> str:
    touch_count = _resistance_touch_count(window, tolerance_pct)
    return f"({touch_count}) / {float(window):.1f}"


def _resistance_touch_count(window: int, tolerance_pct: float) -> str:
    resistance = _resistance(window)
    return f"ts_sum(high >= (({resistance}) * {1.0 - tolerance_pct:.8f}), {window})"


def _support_touch_density(window: int, tolerance_pct: float) -> str:
    touch_count = _support_touch_count(window, tolerance_pct)
    return f"({touch_count}) / {float(window):.1f}"


def _support_touch_count(window: int, tolerance_pct: float) -> str:
    support = _support(window)
    return f"ts_sum(low <= (({support}) * {1.0 + tolerance_pct:.8f}), {window})"


def _box_stack_lift(window: int) -> str:
    resistance = _resistance(window)
    support = _support(window)
    previous_resistance = f"ts_delay(({resistance}), {window})"
    previous_support = f"ts_delay(({support}), {window})"
    return (
        f"((({resistance}) / (({previous_resistance}) + {EPSILON}) - 1)"
        f" + (({support}) / (({previous_support}) + {EPSILON}) - 1)) / 2"
    )


def _pullback_touch(window: int, tolerance_pct: float) -> str:
    resistance = _resistance(window)
    return f"low <= (({resistance}) * {1.0 + tolerance_pct:.8f})"


def _pullback_reclaim(window: int, reclaim_pct: float) -> str:
    resistance = _resistance(window)
    return f"close > (({resistance}) * {1.0 + reclaim_pct:.8f})"


def _pullback_depth_quality(window: int) -> str:
    resistance = _resistance(window)
    return f"-1 * abs(low / (({resistance}) + {EPSILON}) - 1)"


def _reference_depth_quality(reference: str) -> str:
    return f"-1 * abs(low / (({reference}) + {EPSILON}) - 1)"


def _reference_reclaim_strength(reference: str) -> str:
    return f"close / (({reference}) + {EPSILON}) - 1"


def _reference_touch(reference: str, tolerance_pct: float) -> str:
    return f"low <= (({reference}) * {1.0 + tolerance_pct:.8f})"


def _reference_reclaim(reference: str, reclaim_pct: float) -> str:
    return f"close >= (({reference}) * {1.0 + reclaim_pct:.8f})"


def _box_reclaim_mask(window: int, tolerance_pct: float, reclaim_pct: float) -> str:
    resistance = _resistance(window)
    return (
        f"({_reference_touch(resistance, tolerance_pct)})"
        f" * ({_reference_reclaim(resistance, reclaim_pct)})"
        " * (close >= open)"
    )


def _range_reclaim2_mask(
    window: int,
    tolerance_pct: float,
    reclaim_pct: float,
    min_box_height_pct: float,
) -> str:
    box_height_pct = _box_height_pct(window)
    return f"({_box_reclaim_mask(window, tolerance_pct, reclaim_pct)}) * (({box_height_pct}) >= {min_box_height_pct:.8f})"


def _range_reclaim2_confirmed_mask(
    window: int,
    tolerance_pct: float,
    reclaim_pct: float,
    min_box_height_pct: float,
    min_close_above_resistance_pct: float,
) -> str:
    reclaim2_mask = _range_reclaim2_mask(window, tolerance_pct, reclaim_pct, min_box_height_pct)
    reclaim_strength = _breakout_strength(window)
    return f"({reclaim2_mask}) * (({reclaim_strength}) >= {min_close_above_resistance_pct:.8f})"


def _ma_bias(window: int) -> str:
    return f"close / (ts_mean(close, {window}) + {EPSILON}) - 1"


def _prior_breakout_count(window: int, retest_window: int) -> str:
    return f"ts_sum(ts_delay(({_breakout_cross(window)}), 1), {retest_window})"


def _retest_reference_resistance(window: int, retest_window: int) -> str:
    resistance = _resistance(window)
    return f"ts_max(ts_delay(({resistance}), 1), {retest_window})"


def _retest_reference_touch_density(window: int, retest_window: int, tolerance_pct: float) -> str:
    touch_density = _resistance_touch_density(window, tolerance_pct)
    return f"ts_max(ts_delay(({touch_density}), 1), {retest_window})"


def _breakout_retest_mask(window: int, retest_window: int, tolerance_pct: float, reclaim_pct: float) -> str:
    reference = _retest_reference_resistance(window, retest_window)
    prior_breakouts = _prior_breakout_count(window, retest_window)
    return (
        f"({prior_breakouts} > 0)"
        f" * ({_reference_touch(reference, tolerance_pct)})"
        f" * ({_reference_reclaim(reference, reclaim_pct)})"
        " * (close >= open)"
    )


def _filter_specs(specs: list[StrategyBoxFactorSpec], signal_type: str | None) -> list[StrategyBoxFactorSpec]:
    if signal_type is None:
        return specs
    if signal_type == "pullback_bounce":
        return [spec for spec in specs if spec.signal_type in PULLBACK_SIGNAL_TYPES or spec.signal_type == "shared"]
    return [spec for spec in specs if spec.signal_type in {signal_type, "shared"}]


def strategy_box_factor_specs(
    windows: tuple[int, ...] = DEFAULT_BOX_WINDOWS,
    volume_windows: tuple[int, ...] = DEFAULT_VOLUME_WINDOWS,
    retest_windows: tuple[int, ...] = DEFAULT_BREAKOUT_RETEST_WINDOWS,
    box_tolerance_pct: float = DEFAULT_BOX_TOLERANCE_PCT,
    pullback_reclaim_pct: float = DEFAULT_PULLBACK_RECLAIM_PCT,
    range_reclaim2_min_box_height_pct: float = DEFAULT_RANGE_RECLAIM2_MIN_BOX_HEIGHT_PCT,
    range_reclaim2_min_close_above_resistance_pct: float = DEFAULT_RANGE_RECLAIM2_MIN_CLOSE_ABOVE_RESISTANCE_PCT,
    include_trade_rule_factors: bool = False,
) -> list[StrategyBoxFactorSpec]:
    """Return box-strategy factors converted to expression DSL candidates."""
    specs: list[StrategyBoxFactorSpec] = []

    for volume_window in volume_windows:
        volume_expr = _volume_ratio(volume_window)
        specs.append(
            StrategyBoxFactorSpec(
                name=f"box_volume_ratio_{volume_window}",
                expression=volume_expr,
                signal_type="shared",
                description="Current volume versus trailing pre-signal average volume.",
                source_rule="min_volume_ratio / breakout_min_volume_ratio / pullback_min_volume_ratio",
            )
        )

    primary_volume = _volume_ratio(volume_windows[0])
    body_pct = _body_pct()
    upper_shadow = _upper_shadow_ratio()

    for window in windows:
        breakout_strength = _breakout_strength(window)
        breakout_cross = _breakout_cross(window)
        box_height_pct = _box_height_pct(window)
        support_touch_count = _support_touch_count(window, box_tolerance_pct)
        resistance_touch_count = _resistance_touch_count(window, box_tolerance_pct)
        resistance_touch_density = _resistance_touch_density(window, box_tolerance_pct)
        support_touch_density = _support_touch_density(window, box_tolerance_pct)
        stack_lift = _box_stack_lift(window)
        pullback_depth_quality = _pullback_depth_quality(window)
        pullback_mask = _box_reclaim_mask(window, box_tolerance_pct, pullback_reclaim_pct)
        pullback_reclaim_strength = f"({pullback_mask}) * ({breakout_strength})"
        reclaim2_mask = _range_reclaim2_mask(
            window,
            box_tolerance_pct,
            pullback_reclaim_pct,
            range_reclaim2_min_box_height_pct,
        )
        reclaim2_confirmed_mask = _range_reclaim2_confirmed_mask(
            window,
            box_tolerance_pct,
            pullback_reclaim_pct,
            range_reclaim2_min_box_height_pct,
            range_reclaim2_min_close_above_resistance_pct,
        )

        specs.extend(
            [
                StrategyBoxFactorSpec(
                    name=f"box_breakout_strength_{window}",
                    expression=breakout_strength,
                    signal_type="breakout_long",
                    description="Close distance above the trailing box resistance.",
                    source_rule="close_price > box_resistance and breakout close-above-resistance filters",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_breakout_event_strength_{window}",
                    expression=f"({breakout_cross}) * ({breakout_strength})",
                    signal_type="breakout_long",
                    description="First close crossing above resistance, weighted by breakout distance.",
                    source_rule="previous_close <= resistance and close_price > resistance",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_breakout_quality_{window}",
                    expression=(
                        f"({breakout_cross}) * ("
                        f"cs_rank({breakout_strength})"
                        f" + cs_rank({primary_volume})"
                        f" + cs_rank({body_pct})"
                        f" + cs_rank(-1 * ({upper_shadow}))"
                        f" + cs_rank({resistance_touch_density})"
                        f" + cs_rank({stack_lift})"
                        ") / 6"
                    ),
                    signal_type="breakout_long",
                    description="Composite breakout quality from strength, volume, candle body, shadow, touches, and stack lift.",
                    source_rule="breakout quality_score components",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_pullback_reclaim_strength_{window}",
                    expression=pullback_reclaim_strength,
                    signal_type="pullback_bounce",
                    description="Reclaim of resistance after touching the box zone.",
                    source_rule="pullback touched_zone, reclaimed, rebound_bar",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_pullback_depth_quality_{window}",
                    expression=pullback_depth_quality,
                    signal_type="pullback_bounce",
                    description="Higher values mean the pullback low stayed closer to resistance.",
                    source_rule="pullback_low_vs_resistance_pct",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_pullback_quality_{window}",
                    expression=(
                        f"({pullback_mask}) * ("
                        f"cs_rank({pullback_depth_quality})"
                        f" + cs_rank({breakout_strength})"
                        f" + cs_rank({primary_volume})"
                        f" + cs_rank({resistance_touch_density})"
                        ") / 4"
                    ),
                    signal_type="pullback_bounce",
                    description="Composite pullback quality from zone touch, reclaim, depth, volume, and resistance touches.",
                    source_rule="pullback quality_score components",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_reclaim_strength_{window}",
                    expression=pullback_reclaim_strength,
                    signal_type="box_reclaim",
                    description="Current box resistance reclaim after touching the box zone.",
                    source_rule="box_reclaim touched_zone, reclaimed, rebound_bar",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_reclaim_depth_quality_{window}",
                    expression=pullback_depth_quality,
                    signal_type="box_reclaim",
                    description="Higher values mean the reclaim low stayed closer to current box resistance.",
                    source_rule="box_reclaim pullback_low_vs_resistance_pct",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_reclaim_quality_{window}",
                    expression=(
                        f"({pullback_mask}) * ("
                        f"cs_rank({pullback_depth_quality})"
                        f" + cs_rank({breakout_strength})"
                        f" + cs_rank({primary_volume})"
                        f" + cs_rank({resistance_touch_density})"
                        ") / 4"
                    ),
                    signal_type="box_reclaim",
                    description="Composite box-reclaim quality from depth, reclaim strength, volume, and current resistance touches.",
                    source_rule="box_reclaim quality components",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_range_reclaim2_mask_{window}",
                    expression=reclaim2_mask,
                    signal_type="range_reclaim2",
                    description="Large-range reclaim candidate after touching and reclaiming current resistance.",
                    source_rule="pullback_large_box_height_pct + box_reclaim",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_range_reclaim2_confirmed_{window}",
                    expression=reclaim2_confirmed_mask,
                    signal_type="range_reclaim2",
                    description="Large-range reclaim candidate with minimum close-above-resistance confirmation.",
                    source_rule="pullback_large_box_min_close_above_resistance_pct",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_range_reclaim2_close_above_resistance_pct_{window}",
                    expression=breakout_strength,
                    signal_type="range_reclaim2",
                    description="Close distance above current box resistance for reclaim2 diagnostics.",
                    source_rule="pullback_close_above_resistance_pct",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_range_reclaim2_box_height_pct_{window}",
                    expression=box_height_pct,
                    signal_type="range_reclaim2",
                    description="Current box height used to identify large-range reclaim candidates.",
                    source_rule="box_height_pct / pullback_large_box_height_pct",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_range_reclaim2_depth_quality_{window}",
                    expression=pullback_depth_quality,
                    signal_type="range_reclaim2",
                    description="Higher values mean the reclaim2 low stayed closer to resistance.",
                    source_rule="pullback_low_vs_resistance_pct",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_range_reclaim2_support_touches_{window}",
                    expression=support_touch_count,
                    signal_type="range_reclaim2",
                    description="Count of recent support touches inside the reclaim2 range.",
                    source_rule="support_touches",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_range_reclaim2_resistance_touches_{window}",
                    expression=resistance_touch_count,
                    signal_type="range_reclaim2",
                    description="Count of recent resistance touches inside the reclaim2 range.",
                    source_rule="resistance_touches",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_range_reclaim2_strength_{window}",
                    expression=f"({reclaim2_confirmed_mask}) * ({breakout_strength})",
                    signal_type="range_reclaim2",
                    description="Confirmed reclaim2 strength above current box resistance.",
                    source_rule="large-box reclaim strength",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_range_reclaim2_quality_{window}",
                    expression=(
                        f"({reclaim2_confirmed_mask}) * ("
                        f"cs_rank({breakout_strength})"
                        f" + cs_rank({box_height_pct})"
                        f" + cs_rank({pullback_depth_quality})"
                        f" + cs_rank({support_touch_count})"
                        f" + cs_rank(-1 * ({resistance_touch_count}))"
                        ") / 5"
                    ),
                    signal_type="range_reclaim2",
                    description="Composite reclaim2 quality from close confirmation, range width, depth, support touches, and capped resistance touches.",
                    source_rule="range reclaim2 quality components",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_height_compact_{window}",
                    expression=f"-1 * ({box_height_pct})",
                    signal_type="shared",
                    description="Prefers tighter boxes by penalizing high box height.",
                    source_rule="min_box_height_pct / breakout_max_box_height_pct / box height penalties",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_resistance_touch_density_{window}",
                    expression=resistance_touch_density,
                    signal_type="shared",
                    description="Recent density of bars touching resistance.",
                    source_rule="resistance_touches / min_box_touches",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_support_touch_density_{window}",
                    expression=support_touch_density,
                    signal_type="shared",
                    description="Recent density of bars touching support.",
                    source_rule="support_touches",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_stack_lift_{window}",
                    expression=stack_lift,
                    signal_type="shared",
                    description="Average lift of support and resistance versus the prior box window.",
                    source_rule="classify_box_trend / box_stack_lift_pct",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_stack_lift_reversal_{window}",
                    expression=f"-1 * ({stack_lift})",
                    signal_type="shared",
                    description="Reverse of box stack lift; higher values prefer boxes that did not lift versus the prior window.",
                    source_rule="reverse classify_box_trend / box_stack_lift_pct",
                ),
            ]
        )

        for retest_window in retest_windows:
            retest_reference = _retest_reference_resistance(window, retest_window)
            prior_breakouts = _prior_breakout_count(window, retest_window)
            retest_touch_density = _retest_reference_touch_density(window, retest_window, box_tolerance_pct)
            retest_mask = _breakout_retest_mask(window, retest_window, box_tolerance_pct, pullback_reclaim_pct)
            retest_depth_quality = _reference_depth_quality(retest_reference)
            retest_reclaim_strength = _reference_reclaim_strength(retest_reference)

            specs.extend(
                [
                    StrategyBoxFactorSpec(
                        name=f"box_breakout_retest_prior_count_{window}_{retest_window}",
                        expression=prior_breakouts,
                        signal_type="breakout_retest",
                        description="Count of recent breakout crosses available as retest references.",
                        source_rule="_find_prior_breakout_reference breakout candidate scan",
                    ),
                    StrategyBoxFactorSpec(
                        name=f"box_breakout_retest_strength_{window}_{retest_window}",
                        expression=f"({retest_mask}) * ({retest_reclaim_strength})",
                        signal_type="breakout_retest",
                        description="Prior-breakout resistance retest and reclaim strength.",
                        source_rule="breakout_retest touched_zone, reclaimed, rebound_bar",
                    ),
                    StrategyBoxFactorSpec(
                        name=f"box_breakout_retest_depth_quality_{window}_{retest_window}",
                        expression=retest_depth_quality,
                        signal_type="breakout_retest",
                        description="Higher values mean the retest low stayed closer to prior breakout resistance.",
                        source_rule="breakout_retest pullback_low_vs_resistance_pct",
                    ),
                    StrategyBoxFactorSpec(
                        name=f"box_breakout_retest_quality_{window}_{retest_window}",
                        expression=(
                            f"({retest_mask}) * ("
                            f"cs_rank({retest_depth_quality})"
                            f" + cs_rank({retest_reclaim_strength})"
                            f" + cs_rank({primary_volume})"
                            f" + cs_rank({retest_touch_density})"
                            f" + cs_rank({prior_breakouts})"
                            ") / 5"
                        ),
                        signal_type="breakout_retest",
                        description="Composite breakout-retest quality from depth, reclaim strength, volume, reference touches, and recent breakout count.",
                        source_rule="breakout_retest quality components",
                    ),
                ]
            )

    if include_trade_rule_factors:
        specs.extend(
            strategy_box_trade_rule_factor_specs(
                windows=windows,
                box_tolerance_pct=box_tolerance_pct,
                pullback_reclaim_pct=pullback_reclaim_pct,
                range_neutral_min_close_above_resistance_pct=range_reclaim2_min_close_above_resistance_pct,
            )
        )

    return _dedupe_specs(specs)


def strategy_box_trade_rule_factor_specs(
    windows: tuple[int, ...] = DEFAULT_BOX_WINDOWS,
    box_tolerance_pct: float = DEFAULT_BOX_TOLERANCE_PCT,
    pullback_reclaim_pct: float = DEFAULT_PULLBACK_RECLAIM_PCT,
    min_rank_score: float = DEFAULT_TRADE_RULE_MIN_RANK_SCORE,
    max_heat_score: float = DEFAULT_TRADE_RULE_MAX_HEAT_SCORE,
    min_support_touches: int = DEFAULT_TRADE_RULE_MIN_SUPPORT_TOUCHES,
    max_resistance_touches: int = DEFAULT_TRADE_RULE_MAX_RESISTANCE_TOUCHES,
    min_turnover_rate: float = DEFAULT_TRADE_RULE_MIN_TURNOVER_RATE,
    max_turnover_rate: float = DEFAULT_TRADE_RULE_MAX_TURNOVER_RATE,
    min_ma10_bias: float = DEFAULT_TRADE_RULE_MIN_MA10_BIAS,
    range_neutral_min_close_above_resistance_pct: float = DEFAULT_RANGE_RECLAIM2_MIN_CLOSE_ABOVE_RESISTANCE_PCT,
) -> list[StrategyBoxFactorSpec]:
    """Return optional trade-rule factors that depend on enriched A-share panel columns."""
    required_columns = (
        "rank_score",
        "heat_score",
        "turnover_rate",
        "regime_bull_active",
        "regime_range_neutral",
        "regime_weak_defensive",
        "regime_bear_pause",
        "regime_unknown",
    )
    specs: list[StrategyBoxFactorSpec] = [
        StrategyBoxFactorSpec(
            name="box_rule_rank_score",
            expression="rank_score",
            signal_type="trade_rule",
            description="External ranking score used by the original candidate filter.",
            source_rule="rank_score / min_rank_score",
            required_columns=("rank_score",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_rank_pass",
            expression=f"rank_score >= {min_rank_score:.8f}",
            signal_type="trade_rule",
            description="Binary pass flag for the rank-score floor.",
            source_rule="rank_score >= min_rank_score",
            required_columns=("rank_score",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_heat_score",
            expression="heat_score",
            signal_type="trade_rule",
            description="External heat score used to cap overheated candidates.",
            source_rule="heat_score / heat_score_cap",
            required_columns=("heat_score",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_heat_pass",
            expression=f"heat_score <= {max_heat_score:.8f}",
            signal_type="trade_rule",
            description="Binary pass flag for the heat-score cap.",
            source_rule="heat_score <= heat_score_cap",
            required_columns=("heat_score",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_turnover_rate",
            expression="turnover_rate",
            signal_type="trade_rule",
            description="A-share turnover rate in percentage-point units.",
            source_rule="pullback_min_turnover_rate / pullback_max_turnover_rate",
            required_columns=("turnover_rate",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_turnover_rate_band",
            expression=f"(turnover_rate >= {min_turnover_rate:.8f}) * (turnover_rate <= {max_turnover_rate:.8f})",
            signal_type="trade_rule",
            description="Binary pass flag for the preferred turnover-rate band.",
            source_rule="pullback turnover_rate range",
            required_columns=("turnover_rate",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_regime_bull_active",
            expression="regime_bull_active",
            signal_type="trade_rule",
            description="Market-regime one-hot flag for bull_active.",
            source_rule="market_regime == bull_active",
            required_columns=("regime_bull_active",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_regime_range_neutral",
            expression="regime_range_neutral",
            signal_type="trade_rule",
            description="Market-regime one-hot flag for range_neutral.",
            source_rule="market_regime == range_neutral",
            required_columns=("regime_range_neutral",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_regime_weak_defensive",
            expression="regime_weak_defensive",
            signal_type="trade_rule",
            description="Market-regime one-hot flag for weak_defensive.",
            source_rule="market_regime == weak_defensive",
            required_columns=("regime_weak_defensive",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_regime_bear_pause",
            expression="regime_bear_pause",
            signal_type="trade_rule",
            description="Market-regime one-hot flag for bear_pause.",
            source_rule="market_regime == bear_pause",
            required_columns=("regime_bear_pause",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_regime_unknown",
            expression="regime_unknown",
            signal_type="trade_rule",
            description="Market-regime one-hot flag for unknown.",
            source_rule="market_regime == unknown",
            required_columns=("regime_unknown",),
        ),
        StrategyBoxFactorSpec(
            name="box_rule_paused_regime",
            expression="regime_range_neutral + regime_weak_defensive + regime_bear_pause + regime_unknown",
            signal_type="trade_rule",
            description="Paused-regime gate covering range_neutral, weak_defensive, bear_pause, and unknown.",
            source_rule="paused market regimes",
            required_columns=(
                "regime_range_neutral",
                "regime_weak_defensive",
                "regime_bear_pause",
                "regime_unknown",
            ),
        ),
    ]

    ma10_bias = _ma_bias(10)
    specs.extend(
        [
            StrategyBoxFactorSpec(
                name="box_rule_ma10_bias_10",
                expression=ma10_bias,
                signal_type="trade_rule",
                description="Close distance from MA10 in decimal units.",
                source_rule="ma10_bias_pct converted to decimal units",
            ),
            StrategyBoxFactorSpec(
                name="box_rule_ma10_bias_pass_10",
                expression=f"({ma10_bias}) >= {min_ma10_bias:.8f}",
                signal_type="trade_rule",
                description="Binary pass flag for the MA10-bias floor.",
                source_rule="ma10_bias >= threshold",
            ),
        ]
    )

    for window in windows:
        support_touches = _support_touch_count(window, box_tolerance_pct)
        resistance_touches = _resistance_touch_count(window, box_tolerance_pct)
        pullback_shape = _box_reclaim_mask(window, box_tolerance_pct, pullback_reclaim_pct)
        close_above_resistance = _breakout_strength(window)
        ma10_bias = _ma_bias(10)
        rule_mask = (
            f"regime_bull_active * ({pullback_shape})"
            f" * (rank_score >= {min_rank_score:.8f})"
            f" * (heat_score <= {max_heat_score:.8f})"
            f" * (({support_touches}) >= {float(min_support_touches):.1f})"
            f" * (({resistance_touches}) <= {float(max_resistance_touches):.1f})"
            f" * (turnover_rate >= {min_turnover_rate:.8f})"
            f" * (turnover_rate <= {max_turnover_rate:.8f})"
            f" * (({ma10_bias}) >= {min_ma10_bias:.8f})"
        )
        specs.extend(
            [
                StrategyBoxFactorSpec(
                    name=f"box_rule_pullback_shape_{window}",
                    expression=pullback_shape,
                    signal_type="trade_rule",
                    description="Binary pullback shape gate based on touching and reclaiming resistance.",
                    source_rule="only pullback candidates",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_rule_support_touches_{window}",
                    expression=support_touches,
                    signal_type="trade_rule",
                    description="Support-touch count used by the pullback trade rule.",
                    source_rule="support_touches >= min",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_rule_support_touches_pass_{window}",
                    expression=f"({support_touches}) >= {float(min_support_touches):.1f}",
                    signal_type="trade_rule",
                    description="Binary pass flag for the support-touch floor.",
                    source_rule="support_touches >= min",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_rule_resistance_touches_{window}",
                    expression=resistance_touches,
                    signal_type="trade_rule",
                    description="Resistance-touch count used by the pullback trade rule.",
                    source_rule="resistance_touches <= max",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_rule_resistance_touches_pass_{window}",
                    expression=f"({resistance_touches}) <= {float(max_resistance_touches):.1f}",
                    signal_type="trade_rule",
                    description="Binary pass flag for the resistance-touch cap.",
                    source_rule="resistance_touches <= max",
                ),
                StrategyBoxFactorSpec(
                    name=f"box_rule_range_neutral_reclaim2_exception_{window}",
                    expression=(
                        f"regime_range_neutral * ({pullback_shape})"
                        f" * (({close_above_resistance}) >= {range_neutral_min_close_above_resistance_pct:.8f})"
                    ),
                    signal_type="trade_rule",
                    description="Range-neutral exception flag for unusually strong pullback reclaim.",
                    source_rule="range_neutral pullback_close_above_resistance_pct exception",
                    required_columns=("regime_range_neutral",),
                ),
                StrategyBoxFactorSpec(
                    name=f"box_rule_tradable_pullback_mask_{window}",
                    expression=rule_mask,
                    signal_type="trade_rule",
                    description="Composite mask that reproduces the current tradable pullback rule set.",
                    source_rule="bull_active + rank/heat/touch/turnover/ma10 filters",
                    required_columns=required_columns,
                ),
            ]
        )

    return _dedupe_specs(specs)


def strategy_box_expressions(
    windows: tuple[int, ...] = DEFAULT_BOX_WINDOWS,
    volume_windows: tuple[int, ...] = DEFAULT_VOLUME_WINDOWS,
    retest_windows: tuple[int, ...] = DEFAULT_BREAKOUT_RETEST_WINDOWS,
    signal_type: str | None = None,
    include_trade_rule_factors: bool = False,
) -> list[str]:
    """Return only expression strings for statistical alpha loops."""
    specs = strategy_box_factor_specs(
        windows=windows,
        volume_windows=volume_windows,
        retest_windows=retest_windows,
        include_trade_rule_factors=include_trade_rule_factors,
    )
    specs = _filter_specs(specs, signal_type)
    return [spec.expression for spec in specs]


def strategy_box_factor_names(
    windows: tuple[int, ...] = DEFAULT_BOX_WINDOWS,
    volume_windows: tuple[int, ...] = DEFAULT_VOLUME_WINDOWS,
    retest_windows: tuple[int, ...] = DEFAULT_BREAKOUT_RETEST_WINDOWS,
    signal_type: str | None = None,
    include_trade_rule_factors: bool = False,
) -> list[str]:
    """Return factor names in the same order as ``strategy_box_expressions``."""
    specs = strategy_box_factor_specs(
        windows=windows,
        volume_windows=volume_windows,
        retest_windows=retest_windows,
        include_trade_rule_factors=include_trade_rule_factors,
    )
    specs = _filter_specs(specs, signal_type)
    return [spec.name for spec in specs]


__all__ = [
    "DEFAULT_BOX_TOLERANCE_PCT",
    "DEFAULT_BREAKOUT_RETEST_WINDOWS",
    "DEFAULT_BOX_WINDOWS",
    "DEFAULT_PULLBACK_RECLAIM_PCT",
    "DEFAULT_RANGE_RECLAIM2_MIN_BOX_HEIGHT_PCT",
    "DEFAULT_RANGE_RECLAIM2_MIN_CLOSE_ABOVE_RESISTANCE_PCT",
    "DEFAULT_TRADE_RULE_MAX_HEAT_SCORE",
    "DEFAULT_TRADE_RULE_MAX_RESISTANCE_TOUCHES",
    "DEFAULT_TRADE_RULE_MAX_TURNOVER_RATE",
    "DEFAULT_TRADE_RULE_MIN_MA10_BIAS",
    "DEFAULT_TRADE_RULE_MIN_RANK_SCORE",
    "DEFAULT_TRADE_RULE_MIN_SUPPORT_TOUCHES",
    "DEFAULT_TRADE_RULE_MIN_TURNOVER_RATE",
    "DEFAULT_VOLUME_WINDOWS",
    "StrategyBoxFactorSpec",
    "strategy_box_expressions",
    "strategy_box_factor_names",
    "strategy_box_factor_specs",
    "strategy_box_trade_rule_factor_specs",
]
