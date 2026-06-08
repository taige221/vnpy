from __future__ import annotations

from dataclasses import dataclass


COMBO_CANDIDATE_SET: str = "a_share_research_2023_20260430"


@dataclass(frozen=True)
class ComboFactorSpec:
    """Named composite factor expression used by AlphaLab research."""

    name: str
    expression: str
    category: str
    description: str
    components: tuple[str, ...]
    required_columns: tuple[str, ...] = ()


def _rank(expression: str) -> str:
    return f"cs_rank({expression})"


def _mean(expressions: tuple[str, ...]) -> str:
    joined = " + ".join(f"({expression})" for expression in expressions)
    return f"({joined}) / {float(len(expressions)):.1f}"


def _weighted(left: str, left_weight: float, right: str, right_weight: float) -> str:
    return f"({left_weight:.8f} * ({left}) + {right_weight:.8f} * ({right}))"


_COMMON_REVERSAL_20 = "-1 * (close / (ts_delay(close, 20) + 1e-12) - 1)"
_ANTI_TURNOVER_SURGE_120 = "-1 * (turnover / (ts_mean(turnover, 120) + 1e-12) - 1)"
_NEG_PRICE_VOLUME_CORR_20 = "-1 * ts_corr(close, volume, 20)"
_NEG_PRICE_VOLUME_CORR_60 = "-1 * ts_corr(close, volume, 60)"

_PRIOR_RESISTANCE_20 = "ts_max(ts_delay(high, 1), 20)"
_PRIOR_SUPPORT_20 = "ts_min(ts_delay(low, 1), 20)"
_BOX_STACK_LIFT_REVERSAL_20 = (
    "-1 * ("
    f"(({_PRIOR_RESISTANCE_20}) / (ts_delay(({_PRIOR_RESISTANCE_20}), 20) + 1e-12) - 1)"
    " + "
    f"(({_PRIOR_SUPPORT_20}) / (ts_delay(({_PRIOR_SUPPORT_20}), 20) + 1e-12) - 1)"
    ") / 2"
)

_TRIX_14 = (
    "(ta_ema(ta_ema(ta_ema(close, 14), 14), 14))"
    " / (ts_delay((ta_ema(ta_ema(ta_ema(close, 14), 14), 14)), 1) + 1e-12) - 1"
)
_ANTI_LOG_TURNOVER_MEAN_5 = "-1 * log(ts_mean(turnover, 5) + 1)"
_ANTI_TRIX_14 = f"-1 * ({_TRIX_14})"
_CLOSE_RETURN_1 = "close / (ts_delay(close, 1) + 1e-12) - 1"
_ANTI_PVT_PRESSURE_60 = (
    f"-1 * ts_sum(({_CLOSE_RETURN_1}) * volume, 60) / (ts_sum(volume, 60) + 1e-12)"
)
_RELATIVE_BENCHMARK_STRENGTH_20 = (
    "(close / (ts_delay(close, 20) + 1e-12) - 1)"
    " - "
    "(benchmark_close / (ts_delay(benchmark_close, 20) + 1e-12) - 1)"
)
_ANTI_RELATIVE_BENCHMARK_STRENGTH_20 = f"-1 * ({_RELATIVE_BENCHMARK_STRENGTH_20})"
_LIMIT_UP_STATE = "raw_close >= up_limit * 0.999"
_ANTI_LIMIT_UP_HEAT_20 = f"-1 * ts_sum(({_LIMIT_UP_STATE}), 20)"

_CORE4_COMPONENTS: tuple[str, ...] = (
    _rank(_COMMON_REVERSAL_20),
    _rank(_ANTI_TURNOVER_SURGE_120),
    _rank(_NEG_PRICE_VOLUME_CORR_20),
    _rank(_BOX_STACK_LIFT_REVERSAL_20),
)
_COMBO_CORE4_CONTROL = _mean(_CORE4_COMPONENTS)
_COMBO_CORE4_PVCORR60_15PCT = _weighted(
    _COMBO_CORE4_CONTROL,
    0.85,
    _rank(_NEG_PRICE_VOLUME_CORR_60),
    0.15,
)

_BROAD_TOP5_COMPONENTS: tuple[str, ...] = (
    _rank(_ANTI_LOG_TURNOVER_MEAN_5),
    _rank(_ANTI_TRIX_14),
    _rank(_ANTI_PVT_PRESSURE_60),
    _rank(_ANTI_RELATIVE_BENCHMARK_STRENGTH_20),
    _rank(_ANTI_LIMIT_UP_HEAT_20),
)
_COMBO_BROAD_TOP5 = _mean(_BROAD_TOP5_COMPONENTS)
_COMBO_PVCORR15_BROAD_ANTITRIX14_10PCT = _weighted(
    _COMBO_CORE4_PVCORR60_15PCT,
    0.90,
    _rank(_ANTI_TRIX_14),
    0.10,
)
_COMBO_PVCORR15_BROAD_TOP5_10PCT = _weighted(
    _COMBO_CORE4_PVCORR60_15PCT,
    0.90,
    _COMBO_BROAD_TOP5,
    0.10,
)

_CORE_REQUIRED_COLUMNS: tuple[str, ...] = ("close", "high", "low", "turnover", "volume")
_BROAD_REQUIRED_COLUMNS: tuple[str, ...] = (
    "close",
    "turnover",
    "volume",
    "benchmark_close",
    "raw_close",
    "up_limit",
)
_FULL_REQUIRED_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys((*_CORE_REQUIRED_COLUMNS, *_BROAD_REQUIRED_COLUMNS)))


def combo_factor_specs(include_components: bool = True) -> list[ComboFactorSpec]:
    """Return named A-share research composite factor expressions."""
    specs: list[ComboFactorSpec] = []
    if include_components:
        specs.extend(
            [
                ComboFactorSpec(
                    name="combo_core4_control",
                    expression=_COMBO_CORE4_CONTROL,
                    category="combo",
                    description="Equal-weight rank composite of reversal, turnover-surge, price-volume-correlation, and box-lift reversal factors.",
                    components=(
                        "common_reversal_20",
                        "anti_turnover_surge_120",
                        "neg_price_volume_corr_20",
                        "box_stack_lift_reversal_20",
                    ),
                    required_columns=_CORE_REQUIRED_COLUMNS,
                ),
                ComboFactorSpec(
                    name="combo_broad_top5",
                    expression=_COMBO_BROAD_TOP5,
                    category="combo_component",
                    description="Equal-weight rank composite of the top broad anti-crowding and anti-extension factors.",
                    components=(
                        "anti_log_turnover_mean_5",
                        "anti_trix_14",
                        "anti_pvt_pressure_60",
                        "anti_relative_benchmark_strength_20",
                        "anti_limit_up_heat_20",
                    ),
                    required_columns=_BROAD_REQUIRED_COLUMNS,
                ),
            ]
        )

    specs.extend(
        [
            ComboFactorSpec(
                name="combo_core4_pvcorr60_15pct",
                expression=_COMBO_CORE4_PVCORR60_15PCT,
                category="candidate_combo",
                description="Core four-factor rank composite with a 15% overlay on negative 60-bar price-volume correlation.",
                components=("combo_core4_control", "neg_price_volume_corr_60"),
                required_columns=_CORE_REQUIRED_COLUMNS,
            ),
            ComboFactorSpec(
                name="combo_pvcorr15_broad_antitrix14_10pct",
                expression=_COMBO_PVCORR15_BROAD_ANTITRIX14_10PCT,
                category="candidate_combo",
                description="Pvcorr60 candidate with a 10% anti-TRIX14 overlay. This matched the strongest quick combo result before final TA-Lib EMA robustness review.",
                components=("combo_core4_pvcorr60_15pct", "anti_trix_14"),
                required_columns=_CORE_REQUIRED_COLUMNS,
            ),
            ComboFactorSpec(
                name="combo_pvcorr15_broad_top5_10pct",
                expression=_COMBO_PVCORR15_BROAD_TOP5_10PCT,
                category="candidate_combo",
                description="Pvcorr60 candidate with a 10% broad-top5 anti-crowding and anti-extension overlay.",
                components=("combo_core4_pvcorr60_15pct", "combo_broad_top5"),
                required_columns=_FULL_REQUIRED_COLUMNS,
            ),
        ]
    )
    return specs


def candidate_combo_factor_specs() -> list[ComboFactorSpec]:
    """Return only promoted candidate combo factors without intermediate components."""
    return combo_factor_specs(include_components=False)


def combo_factor_names(include_components: bool = True) -> list[str]:
    """Return combo factor names in spec order."""
    return [spec.name for spec in combo_factor_specs(include_components=include_components)]


def combo_expressions(include_components: bool = True) -> list[str]:
    """Return combo factor expressions in spec order."""
    return [spec.expression for spec in combo_factor_specs(include_components=include_components)]


__all__ = [
    "COMBO_CANDIDATE_SET",
    "ComboFactorSpec",
    "candidate_combo_factor_specs",
    "combo_expressions",
    "combo_factor_names",
    "combo_factor_specs",
]
