"""Small, versioned contracts shared by post-run evaluators."""

from __future__ import annotations

from dataclasses import dataclass
import enum
import math
from statistics import mean
from types import MappingProxyType
from typing import Iterable, Mapping


class AggregationRule(str, enum.Enum):
    """How observations of one metric combine across benchmark runs."""

    MEAN = "mean"
    SUM = "sum"
    RATIO_OF_SUMS = "ratio_of_sums"
    MAX = "max"


@dataclass(frozen=True)
class MetricDefinition:
    """Stable meaning and cross-run aggregation rule for one metric."""

    metric_id: str
    unit: str
    aggregation: AggregationRule = AggregationRule.MEAN
    semantics_version: str = "1"

    def __post_init__(self) -> None:
        if not self.metric_id.strip():
            raise ValueError("metric_id must be non-blank")
        if not self.unit.strip():
            raise ValueError("metric unit must be non-blank")
        if not self.semantics_version.strip():
            raise ValueError("metric semantics_version must be non-blank")


@dataclass(frozen=True)
class MetricObservation:
    """One metric value, optionally retaining its additive ratio terms."""

    definition: MetricDefinition
    value: float | int
    numerator: float | int | None = None
    denominator: float | int | None = None

    def __post_init__(self) -> None:
        values = (self.value, self.numerator, self.denominator)
        if any(
            item is not None
            and (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or math.isnan(float(item))
            )
            for item in values
        ):
            raise ValueError("metric values must be numeric and not NaN")
        if self.definition.aggregation is AggregationRule.RATIO_OF_SUMS:
            if self.numerator is None or self.denominator is None:
                raise ValueError(
                    "ratio_of_sums observations require numerator and denominator"
                )
            if float(self.denominator) < 0:
                raise ValueError("metric denominator must be non-negative")


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable output of evaluating one complete run artifact."""

    observations: tuple[MetricObservation, ...]
    schema_version: str = "mars.workflow-metrics.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        metric_ids = tuple(
            observation.definition.metric_id
            for observation in self.observations
        )
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("evaluation contains duplicate metric ids")
        if not self.schema_version.strip():
            raise ValueError("evaluation schema_version must be non-blank")

    @property
    def metrics(self) -> Mapping[str, float | int]:
        return MappingProxyType(
            {
                observation.definition.metric_id: observation.value
                for observation in self.observations
            }
        )

    def as_dict(self) -> dict[str, float | int]:
        """Return the compatibility metric mapping consumed by existing APIs."""

        return dict(self.metrics)


def aggregate_evaluations(
    results: Iterable[EvaluationResult],
) -> dict[str, float]:
    """Aggregate compatible observations without averaging pre-computed ratios."""

    grouped: dict[str, list[MetricObservation]] = {}
    for result in results:
        for observation in result.observations:
            grouped.setdefault(
                observation.definition.metric_id,
                [],
            ).append(observation)

    aggregated: dict[str, float] = {}
    for metric_id, observations in grouped.items():
        definition = observations[0].definition
        if any(item.definition != definition for item in observations[1:]):
            raise ValueError(
                f"incompatible metric definitions for {metric_id!r}"
            )
        values = [float(item.value) for item in observations]
        if definition.aggregation is AggregationRule.SUM:
            value = sum(values)
        elif definition.aggregation is AggregationRule.MAX:
            value = max(values)
        elif definition.aggregation is AggregationRule.RATIO_OF_SUMS:
            numerator = sum(float(item.numerator or 0.0) for item in observations)
            denominator = sum(
                float(item.denominator or 0.0) for item in observations
            )
            value = numerator / denominator if denominator else 0.0
        else:
            value = mean(values)
        aggregated[metric_id] = value
    return aggregated


__all__ = [
    "AggregationRule",
    "EvaluationResult",
    "MetricDefinition",
    "MetricObservation",
    "aggregate_evaluations",
]
