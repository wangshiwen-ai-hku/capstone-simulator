"""Scheduling intent shared by problem builders and replaceable optimizers."""

from __future__ import annotations

from dataclasses import dataclass
import enum
import math
from types import MappingProxyType
from typing import Mapping


class OptimizationDirection(str, enum.Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveAggregation(str, enum.Enum):
    WEIGHTED_SUM = "weighted_sum"
    LEXICOGRAPHIC = "lexicographic"


class ObjectiveMetric(str, enum.Enum):
    """Typed quantities understood by the shared evaluator."""

    MAKESPAN_MS = "makespan_ms"
    TOTAL_DEADLINE_VIOLATION_MS = "total_deadline_violation_ms"
    TOTAL_COMPLETION_TIME_MS = "total_completion_time_ms"
    CRITICAL_PATH_FINISH_MS = "critical_path_finish_ms"
    TOTAL_ENERGY_J = "total_energy_j"
    TOTAL_COMMUNICATION_MS = "total_communication_ms"
    LOCALITY_PENALTY = "locality_penalty"
    DROPPED_TASKS = "dropped_tasks"
    NON_SOURCE_ASSIGNMENTS = "non_source_assignments"
    NON_EDGE_ASSIGNMENTS = "non_edge_assignments"
    PLACEMENT_PREFERENCE_PENALTY = "placement_preference_penalty"
    RULE_MISMATCH_COUNT = "rule_mismatch_count"
    EXPECTED_WEIGHTED_SUCCESS_RATIO = (
        "expected_weighted_success_ratio"
    )
    NORMALIZED_COMMUNICATION_RATIO = (
        "normalized_communication_ratio"
    )
    MAXIMUM_RESOURCE_UTILIZATION = (
        "maximum_resource_utilization"
    )
    DEFERRED_PRIORITY_PENALTY = "deferred_priority_penalty"


class ConstraintRelation(str, enum.Enum):
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"


@dataclass(frozen=True)
class ObjectiveSpec:
    """One normalized term in a policy's objective definition."""

    objective_id: str
    metric: ObjectiveMetric
    direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    weight: float = 1.0
    normalization_scale: float = 1.0
    priority_order: int = 0

    def __post_init__(self) -> None:
        if not self.objective_id.strip():
            raise ValueError("objective_id must be non-blank")
        if not isinstance(self.metric, ObjectiveMetric):
            raise TypeError("objective metric must be an ObjectiveMetric")
        if not isinstance(self.direction, OptimizationDirection):
            raise TypeError(
                "objective direction must be an OptimizationDirection"
            )
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("objective weight must be positive")
        if (
            not math.isfinite(self.normalization_scale)
            or self.normalization_scale <= 0
        ):
            raise ValueError("objective normalization_scale must be positive")
        if self.priority_order < 0:
            raise ValueError("objective priority_order must be non-negative")


@dataclass(frozen=True)
class ConstraintSpec:
    """A typed hard bound or soft penalty over an evaluated metric."""

    constraint_id: str
    metric: ObjectiveMetric
    relation: ConstraintRelation
    bound: float
    hard: bool = True
    violation_penalty: float = 0.0
    priority_order: int = 0

    def __post_init__(self) -> None:
        if not self.constraint_id.strip():
            raise ValueError("constraint_id must be non-blank")
        if not isinstance(self.metric, ObjectiveMetric):
            raise TypeError("constraint metric must be an ObjectiveMetric")
        if not isinstance(self.relation, ConstraintRelation):
            raise TypeError(
                "constraint relation must be a ConstraintRelation"
            )
        if not math.isfinite(self.bound):
            raise ValueError("constraint bound must be finite")
        if (
            not math.isfinite(self.violation_penalty)
            or self.violation_penalty < 0
        ):
            raise ValueError(
                "constraint violation_penalty must be non-negative"
            )
        if self.hard and self.violation_penalty != 0:
            raise ValueError(
                "hard constraints cannot declare a violation penalty"
            )
        if self.priority_order < 0:
            raise ValueError(
                "constraint priority_order must be non-negative"
            )


@dataclass(frozen=True)
class SchedulingPolicy:
    """Versioned scheduling intent, independent of system state and solver."""

    policy_id: str
    version: str
    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[ConstraintSpec, ...] = ()
    objective_aggregation: ObjectiveAggregation = (
        ObjectiveAggregation.LEXICOGRAPHIC
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not self.policy_id.strip():
            raise ValueError("policy_id must be non-blank")
        if not self.version.strip():
            raise ValueError("policy version must be non-blank")
        if not self.objectives:
            raise ValueError("a scheduling policy requires an objective")
        if not isinstance(self.objective_aggregation, ObjectiveAggregation):
            raise TypeError(
                "objective_aggregation must be an ObjectiveAggregation"
            )
        objective_ids = tuple(
            item.objective_id for item in self.objectives
        )
        if len(objective_ids) != len(set(objective_ids)):
            raise ValueError("policy objective ids must be unique")
        constraint_ids = tuple(
            item.constraint_id for item in self.constraints
        )
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("policy constraint ids must be unique")


@dataclass(frozen=True)
class SolveLimits:
    """Generic solve controls; optimizer-specific tuning stays in the solver."""

    solve_budget_ms: float = 50.0
    max_iterations: int = 0
    deterministic: bool = True
    random_seed: int = 0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.solve_budget_ms)
            or self.solve_budget_ms <= 0
        ):
            raise ValueError("solve_budget_ms must be positive")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")


@dataclass(frozen=True)
class ObjectiveEvaluation:
    objective_id: str
    metric: ObjectiveMetric
    priority_order: int
    raw_value: float
    normalized_value: float
    weighted_value: float


@dataclass(frozen=True)
class ConstraintEvaluation:
    constraint_id: str
    metric: ObjectiveMetric
    priority_order: int
    raw_value: float
    bound: float
    violation: float
    satisfied: bool
    hard: bool
    penalty: float


_AVOID_DROPS = ConstraintSpec(
    constraint_id="avoid_dropped_tasks",
    metric=ObjectiveMetric.DROPPED_TASKS,
    relation=ConstraintRelation.LESS_THAN_OR_EQUAL,
    bound=0.0,
    hard=False,
    violation_penalty=1_000_000.0,
)


def _objective(
    metric: ObjectiveMetric,
    priority: int,
    *,
    weight: float = 1.0,
) -> ObjectiveSpec:
    return ObjectiveSpec(
        objective_id=metric.value,
        metric=metric,
        priority_order=priority,
        weight=weight,
    )


def binary_offload_policy(
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 2.0,
) -> SchedulingPolicy:
    """Build the dimensionless policy solved by binary offloading.

    Binary refers to the one-hot placement decision for each task, rather than
    restricting the eligible target kinds to a local/edge pair.
    """

    weights = (alpha, beta, gamma)
    if not all(
        math.isfinite(value) and value >= 0.0 for value in weights
    ):
        raise ValueError(
            "binary-offload weights must be finite and non-negative"
        )
    if not any(value > 0.0 for value in weights):
        raise ValueError(
            "binary-offload requires at least one positive weight"
        )
    objectives = []
    if alpha > 0.0:
        objectives.append(
            ObjectiveSpec(
                objective_id=(
                    ObjectiveMetric.EXPECTED_WEIGHTED_SUCCESS_RATIO.value
                ),
                metric=ObjectiveMetric.EXPECTED_WEIGHTED_SUCCESS_RATIO,
                direction=OptimizationDirection.MAXIMIZE,
                weight=alpha,
            )
        )
    if beta > 0.0:
        objectives.append(
            ObjectiveSpec(
                objective_id=(
                    ObjectiveMetric.NORMALIZED_COMMUNICATION_RATIO.value
                ),
                metric=ObjectiveMetric.NORMALIZED_COMMUNICATION_RATIO,
                weight=beta,
            )
        )
    if gamma > 0.0:
        objectives.append(
            ObjectiveSpec(
                objective_id=(
                    ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION.value
                ),
                metric=ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION,
                weight=gamma,
            )
        )
    return SchedulingPolicy(
        policy_id="binary_offload",
        version="2",
        objectives=tuple(objectives),
        constraints=(_AVOID_DROPS,),
        objective_aggregation=ObjectiveAggregation.WEIGHTED_SUM,
    )


def deferred_offload_policy(
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 2.0,
    delta: float = 1.0,
) -> SchedulingPolicy:
    """Binary-offload trade-offs plus priority-weighted deferral cost."""

    base = binary_offload_policy(alpha=alpha, beta=beta, gamma=gamma)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("deferred-offload weight must be positive and finite")
    return SchedulingPolicy(
        policy_id="deferred_offload",
        version="1",
        objectives=(
            *base.objectives,
            ObjectiveSpec(
                objective_id=ObjectiveMetric.DEFERRED_PRIORITY_PENALTY.value,
                metric=ObjectiveMetric.DEFERRED_PRIORITY_PENALTY,
                direction=OptimizationDirection.MINIMIZE,
                weight=delta,
            ),
        ),
        # The formulation itself forbids DROP. Deferred tasks are intentional
        # rolling-horizon decisions and must not violate the legacy drop guard.
        constraints=(),
        objective_aggregation=ObjectiveAggregation.WEIGHTED_SUM,
    )


_BUILT_IN_POLICIES: Mapping[str, SchedulingPolicy] = MappingProxyType(
    {
        "binary_offload": binary_offload_policy(),
        "dag_deadline": SchedulingPolicy(
            policy_id="dag_deadline",
            version="1",
            objectives=(
                _objective(
                    ObjectiveMetric.TOTAL_DEADLINE_VIOLATION_MS,
                    0,
                ),
                _objective(
                    ObjectiveMetric.PLACEMENT_PREFERENCE_PENALTY,
                    1,
                ),
                _objective(
                    ObjectiveMetric.CRITICAL_PATH_FINISH_MS,
                    2,
                ),
                _objective(ObjectiveMetric.LOCALITY_PENALTY, 2),
                _objective(ObjectiveMetric.TOTAL_ENERGY_J, 3),
            ),
            constraints=(_AVOID_DROPS,),
        ),
        "rule_based": SchedulingPolicy(
            policy_id="rule_based",
            version="1",
            objectives=(
                _objective(ObjectiveMetric.RULE_MISMATCH_COUNT, 0),
                _objective(
                    ObjectiveMetric.TOTAL_COMPLETION_TIME_MS,
                    1,
                ),
                _objective(ObjectiveMetric.TOTAL_ENERGY_J, 2),
            ),
            constraints=(_AVOID_DROPS,),
        ),
        "local_first": SchedulingPolicy(
            policy_id="local_first",
            version="1",
            objectives=(
                _objective(ObjectiveMetric.NON_SOURCE_ASSIGNMENTS, 0),
                _objective(
                    ObjectiveMetric.TOTAL_COMPLETION_TIME_MS,
                    1,
                ),
                _objective(ObjectiveMetric.TOTAL_ENERGY_J, 2),
            ),
            constraints=(_AVOID_DROPS,),
        ),
        "edge_first": SchedulingPolicy(
            policy_id="edge_first",
            version="1",
            objectives=(
                _objective(ObjectiveMetric.NON_EDGE_ASSIGNMENTS, 0),
                _objective(
                    ObjectiveMetric.TOTAL_COMPLETION_TIME_MS,
                    1,
                ),
                _objective(ObjectiveMetric.TOTAL_ENERGY_J, 2),
            ),
            constraints=(_AVOID_DROPS,),
        ),
        "greedy_cost": SchedulingPolicy(
            policy_id="greedy_cost",
            version="1",
            objectives=(
                _objective(
                    ObjectiveMetric.PLACEMENT_PREFERENCE_PENALTY,
                    0,
                ),
                _objective(
                    ObjectiveMetric.TOTAL_COMPLETION_TIME_MS,
                    1,
                ),
                _objective(ObjectiveMetric.TOTAL_ENERGY_J, 2),
            ),
            constraints=(_AVOID_DROPS,),
        ),
    }
)

_POLICY_ORDER = (
    "binary_offload",
    "dag_deadline",
    "rule_based",
    "local_first",
    "edge_first",
    "greedy_cost",
)

_ALGORITHM_ALIAS_ORDER = tuple(
    policy_id
    for policy_id in _POLICY_ORDER
    if policy_id != "binary_offload"
)


def built_in_policy_ids() -> tuple[str, ...]:
    return _POLICY_ORDER


def built_in_policy(policy_id: str) -> SchedulingPolicy:
    try:
        return _BUILT_IN_POLICIES[policy_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown policy {policy_id!r}; available={list(_POLICY_ORDER)}"
        ) from exc


def algorithm_aliases() -> dict[str, dict[str, str]]:
    """Return stable combined-selection aliases used by the existing API."""

    return {
        policy_id: {
            "optimizer_id": "heuristic",
            "policy_id": policy_id,
        }
        for policy_id in _ALGORITHM_ALIAS_ORDER
    }
