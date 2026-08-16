"""Shared, solver-independent objective and constraint evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import enum
import hashlib
import json
from types import MappingProxyType
from typing import Callable, Mapping

from ..domain.execution import Assignment, ExecutionMode
from ..domain.task import TaskInstance, resolved_placement_constraints
from ..domain.topology import NodeKind
from .base import (
    CandidateEstimate,
    PlannedResourceReservation,
    SchedulingPlan,
    SchedulingProblem,
    maximum_resource_utilization,
)
from .policy import (
    ConstraintEvaluation,
    ConstraintRelation,
    ObjectiveAggregation,
    ObjectiveEvaluation,
    ObjectiveMetric,
    OptimizationDirection,
    SchedulingPolicy,
)


class MetricScope(str, enum.Enum):
    """The plan context required to evaluate a raw metric."""

    ASSIGNMENT_ADDITIVE = "assignment_additive"
    PLAN_GLOBAL = "plan_global"
    TIMELINE = "timeline"


class CandidateFidelity(str, enum.Enum):
    """Fidelity of a local raw contribution to the plan-level metric.

    This describes metric semantics only; it does not claim that a greedy
    optimizer using an EXACT contribution is globally optimal.
    """

    EXACT = "EXACT"
    PROXY = "PROXY"
    UNSUPPORTED = "UNSUPPORTED"


PlanMetricEvaluator = Callable[[SchedulingProblem, SchedulingPlan], float]
CandidateMetricProxy = Callable[
    [SchedulingProblem, str, CandidateEstimate],
    float,
]


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """One versioned raw metric understood by every optimizer."""

    metric: ObjectiveMetric
    semantics_version: str
    unit: str
    scope: MetricScope
    evaluate_plan: PlanMetricEvaluator
    candidate_proxy: CandidateMetricProxy | None
    candidate_fidelity: CandidateFidelity
    proto_enum_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.metric, ObjectiveMetric):
            raise TypeError("metric definition requires an ObjectiveMetric")
        if not self.semantics_version.strip():
            raise ValueError("metric semantics_version must be non-blank")
        if not self.unit.strip():
            raise ValueError("metric unit must be non-blank")
        if not isinstance(self.scope, MetricScope):
            raise TypeError("metric definition requires a MetricScope")
        if not callable(self.evaluate_plan):
            raise TypeError("metric evaluate_plan must be callable")
        if self.candidate_proxy is not None and not callable(
            self.candidate_proxy
        ):
            raise TypeError("metric candidate_proxy must be callable")
        if not isinstance(self.candidate_fidelity, CandidateFidelity):
            raise TypeError(
                "metric definition requires a CandidateFidelity"
            )
        if not self.proto_enum_name.strip():
            raise ValueError("metric proto_enum_name must be non-blank")
        if not self.proto_enum_name.startswith("OBJECTIVE_METRIC_"):
            raise ValueError("metric proto_enum_name must use OBJECTIVE_METRIC_ prefix")
        if (self.candidate_fidelity is CandidateFidelity.UNSUPPORTED) != (
            self.candidate_proxy is None
        ):
            raise ValueError(
                "UNSUPPORTED candidate fidelity requires no candidate proxy"
            )


def metric_contract_id(policy: SchedulingPolicy) -> str:
    """Fingerprint the metric semantics referenced by one policy.

    A policy carries stable metric IDs, while the registry owns their executable
    semantics.  Binding the referenced semantic versions into the Problem ID
    prevents an evaluator change from silently reusing an old problem identity.
    Metrics that the policy does not reference intentionally do not affect the
    fingerprint.
    """

    referenced = sorted(
        {
            item.metric
            for item in (*policy.objectives, *policy.constraints)
        },
        key=lambda metric: metric.value,
    )
    payload = {
        "schema_version": "mars.metric-contract.v1",
        "metrics": [
            {
                "metric": metric.value,
                "semantics_version": _metric_definition(
                    metric
                ).semantics_version,
            }
            for metric in referenced
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"metric-contract:{hashlib.sha256(encoded).hexdigest()[:20]}"


def candidate_proxy_key(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> tuple[float, ...]:
    """Rank one candidate through each metric's declared local proxy.

    This is a heuristic selection aid, not a substitute for authoritative
    plan-level objective and constraint evaluation.
    """

    values = tuple(
        (
            objective.priority_order,
            _directed_weighted_value(
                _candidate_metric_value(
                    problem,
                    task_id,
                    candidate,
                    objective.metric,
                ),
                objective.direction,
                objective.weight,
                objective.normalization_scale,
            ),
        )
        for objective in problem.policy.objectives
    )
    return _aggregate_objective_key(
        values,
        problem.policy.objective_aggregation,
    )


def candidate_objective_key(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> tuple[float, ...]:
    """Compatibility alias for :func:`candidate_proxy_key`."""

    return candidate_proxy_key(problem, task_id, candidate)


def evaluate_objectives(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> tuple[ObjectiveEvaluation, ...]:
    """Recompute every declared objective from the returned plan."""

    evaluations: list[ObjectiveEvaluation] = []
    for objective in problem.policy.objectives:
        raw_value = _plan_metric_value(
            problem,
            plan,
            objective.metric,
        )
        normalized = raw_value / objective.normalization_scale
        weighted = _directed_weighted_value(
            raw_value,
            objective.direction,
            objective.weight,
            objective.normalization_scale,
        )
        evaluations.append(
            ObjectiveEvaluation(
                objective_id=objective.objective_id,
                metric=objective.metric,
                priority_order=objective.priority_order,
                raw_value=raw_value,
                normalized_value=normalized,
                weighted_value=weighted,
            )
        )
    return tuple(evaluations)


def objective_key(
    policy: SchedulingPolicy,
    evaluations: tuple[ObjectiveEvaluation, ...],
    constraint_evaluations: tuple[ConstraintEvaluation, ...] = (),
) -> tuple[float, ...]:
    """Return the complete comparable score declared by a policy."""

    return _aggregate_objective_key(
        (
            *((item.priority_order, item.weighted_value) for item in evaluations),
            *(
                (item.priority_order, item.penalty)
                for item in constraint_evaluations
                if not item.hard
            ),
        ),
        policy.objective_aggregation,
    )


def evaluate_constraints(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> tuple[ConstraintEvaluation, ...]:
    """Evaluate policy bounds independently of domain safety validation."""

    evaluations: list[ConstraintEvaluation] = []
    for constraint in problem.policy.constraints:
        raw_value = _plan_metric_value(
            problem,
            plan,
            constraint.metric,
        )
        if constraint.relation is ConstraintRelation.LESS_THAN_OR_EQUAL:
            violation = max(0.0, raw_value - constraint.bound)
        else:
            violation = max(0.0, constraint.bound - raw_value)
        evaluations.append(
            ConstraintEvaluation(
                constraint_id=constraint.constraint_id,
                metric=constraint.metric,
                priority_order=constraint.priority_order,
                raw_value=raw_value,
                bound=constraint.bound,
                violation=violation,
                satisfied=violation <= 1e-9,
                hard=constraint.hard,
                penalty=(
                    0.0 if constraint.hard else violation * constraint.violation_penalty
                ),
            )
        )
    return tuple(evaluations)


def _aggregate_objective_key(
    values: tuple[tuple[int, float], ...],
    aggregation: ObjectiveAggregation,
) -> tuple[float, ...]:
    if aggregation is ObjectiveAggregation.WEIGHTED_SUM:
        return (sum(value for _, value in values),)
    priorities = sorted({priority for priority, _ in values})
    return tuple(
        sum(value for priority, value in values if priority == current)
        for current in priorities
    )


def _directed_weighted_value(
    raw_value: float,
    direction: OptimizationDirection,
    weight: float,
    normalization_scale: float,
) -> float:
    normalized = raw_value / normalization_scale
    directed = (
        normalized if direction is OptimizationDirection.MINIMIZE else -normalized
    )
    return directed * weight


def _candidate_metric_value(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
    metric: ObjectiveMetric,
) -> float:
    definition = _metric_definition(metric)
    if definition.candidate_proxy is None:
        raise ValueError(f"objective metric {metric.value!r} has no candidate proxy")
    return definition.candidate_proxy(problem, task_id, candidate)


def _plan_metric_value(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
    metric: ObjectiveMetric,
) -> float:
    return _metric_definition(metric).evaluate_plan(problem, plan)


def _task_priority_weights(
    problem: SchedulingProblem,
) -> tuple[dict[str, float], float]:
    weights = {
        task.task_id: max(0.0, float(task.priority))
        for task in problem.epoch.ready_tasks
    }
    total = sum(weights.values())
    return weights, max(1.0, total)


def normalized_communication_ratio(
    communication_ms: float,
    latency_budget_ms: float,
) -> float:
    """Normalize communication time by the declared latency budget."""

    return communication_ms / max(1.0, latency_budget_ms)


def expected_weighted_success_ratio(
    problem: SchedulingProblem,
    assignments: tuple[Assignment, ...],
) -> float:
    """Return priority-weighted nominal success over all ready tasks."""

    weights, total_weight = _task_priority_weights(problem)
    return (
        sum(
            weights[item.task_id] * item.success_probability
            for item in assignments
            if item.execution_mode is not ExecutionMode.DROP
        )
        / total_weight
    )


def plan_normalized_communication_ratio(
    problem: SchedulingProblem,
    assignments: tuple[Assignment, ...],
) -> float:
    """Normalize planned communication by all ready-task latency budgets."""

    return normalized_communication_ratio(
        sum(
            item.communication_ms
            for item in assignments
            if item.execution_mode is not ExecutionMode.DROP
        ),
        sum(
            max(1.0, task.spec.latency_budget_ms) for task in problem.epoch.ready_tasks
        ),
    )


def _selected_candidates(
    problem: SchedulingProblem,
) -> dict[str, dict[str, CandidateEstimate]]:
    return {
        task_id: {candidate.node_id: candidate for candidate in candidates}
        for task_id, candidates in problem.candidates.items()
    }


def _candidate_locality_penalty(candidate: CandidateEstimate) -> float:
    return float(len(set(candidate.input_locations) - {candidate.node_id}) * 2.0)


def _preference_rank(
    task: TaskInstance,
    candidate: CandidateEstimate,
) -> int:
    preferred = resolved_placement_constraints(task).preferred_node_kinds
    if not preferred:
        return 0
    try:
        return preferred.index(candidate.node_kind)
    except ValueError:
        return len(preferred)


def _rule_mismatch(
    problem: SchedulingProblem,
    task: TaskInstance,
    candidate: CandidateEstimate,
) -> bool:
    constraints = resolved_placement_constraints(task)
    source_snapshot = problem.snapshot_by_id[task.source_node_id]
    should_offload = (
        NodeKind.EDGE in constraints.preferred_node_kinds
        or source_snapshot.cpu_util > 0.8
        or source_snapshot.gpu_util > 0.8
        or task.spec.compute_demand > 2.5
    )
    return (
        candidate.node_kind is not NodeKind.EDGE
        if should_offload
        else not candidate.is_source
    )


def _non_drop_assignments(
    plan: SchedulingPlan,
) -> tuple[Assignment, ...]:
    return tuple(
        assignment
        for assignment in plan.assignments
        if assignment.execution_mode is not ExecutionMode.DROP
    )


def _candidate_finish_ms(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    del problem, task_id
    return candidate.finish_ms


def _candidate_deadline_violation_ms(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    projected_finish = candidate.finish_ms + problem.critical_tail_ms.get(task_id, 0.0)
    return max(
        0.0,
        projected_finish - problem.task_by_id[task_id].deadline_time_ms,
    )


def _candidate_critical_path_finish_ms(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    return candidate.finish_ms + problem.critical_tail_ms.get(task_id, 0.0)


def _candidate_energy_j(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    del problem, task_id
    return candidate.energy_j


def _candidate_communication_ms(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    del problem, task_id
    return candidate.communication_ms


def _candidate_locality(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    del problem, task_id
    return _candidate_locality_penalty(candidate)


def _candidate_dropped_tasks(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    del problem, task_id, candidate
    return 0.0


def _candidate_non_source_assignments(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    del problem, task_id
    return float(not candidate.is_source)


def _candidate_non_edge_assignments(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    del problem, task_id
    return float(candidate.node_kind is not NodeKind.EDGE)


def _candidate_placement_preference_penalty(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    return float(_preference_rank(problem.task_by_id[task_id], candidate))


def _candidate_rule_mismatch_count(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    return float(
        _rule_mismatch(
            problem,
            problem.task_by_id[task_id],
            candidate,
        )
    )


def _candidate_expected_weighted_success_ratio(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    weights, total_weight = _task_priority_weights(problem)
    return weights[task_id] * candidate.success_probability / total_weight


def _candidate_normalized_communication_ratio(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    return normalized_communication_ratio(
        candidate.communication_ms,
        problem.task_by_id[task_id].spec.latency_budget_ms,
    )


def _candidate_maximum_resource_utilization(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> float:
    finish_ms = candidate.finish_ms
    compute_start = max(
        candidate.ready_time_ms,
        finish_ms - candidate.compute_ms,
    )
    reservation = PlannedResourceReservation(
        reservation_id=(f"candidate-metric:{task_id}:{candidate.node_id}"),
        epoch_id=problem.epoch.epoch_id,
        task_id=task_id,
        node_id=candidate.node_id,
        start_ms=compute_start,
        finish_ms=finish_ms,
        demand=candidate.resource_demand,
    )
    return maximum_resource_utilization(problem, (reservation,))


def _plan_makespan_ms(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    return max(
        (assignment.estimated_finish_ms for assignment in _non_drop_assignments(plan)),
        default=problem.epoch.now_ms,
    )


def _plan_total_deadline_violation_ms(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    return sum(
        max(
            0.0,
            assignment.estimated_finish_ms
            + problem.critical_tail_ms.get(assignment.task_id, 0.0)
            - problem.task_by_id[assignment.task_id].deadline_time_ms,
        )
        for assignment in _non_drop_assignments(plan)
    )


def _plan_total_completion_time_ms(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    del problem
    return sum(
        assignment.estimated_finish_ms for assignment in _non_drop_assignments(plan)
    )


def _plan_critical_path_finish_ms(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    return max(
        (
            assignment.estimated_finish_ms
            + problem.critical_tail_ms.get(assignment.task_id, 0.0)
            for assignment in _non_drop_assignments(plan)
        ),
        default=problem.epoch.now_ms,
    )


def _plan_total_energy_j(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    del problem
    return sum(assignment.energy_j for assignment in _non_drop_assignments(plan))


def _plan_total_communication_ms(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    del problem
    return sum(
        assignment.communication_ms for assignment in _non_drop_assignments(plan)
    )


def _plan_locality_penalty(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    del problem
    return sum(
        len(set(assignment.input_locations) - {assignment.target_node_id}) * 2.0
        for assignment in _non_drop_assignments(plan)
    )


def _plan_dropped_tasks(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    del problem
    return float(
        sum(
            assignment.execution_mode is ExecutionMode.DROP
            for assignment in plan.assignments
        )
        + len(plan.deferred_task_ids)
    )


def _plan_non_source_assignments(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    return float(
        sum(
            assignment.target_node_id
            != problem.task_by_id[assignment.task_id].source_node_id
            for assignment in _non_drop_assignments(plan)
        )
    )


def _plan_non_edge_assignments(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    return float(
        sum(
            problem.node_by_id[assignment.target_node_id].kind is not NodeKind.EDGE
            for assignment in _non_drop_assignments(plan)
        )
    )


def _plan_placement_preference_penalty(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    selected = _selected_candidates(problem)
    return float(
        sum(
            _preference_rank(
                problem.task_by_id[assignment.task_id],
                selected[assignment.task_id][assignment.target_node_id],
            )
            for assignment in _non_drop_assignments(plan)
        )
    )


def _plan_rule_mismatch_count(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    selected = _selected_candidates(problem)
    return float(
        sum(
            _rule_mismatch(
                problem,
                problem.task_by_id[assignment.task_id],
                selected[assignment.task_id][assignment.target_node_id],
            )
            for assignment in _non_drop_assignments(plan)
        )
    )


def _plan_expected_weighted_success_ratio(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    return expected_weighted_success_ratio(
        problem,
        _non_drop_assignments(plan),
    )


def _plan_normalized_communication_ratio_metric(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    return plan_normalized_communication_ratio(
        problem,
        _non_drop_assignments(plan),
    )


def _plan_maximum_resource_utilization(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    return maximum_resource_utilization(
        problem,
        plan.node_reservations,
    )


def _plan_deferred_priority_penalty(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> float:
    return float(
        sum(
            2 ** problem.task_by_id[task_id].priority
            for task_id in plan.deferred_task_ids
        )
    )


BUILTIN_METRICS: Mapping[ObjectiveMetric, MetricDefinition] = MappingProxyType(
    {
        ObjectiveMetric.MAKESPAN_MS: MetricDefinition(
            metric=ObjectiveMetric.MAKESPAN_MS,
            semantics_version="1",
            unit="ms",
            scope=MetricScope.PLAN_GLOBAL,
            evaluate_plan=_plan_makespan_ms,
            candidate_proxy=_candidate_finish_ms,
            candidate_fidelity=CandidateFidelity.PROXY,
            proto_enum_name="OBJECTIVE_METRIC_MAKESPAN_MS",
        ),
        ObjectiveMetric.TOTAL_DEADLINE_VIOLATION_MS: MetricDefinition(
            metric=ObjectiveMetric.TOTAL_DEADLINE_VIOLATION_MS,
            semantics_version="1",
            unit="ms",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_total_deadline_violation_ms,
            candidate_proxy=_candidate_deadline_violation_ms,
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name=("OBJECTIVE_METRIC_TOTAL_DEADLINE_VIOLATION_MS"),
        ),
        ObjectiveMetric.TOTAL_COMPLETION_TIME_MS: MetricDefinition(
            metric=ObjectiveMetric.TOTAL_COMPLETION_TIME_MS,
            semantics_version="1",
            unit="ms",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_total_completion_time_ms,
            candidate_proxy=_candidate_finish_ms,
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name=("OBJECTIVE_METRIC_TOTAL_COMPLETION_TIME_MS"),
        ),
        ObjectiveMetric.CRITICAL_PATH_FINISH_MS: MetricDefinition(
            metric=ObjectiveMetric.CRITICAL_PATH_FINISH_MS,
            semantics_version="1",
            unit="ms",
            scope=MetricScope.PLAN_GLOBAL,
            evaluate_plan=_plan_critical_path_finish_ms,
            candidate_proxy=_candidate_critical_path_finish_ms,
            candidate_fidelity=CandidateFidelity.PROXY,
            proto_enum_name=("OBJECTIVE_METRIC_CRITICAL_PATH_FINISH_MS"),
        ),
        ObjectiveMetric.TOTAL_ENERGY_J: MetricDefinition(
            metric=ObjectiveMetric.TOTAL_ENERGY_J,
            semantics_version="1",
            unit="J",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_total_energy_j,
            candidate_proxy=_candidate_energy_j,
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name="OBJECTIVE_METRIC_TOTAL_ENERGY_J",
        ),
        ObjectiveMetric.TOTAL_COMMUNICATION_MS: MetricDefinition(
            metric=ObjectiveMetric.TOTAL_COMMUNICATION_MS,
            semantics_version="1",
            unit="ms",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_total_communication_ms,
            candidate_proxy=_candidate_communication_ms,
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name=("OBJECTIVE_METRIC_TOTAL_COMMUNICATION_TIME_MS"),
        ),
        ObjectiveMetric.LOCALITY_PENALTY: MetricDefinition(
            metric=ObjectiveMetric.LOCALITY_PENALTY,
            semantics_version="1",
            unit="penalty",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_locality_penalty,
            candidate_proxy=_candidate_locality,
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name="OBJECTIVE_METRIC_LOCALITY_PENALTY",
        ),
        ObjectiveMetric.DROPPED_TASKS: MetricDefinition(
            metric=ObjectiveMetric.DROPPED_TASKS,
            semantics_version="1",
            unit="task_count",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_dropped_tasks,
            candidate_proxy=_candidate_dropped_tasks,
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name="OBJECTIVE_METRIC_DROPPED_TASK_COUNT",
        ),
        ObjectiveMetric.NON_SOURCE_ASSIGNMENTS: MetricDefinition(
            metric=ObjectiveMetric.NON_SOURCE_ASSIGNMENTS,
            semantics_version="1",
            unit="assignment_count",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_non_source_assignments,
            candidate_proxy=_candidate_non_source_assignments,
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name=("OBJECTIVE_METRIC_NON_SOURCE_ASSIGNMENT_COUNT"),
        ),
        ObjectiveMetric.NON_EDGE_ASSIGNMENTS: MetricDefinition(
            metric=ObjectiveMetric.NON_EDGE_ASSIGNMENTS,
            semantics_version="1",
            unit="assignment_count",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_non_edge_assignments,
            candidate_proxy=_candidate_non_edge_assignments,
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name=("OBJECTIVE_METRIC_NON_EDGE_ASSIGNMENT_COUNT"),
        ),
        ObjectiveMetric.PLACEMENT_PREFERENCE_PENALTY: MetricDefinition(
            metric=ObjectiveMetric.PLACEMENT_PREFERENCE_PENALTY,
            semantics_version="1",
            unit="penalty",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_placement_preference_penalty,
            candidate_proxy=(_candidate_placement_preference_penalty),
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name=("OBJECTIVE_METRIC_PLACEMENT_PREFERENCE_PENALTY"),
        ),
        ObjectiveMetric.RULE_MISMATCH_COUNT: MetricDefinition(
            metric=ObjectiveMetric.RULE_MISMATCH_COUNT,
            semantics_version="1",
            unit="count",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_rule_mismatch_count,
            candidate_proxy=_candidate_rule_mismatch_count,
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name="OBJECTIVE_METRIC_RULE_MISMATCH_COUNT",
        ),
        ObjectiveMetric.EXPECTED_WEIGHTED_SUCCESS_RATIO: MetricDefinition(
            metric=ObjectiveMetric.EXPECTED_WEIGHTED_SUCCESS_RATIO,
            semantics_version="1",
            unit="ratio",
            scope=MetricScope.ASSIGNMENT_ADDITIVE,
            evaluate_plan=_plan_expected_weighted_success_ratio,
            candidate_proxy=(_candidate_expected_weighted_success_ratio),
            candidate_fidelity=CandidateFidelity.EXACT,
            proto_enum_name=("OBJECTIVE_METRIC_EXPECTED_WEIGHTED_SUCCESS_RATIO"),
        ),
        ObjectiveMetric.NORMALIZED_COMMUNICATION_RATIO: MetricDefinition(
            metric=ObjectiveMetric.NORMALIZED_COMMUNICATION_RATIO,
            semantics_version="1",
            unit="ratio",
            scope=MetricScope.PLAN_GLOBAL,
            evaluate_plan=(_plan_normalized_communication_ratio_metric),
            candidate_proxy=(_candidate_normalized_communication_ratio),
            candidate_fidelity=CandidateFidelity.PROXY,
            proto_enum_name=("OBJECTIVE_METRIC_NORMALIZED_COMMUNICATION_RATIO"),
        ),
        ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION: MetricDefinition(
            metric=ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION,
            semantics_version="1",
            unit="ratio",
            scope=MetricScope.TIMELINE,
            evaluate_plan=_plan_maximum_resource_utilization,
            candidate_proxy=(_candidate_maximum_resource_utilization),
            candidate_fidelity=CandidateFidelity.PROXY,
            proto_enum_name=("OBJECTIVE_METRIC_MAXIMUM_RESOURCE_UTILIZATION"),
        ),
        ObjectiveMetric.DEFERRED_PRIORITY_PENALTY: MetricDefinition(
            metric=ObjectiveMetric.DEFERRED_PRIORITY_PENALTY,
            semantics_version="1",
            unit="penalty",
            scope=MetricScope.PLAN_GLOBAL,
            evaluate_plan=_plan_deferred_priority_penalty,
            candidate_proxy=None,
            candidate_fidelity=CandidateFidelity.UNSUPPORTED,
            proto_enum_name=("OBJECTIVE_METRIC_DEFERRED_PRIORITY_PENALTY"),
        ),
    }
)

def _metric_definition(metric: ObjectiveMetric) -> MetricDefinition:
    try:
        return BUILTIN_METRICS[metric]
    except KeyError as exc:
        value = metric.value if isinstance(metric, ObjectiveMetric) else metric
        raise ValueError(f"unsupported objective metric {value!r}") from exc


if frozenset(BUILTIN_METRICS) != frozenset(ObjectiveMetric):
    raise RuntimeError("built-in metric registry must cover ObjectiveMetric")
if any(
    metric is not definition.metric for metric, definition in BUILTIN_METRICS.items()
):
    raise RuntimeError("built-in metric registry keys must match definitions")
if len({definition.proto_enum_name for definition in BUILTIN_METRICS.values()}) != len(
    BUILTIN_METRICS
):
    raise RuntimeError("built-in metric proto enum names must be unique")
