"""Shared, solver-independent objective and constraint evaluation."""

from __future__ import annotations

from ..models import (
    ExecutionMode,
    NodeKind,
    TaskInstance,
    resolved_placement_constraints,
)
from .base import CandidateEstimate, SchedulingPlan, SchedulingProblem
from .policy import (
    ConstraintEvaluation,
    ConstraintRelation,
    ObjectiveAggregation,
    ObjectiveEvaluation,
    ObjectiveMetric,
    OptimizationDirection,
    SchedulingPolicy,
)


def candidate_objective_key(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
) -> tuple[float, ...]:
    """Evaluate one candidate using the same typed policy terms as a plan."""

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
            *(
                (item.priority_order, item.weighted_value)
                for item in evaluations
            ),
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
        if (
            constraint.relation
            is ConstraintRelation.LESS_THAN_OR_EQUAL
        ):
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
                    0.0
                    if constraint.hard
                    else violation * constraint.violation_penalty
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
        normalized
        if direction is OptimizationDirection.MINIMIZE
        else -normalized
    )
    return directed * weight


def _candidate_metric_value(
    problem: SchedulingProblem,
    task_id: str,
    candidate: CandidateEstimate,
    metric: ObjectiveMetric,
) -> float:
    task = problem.task_by_id[task_id]
    projected_finish = (
        candidate.finish_ms
        + problem.critical_tail_ms.get(task_id, 0.0)
    )
    if metric in {
        ObjectiveMetric.MAKESPAN_MS,
        ObjectiveMetric.TOTAL_COMPLETION_TIME_MS,
    }:
        return candidate.finish_ms
    if metric is ObjectiveMetric.TOTAL_DEADLINE_VIOLATION_MS:
        return max(0.0, projected_finish - task.deadline_time_ms)
    if metric is ObjectiveMetric.CRITICAL_PATH_FINISH_MS:
        return projected_finish
    if metric is ObjectiveMetric.TOTAL_ENERGY_J:
        return candidate.energy_j
    if metric is ObjectiveMetric.TOTAL_COMMUNICATION_MS:
        return candidate.communication_ms
    if metric is ObjectiveMetric.LOCALITY_PENALTY:
        return _candidate_locality_penalty(candidate)
    if metric is ObjectiveMetric.DROPPED_TASKS:
        return 0.0
    if metric is ObjectiveMetric.NON_SOURCE_ASSIGNMENTS:
        return float(not candidate.is_source)
    if metric is ObjectiveMetric.NON_EDGE_ASSIGNMENTS:
        return float(candidate.node_kind is not NodeKind.EDGE)
    if metric is ObjectiveMetric.PLACEMENT_PREFERENCE_PENALTY:
        return float(_preference_rank(task, candidate))
    if metric is ObjectiveMetric.RULE_MISMATCH_COUNT:
        return float(_rule_mismatch(problem, task, candidate))
    raise ValueError(f"unsupported objective metric {metric.value!r}")


def _plan_metric_value(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
    metric: ObjectiveMetric,
) -> float:
    non_drop = tuple(
        assignment
        for assignment in plan.assignments
        if assignment.execution_mode is not ExecutionMode.DROP
    )
    if metric is ObjectiveMetric.MAKESPAN_MS:
        return max(
            (
                assignment.estimated_finish_ms
                for assignment in non_drop
            ),
            default=problem.epoch.now_ms,
        )
    if metric is ObjectiveMetric.TOTAL_DEADLINE_VIOLATION_MS:
        return sum(
            max(
                0.0,
                assignment.estimated_finish_ms
                + problem.critical_tail_ms.get(
                    assignment.task_id,
                    0.0,
                )
                - problem.task_by_id[
                    assignment.task_id
                ].deadline_time_ms,
            )
            for assignment in non_drop
        )
    if metric is ObjectiveMetric.TOTAL_COMPLETION_TIME_MS:
        return sum(
            assignment.estimated_finish_ms
            for assignment in non_drop
        )
    if metric is ObjectiveMetric.CRITICAL_PATH_FINISH_MS:
        return max(
            (
                assignment.estimated_finish_ms
                + problem.critical_tail_ms.get(
                    assignment.task_id,
                    0.0,
                )
                for assignment in non_drop
            ),
            default=problem.epoch.now_ms,
        )
    if metric is ObjectiveMetric.TOTAL_ENERGY_J:
        return sum(assignment.energy_j for assignment in non_drop)
    if metric is ObjectiveMetric.TOTAL_COMMUNICATION_MS:
        return sum(
            assignment.communication_ms for assignment in non_drop
        )
    if metric is ObjectiveMetric.LOCALITY_PENALTY:
        return sum(
            len(
                set(assignment.input_locations)
                - {assignment.target_node_id}
            )
            * 2.0
            for assignment in non_drop
        )
    if metric is ObjectiveMetric.DROPPED_TASKS:
        return float(
            sum(
                assignment.execution_mode is ExecutionMode.DROP
                for assignment in plan.assignments
            )
            + len(plan.deferred_task_ids)
        )
    if metric is ObjectiveMetric.NON_SOURCE_ASSIGNMENTS:
        return float(
            sum(
                assignment.target_node_id
                != problem.task_by_id[
                    assignment.task_id
                ].source_node_id
                for assignment in non_drop
            )
        )
    if metric is ObjectiveMetric.NON_EDGE_ASSIGNMENTS:
        return float(
            sum(
                problem.node_by_id[
                    assignment.target_node_id
                ].kind
                is not NodeKind.EDGE
                for assignment in non_drop
            )
        )
    if metric is ObjectiveMetric.PLACEMENT_PREFERENCE_PENALTY:
        selected = _selected_candidates(problem)
        return float(
            sum(
                _preference_rank(
                    problem.task_by_id[assignment.task_id],
                    selected[assignment.task_id][
                        assignment.target_node_id
                    ],
                )
                for assignment in non_drop
            )
        )
    if metric is ObjectiveMetric.RULE_MISMATCH_COUNT:
        selected = _selected_candidates(problem)
        return float(
            sum(
                _rule_mismatch(
                    problem,
                    problem.task_by_id[assignment.task_id],
                    selected[assignment.task_id][
                        assignment.target_node_id
                    ],
                )
                for assignment in non_drop
            )
        )
    raise ValueError(f"unsupported objective metric {metric.value!r}")


def _selected_candidates(
    problem: SchedulingProblem,
) -> dict[str, dict[str, CandidateEstimate]]:
    return {
        task_id: {
            candidate.node_id: candidate
            for candidate in candidates
        }
        for task_id, candidates in problem.candidates.items()
    }


def _candidate_locality_penalty(candidate: CandidateEstimate) -> float:
    return float(
        len(set(candidate.input_locations) - {candidate.node_id}) * 2.0
    )


def _preference_rank(
    task: TaskInstance,
    candidate: CandidateEstimate,
) -> int:
    preferred = resolved_placement_constraints(
        task
    ).preferred_node_kinds
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
