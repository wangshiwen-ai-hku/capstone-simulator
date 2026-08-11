"""Exhaustive one-hot computation-placement optimizer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import product
import math
from time import perf_counter

from ..domain.execution import Assignment
from ..domain.transfer import TransferReservation
from .base import (
    CandidateEstimate,
    CandidateMaterializationError,
    PlannedResourceReservation,
    SchedulingPlan,
    SchedulingProblem,
    SolveStatus,
    execution_mode_for_candidate,
    maximum_resource_utilization,
)
from .evaluation import (
    evaluate_constraints,
    evaluate_objectives,
    expected_weighted_success_ratio,
    objective_key,
    plan_normalized_communication_ratio,
)
from .heuristics import _materialize_candidate
from .policy import binary_offload_policy


class BinaryOffloadOptimizer:
    """Enumerate one feasible placement target for every ready task."""

    optimizer_id = "binary_offload"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 2.0,
    ) -> None:
        self.default_policy = binary_offload_policy(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.solve_history: list[dict[str, object]] = []

    def solve(self, problem: SchedulingProblem) -> SchedulingPlan:
        """Return the best complete incumbent within the declared limits."""

        solve_started = perf_counter()
        history: dict[str, object] = {
            "epoch_id": problem.epoch.epoch_id,
            "ready_task_count": len(problem.epoch.ready_tasks),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "solve_status": SolveStatus.ERROR.value,
            "termination_reason": "solve_raised_before_completion",
            "has_incumbent": False,
            "solve_elapsed_ms": 0.0,
            "enumerated_combinations": 0,
            "evaluated_combinations": 0,
            "total_combinations": 0,
            "solve_budget_ms": problem.solve_budget_ms,
            "max_iterations": problem.solve_limits.max_iterations,
        }
        self.solve_history.append(history)
        try:
            return self._solve(problem, solve_started, history)
        except Exception as exc:
            if history["termination_reason"] == "solve_raised_before_completion":
                history["solve_status"] = (
                    SolveStatus.INFEASIBLE.value
                    if isinstance(exc, ValueError)
                    and "no feasible" in str(exc).lower()
                    else SolveStatus.ERROR.value
                )
                history["termination_reason"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            raise
        finally:
            history["solve_elapsed_ms"] = self._elapsed_ms(solve_started)

    def _solve(
        self,
        problem: SchedulingProblem,
        solve_started: float,
        history: dict[str, object],
    ) -> SchedulingPlan:
        tasks = tuple(problem.epoch.ready_tasks)
        candidate_options = tuple(
            self._placement_candidates(problem, task.task_id)
            for task in tasks
        )
        total_combinations = math.prod(
            len(options) for options in candidate_options
        )
        history["total_combinations"] = total_combinations
        best_key: tuple[float, ...] | None = None
        best_tie_break: tuple[str, ...] | None = None
        best_selection: tuple[CandidateEstimate, ...] | None = None
        best_plan: SchedulingPlan | None = None
        evaluated = 0
        bounded_status: SolveStatus | None = None

        for selection in product(*candidate_options):
            if (
                problem.solve_limits.max_iterations
                and evaluated >= problem.solve_limits.max_iterations
            ):
                bounded_status = SolveStatus.ITERATION_LIMIT
                break
            if self._elapsed_ms(solve_started) >= problem.solve_budget_ms:
                bounded_status = SolveStatus.TIME_LIMIT
                break

            evaluated += 1
            history["enumerated_combinations"] = evaluated
            history["evaluated_combinations"] = evaluated
            if not self._is_feasible(problem, selection):
                continue
            try:
                materialized = self._materialize(problem, selection)
            except CandidateMaterializationError:
                continue
            draft = self._draft_plan(
                problem,
                materialized,
                iteration_count=evaluated,
            )
            objective_evaluations = evaluate_objectives(problem, draft)
            constraint_evaluations = evaluate_constraints(problem, draft)
            if any(
                item.hard and not item.satisfied
                for item in constraint_evaluations
            ):
                continue
            evaluated_key = objective_key(
                problem.policy,
                objective_evaluations,
                constraint_evaluations,
            )
            if any(not math.isfinite(value) for value in evaluated_key):
                continue
            tie_break = tuple(item.node_id for item in selection)
            if (
                best_key is None
                or evaluated_key < best_key
                or (
                    evaluated_key == best_key
                    and tie_break < (best_tie_break or ())
                )
            ):
                best_key = evaluated_key
                best_tie_break = tie_break
                best_selection = selection
                history["has_incumbent"] = True
                best_plan = replace(
                    draft,
                    objective_value=(
                        evaluated_key[0] if evaluated_key else 0.0
                    ),
                    objective_key=evaluated_key,
                    objective_evaluations=objective_evaluations,
                    constraint_evaluations=constraint_evaluations,
                )

        if best_selection is None or best_plan is None or best_key is None:
            if bounded_status is SolveStatus.TIME_LIMIT:
                history["solve_status"] = SolveStatus.TIME_LIMIT.value
                history["termination_reason"] = (
                    "solve_budget_reached_without_incumbent"
                )
                raise TimeoutError(
                    "binary offload solve budget expired before an incumbent"
                )
            if bounded_status is SolveStatus.ITERATION_LIMIT:
                history["solve_status"] = (
                    SolveStatus.ITERATION_LIMIT.value
                )
                history["termination_reason"] = (
                    "max_iterations_reached_without_incumbent"
                )
                raise RuntimeError(
                    "binary offload iteration limit reached before an incumbent"
                )
            history["solve_status"] = SolveStatus.INFEASIBLE.value
            history["termination_reason"] = (
                "exhaustive_search_found_no_feasible_assignment"
            )
            raise ValueError(
                "binary offload problem has no feasible complete assignment"
            )

        exhaustive = bounded_status is None
        solve_status = (
            SolveStatus.OPTIMAL if exhaustive else bounded_status
        )
        assert solve_status is not None
        termination_reason = (
            "exhaustive_one_hot_search_complete"
            if exhaustive
            else "solve_budget_reached_with_incumbent"
            if solve_status is SolveStatus.TIME_LIMIT
            else "max_iterations_reached_with_incumbent"
        )
        assignment_reason = (
            "optimal exhaustive one-hot placement"
            if exhaustive
            else "best one-hot placement incumbent within solve limits"
        )
        elapsed_ms = self._elapsed_ms(solve_started)
        final_plan = replace(
            best_plan,
            solve_status=solve_status,
            solve_elapsed_ms=elapsed_ms,
            iteration_count=evaluated,
            termination_reason=termination_reason,
            assignments=tuple(
                replace(item, reason=assignment_reason)
                for item in best_plan.assignments
            ),
            diagnostics={
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "binary_objective_value": best_key[0],
                "enumerated_combinations": evaluated,
                "total_combinations": total_combinations,
                "placement_search_exhaustive": int(exhaustive),
                "solve_budget_ms": problem.solve_budget_ms,
                "max_iterations": problem.solve_limits.max_iterations,
            },
        )
        success_ratio = expected_weighted_success_ratio(
            problem,
            final_plan.assignments,
        )
        communication_ratio = plan_normalized_communication_ratio(
            problem,
            final_plan.assignments,
        )
        maximum_utilization = maximum_resource_utilization(
            problem,
            final_plan.node_reservations,
        )
        communication_ms = sum(
            item.communication_ms for item in final_plan.assignments
        )
        history.update(
            {
                "epoch_id": problem.epoch.epoch_id,
                "ready_task_count": len(tasks),
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "success_reward": success_ratio,
                "expected_weighted_success_ratio": success_ratio,
                "communication_time_ms": communication_ms,
                "normalized_communication_ratio": communication_ratio,
                "maximum_resource_utilization": maximum_utilization,
                "objective_value": best_key[0],
                "solve_elapsed_ms": elapsed_ms,
                "enumerated_combinations": evaluated,
                "total_combinations": total_combinations,
                "placement_search_exhaustive": exhaustive,
                "has_incumbent": True,
                "solve_status": solve_status.value,
                "termination_reason": termination_reason,
                "solve_budget_ms": problem.solve_budget_ms,
                "max_iterations": problem.solve_limits.max_iterations,
                "assignments": {
                    item.task_id: item.node_id for item in best_selection
                },
            }
        )
        return final_plan

    @staticmethod
    def _elapsed_ms(solve_started: float) -> float:
        return (perf_counter() - solve_started) * 1000.0

    @staticmethod
    def _placement_candidates(
        problem: SchedulingProblem,
        task_id: str,
    ) -> tuple[CandidateEstimate, ...]:
        candidates = tuple(
            sorted(
                (
                    candidate
                    for candidate in problem.candidates[task_id]
                    if candidate.feasible
                ),
                key=lambda item: (item.node_id, item.node_kind.value),
            )
        )
        if not candidates:
            raise ValueError(
                f"task {task_id} has no feasible placement candidate"
            )
        return candidates

    @staticmethod
    def _is_feasible(
        problem: SchedulingProblem,
        selection: tuple[CandidateEstimate, ...],
    ) -> bool:
        energy_by_node = defaultdict(float)
        snapshots = problem.snapshot_by_id

        for candidate in selection:
            if not snapshots[candidate.node_id].online:
                return False
            energy_by_node[candidate.node_id] += candidate.energy_j

        for node_id, requested_energy in energy_by_node.items():
            remaining_energy = snapshots[node_id].remaining_energy_j
            if (
                remaining_energy is not None
                and requested_energy > remaining_energy + 1e-9
            ):
                return False
        return True

    def _draft_plan(
        self,
        problem: SchedulingProblem,
        materialized: tuple[
            tuple[Assignment, ...],
            tuple[PlannedResourceReservation, ...],
            tuple[TransferReservation, ...],
        ],
        *,
        iteration_count: int,
    ) -> SchedulingPlan:
        assignments, node_reservations, transfer_reservations = materialized
        return SchedulingPlan(
            problem_id=problem.problem_id,
            snapshot_id=problem.snapshot.snapshot_id,
            policy_id=problem.policy.policy_id,
            policy_version=problem.policy.version,
            epoch_id=problem.epoch.epoch_id,
            optimizer_id=self.optimizer_id,
            optimizer_version="2",
            solve_status=SolveStatus.FEASIBLE,
            iteration_count=iteration_count,
            assignments=assignments,
            node_reservations=node_reservations,
            transfer_reservations=transfer_reservations,
        )

    def _materialize(
        self,
        problem: SchedulingProblem,
        selection: tuple[CandidateEstimate, ...],
    ) -> tuple[
        tuple[Assignment, ...],
        tuple[PlannedResourceReservation, ...],
        tuple[TransferReservation, ...],
    ]:
        link_available = dict(problem.link_available_ms)
        assignments: list[Assignment] = []
        node_reservations: list[PlannedResourceReservation] = []
        reservations_by_node: dict[
            str, list[PlannedResourceReservation]
        ] = defaultdict(list)
        for reservation in problem.existing_node_reservations:
            reservations_by_node[reservation.node_id].append(reservation)
        transfer_reservations: list[TransferReservation] = []

        for candidate in selection:
            task = problem.task_by_id[candidate.task_id]
            (
                materialized,
                task_transfers,
                next_links,
                compute_start,
            ) = _materialize_candidate(
                problem,
                candidate,
                reservations_by_node,
                link_available,
            )
            link_available.update(next_links)
            transfer_reservations.extend(task_transfers)
            link_ids = tuple(
                dict.fromkeys(
                    link_id
                    for reservation in task_transfers
                    for link_id in reservation.path_link_ids
                )
            )
            assignments.append(
                Assignment(
                    task_id=candidate.task_id,
                    target_node_id=candidate.node_id,
                    execution_mode=execution_mode_for_candidate(
                        task,
                        candidate,
                    ),
                    estimated_start_ms=materialized.start_ms,
                    estimated_finish_ms=materialized.finish_ms,
                    compute_ms=materialized.compute_ms,
                    communication_ms=materialized.communication_ms,
                    energy_j=materialized.energy_j,
                    reason="one-hot placement candidate",
                    input_locations=materialized.input_locations,
                    transfer_link_ids=link_ids,
                    optimizer_id=self.optimizer_id,
                    epoch_id=problem.epoch.epoch_id,
                    output_size_mb=materialized.output_size_mb,
                    success_probability=(
                        materialized.success_probability
                    ),
                )
            )
            node_reservation = PlannedResourceReservation(
                reservation_id=(
                    f"binary-node:{problem.epoch.epoch_id}:"
                    f"{candidate.task_id}:{candidate.node_id}"
                ),
                epoch_id=problem.epoch.epoch_id,
                task_id=candidate.task_id,
                node_id=candidate.node_id,
                start_ms=compute_start,
                finish_ms=materialized.finish_ms,
                demand=materialized.resource_demand,
            )
            node_reservations.append(node_reservation)
            reservations_by_node[candidate.node_id].append(node_reservation)

        return (
            tuple(assignments),
            tuple(node_reservations),
            tuple(transfer_reservations),
        )


__all__ = ["BinaryOffloadOptimizer"]
