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
)
from .evaluation import (
    evaluate_constraints,
    evaluate_objectives,
    objective_key,
)
from .materialization import (
    build_assignment,
    build_node_reservation,
    materialize_candidate,
)
from .policy import binary_offload_policy
from .state import (
    OptimizerSolveState,
    SolveTraceContext,
    SolveTracePhase,
)


class BinaryOffloadOptimizer:
    """Enumerate one feasible placement target for every ready task."""

    optimizer_id = "binary_offload"
    optimizer_version = "3"
    solve_work_unit = "placement_combination"

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

    def solve(self, problem: SchedulingProblem) -> SchedulingPlan:
        """Return a plan without retaining state on the optimizer instance."""

        state = OptimizerSolveState(
            session_id=f"standalone:{problem.problem_id}"
        )
        return self.solve_with_state(problem, state)

    def solve_with_state(
        self,
        problem: SchedulingProblem,
        state: OptimizerSolveState,
        *,
        context: SolveTraceContext | None = None,
    ) -> SchedulingPlan:
        """Solve while appending an auditable trace to caller-owned state."""

        solve_started = perf_counter()
        if context is None:
            context = state.begin(
                problem,
                optimizer_id=self.optimizer_id,
                optimizer_version=self.optimizer_version,
                work_unit=self.solve_work_unit,
            )
        elif (
            context.problem_id != problem.problem_id
            or context.optimizer_id != self.optimizer_id
        ):
            raise ValueError(
                "solve trace context does not match problem and optimizer"
            )
        try:
            return self._solve(
                problem,
                solve_started,
                state,
                context,
            )
        except Exception as exc:
            if not any(
                entry.context.solve_id == context.solve_id
                and entry.phase is SolveTracePhase.FAILED
                for entry in state.entries
            ):
                state.record(
                    context,
                    SolveTracePhase.FAILED,
                    elapsed_ms=self._elapsed_ms(solve_started),
                    solve_status=(
                        SolveStatus.INFEASIBLE
                        if isinstance(exc, ValueError)
                        and "no feasible" in str(exc).lower()
                        else SolveStatus.ERROR
                    ),
                    termination_reason=f"{type(exc).__name__}: {exc}",
                    details={
                        "ready_task_count": len(
                            problem.epoch.ready_tasks
                        )
                    },
                )
            raise

    def _solve(
        self,
        problem: SchedulingProblem,
        solve_started: float,
        state: OptimizerSolveState,
        context: SolveTraceContext,
    ) -> SchedulingPlan:
        tasks = tuple(problem.epoch.ready_tasks)
        candidate_options = tuple(
            self._placement_candidates(problem, task.task_id)
            for task in tasks
        )
        total_combinations = math.prod(
            len(options) for options in candidate_options
        )
        best_key: tuple[float, ...] | None = None
        best_tie_break: tuple[str, ...] | None = None
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
                best_plan = replace(
                    draft,
                    objective_value=(
                        evaluated_key[0] if evaluated_key else 0.0
                    ),
                    objective_key=evaluated_key,
                    objective_evaluations=objective_evaluations,
                    constraint_evaluations=constraint_evaluations,
                )
                state.record(
                    context,
                    SolveTracePhase.INCUMBENT,
                    iteration=evaluated,
                    elapsed_ms=self._elapsed_ms(solve_started),
                    has_incumbent=True,
                    evaluated_work_units=evaluated,
                    total_work_units=total_combinations,
                    objective_key=evaluated_key,
                    objective_components={
                        item.objective_id: item.raw_value
                        for item in objective_evaluations
                    },
                    selected_targets={
                        item.task_id: item.node_id for item in selection
                    },
                    details={
                        "ready_task_count": len(tasks),
                    },
                )

        if best_plan is None or best_key is None:
            if bounded_status is SolveStatus.TIME_LIMIT:
                self._record_failure(
                    state,
                    context,
                    solve_started,
                    status=SolveStatus.TIME_LIMIT,
                    reason="solve_budget_reached_without_incumbent",
                    evaluated=evaluated,
                    total=total_combinations,
                    ready_task_count=len(tasks),
                )
                raise TimeoutError(
                    "binary offload solve budget expired before an incumbent"
                )
            if bounded_status is SolveStatus.ITERATION_LIMIT:
                self._record_failure(
                    state,
                    context,
                    solve_started,
                    status=SolveStatus.ITERATION_LIMIT,
                    reason="max_iterations_reached_without_incumbent",
                    evaluated=evaluated,
                    total=total_combinations,
                    ready_task_count=len(tasks),
                )
                raise RuntimeError(
                    "binary offload iteration limit reached before an incumbent"
                )
            self._record_failure(
                state,
                context,
                solve_started,
                status=SolveStatus.INFEASIBLE,
                reason="exhaustive_search_found_no_feasible_assignment",
                evaluated=evaluated,
                total=total_combinations,
                ready_task_count=len(tasks),
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
                "total_combinations": total_combinations,
            },
        )
        state.record(
            context,
            SolveTracePhase.COMPLETED,
            iteration=evaluated,
            elapsed_ms=elapsed_ms,
            solve_status=solve_status,
            termination_reason=termination_reason,
            has_incumbent=True,
            evaluated_work_units=evaluated,
            total_work_units=total_combinations,
            objective_key=best_key,
            objective_components={
                item.objective_id: item.raw_value
                for item in final_plan.objective_evaluations
            },
            selected_targets={
                item.task_id: item.target_node_id
                for item in final_plan.assignments
            },
            details={
                "ready_task_count": len(tasks),
                "communication_time_ms": sum(
                    item.communication_ms
                    for item in final_plan.assignments
                ),
            },
        )
        return final_plan

    def _record_failure(
        self,
        state: OptimizerSolveState,
        context: SolveTraceContext,
        solve_started: float,
        *,
        status: SolveStatus,
        reason: str,
        evaluated: int,
        total: int,
        ready_task_count: int,
    ) -> None:
        state.record(
            context,
            SolveTracePhase.FAILED,
            iteration=evaluated,
            elapsed_ms=self._elapsed_ms(solve_started),
            solve_status=status,
            termination_reason=reason,
            has_incumbent=False,
            evaluated_work_units=evaluated,
            total_work_units=total,
            details={"ready_task_count": ready_task_count},
        )

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
            optimizer_version=self.optimizer_version,
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
            (
                materialized,
                task_transfers,
                next_links,
                compute_start,
            ) = materialize_candidate(
                problem,
                candidate,
                reservations_by_node,
                link_available,
            )
            link_available.update(next_links)
            transfer_reservations.extend(task_transfers)
            assignments.append(
                build_assignment(
                    problem,
                    materialized,
                    task_transfers,
                    optimizer_id=self.optimizer_id,
                    reason="one-hot placement candidate",
                )
            )
            node_reservation = build_node_reservation(
                problem,
                materialized,
                compute_start_ms=compute_start,
                reservation_id=(
                    f"binary-node:{problem.epoch.epoch_id}:"
                    f"{candidate.task_id}:{candidate.node_id}"
                ),
            )
            node_reservations.append(node_reservation)
            reservations_by_node[candidate.node_id].append(node_reservation)

        return (
            tuple(assignments),
            tuple(node_reservations),
            tuple(transfer_reservations),
        )


__all__ = ["BinaryOffloadOptimizer"]
