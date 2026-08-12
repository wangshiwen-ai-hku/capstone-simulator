"""Exactly-one placement formulation for a rolling-horizon ready batch."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from dataclasses import replace
import math

from ..base import CandidateEstimate, SchedulingPlan, SchedulingProblem
from ..evaluation import (
    BUILTIN_METRICS,
    evaluate_constraints,
    evaluate_objectives,
    metric_contract_id,
    objective_key,
)
from ..formulation import (
    CompiledFormulation,
    FormulationCompatibilityError,
    FormulationDomainError,
    FormulationError,
    FormulationEvaluation,
    FormulationSpec,
    MaterializedSchedule,
)
from ..materialization import (
    build_assignment,
    build_node_reservation,
    materialize_candidate,
)


ONE_HOT_PLACEMENT_SPEC = FormulationSpec(
    formulation_id="one_hot_placement",
    formulation_version="1",
    materializer_id="serial_transfer_earliest_resource",
    materializer_version="1",
    options={
        "assignment_cardinality": "exactly_one",
        "allow_drop": False,
        "allow_defer": False,
        "allow_split": False,
        "allow_replication": False,
        "task_order": "epoch",
        "candidate_order": "node_id",
    },
)


@dataclass(frozen=True, slots=True)
class OneHotPlacementModel:
    """Compiled exactly-one decision domain for one immutable Problem."""

    problem_id: str
    formulation_spec: FormulationSpec
    metric_contract_id: str
    ordered_task_ids: tuple[str, ...]
    candidate_options: tuple[tuple[CandidateEstimate, ...], ...]
    objective_ids: tuple[str, ...]
    constraint_ids: tuple[str, ...]
    referenced_metric_versions: tuple[tuple[str, str], ...]
    total_decisions: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ordered_task_ids",
            tuple(self.ordered_task_ids),
        )
        object.__setattr__(
            self,
            "candidate_options",
            tuple(tuple(options) for options in self.candidate_options),
        )
        object.__setattr__(self, "objective_ids", tuple(self.objective_ids))
        object.__setattr__(self, "constraint_ids", tuple(self.constraint_ids))
        object.__setattr__(
            self,
            "referenced_metric_versions",
            tuple(self.referenced_metric_versions),
        )
        if len(self.ordered_task_ids) != len(self.candidate_options):
            raise ValueError(
                "one-hot task ids and candidate groups must align"
            )
        if any(not options for options in self.candidate_options):
            raise ValueError(
                "one-hot placement requires a candidate for every task"
            )
        expected_total = math.prod(
            len(options) for options in self.candidate_options
        )
        if self.total_decisions != expected_total:
            raise ValueError("one-hot total_decisions is inconsistent")


@dataclass(frozen=True, slots=True)
class OneHotPlacementDecision:
    """One selected feasible candidate for every compiled task group."""

    selected_candidates: tuple[CandidateEstimate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_candidates",
            tuple(self.selected_candidates),
        )


@dataclass(frozen=True, slots=True)
class OneHotPlacementFormulation:
    """Compile and decode exactly-one task-to-node placement decisions.

    ``OPTIMAL`` for this formulation means optimal over the fixed ready-task
    order and candidate groups compiled here.  Drop, defer, split, replication,
    alternate task order, and free start-time decisions are outside this v1
    decision domain.
    """

    spec: FormulationSpec = field(
        default=ONE_HOT_PLACEMENT_SPEC,
        init=False,
    )

    def compile(self, problem: SchedulingProblem) -> OneHotPlacementModel:
        expected_metric_contract = metric_contract_id(problem.policy)
        if problem.metric_contract_id != expected_metric_contract:
            raise FormulationCompatibilityError(
                "problem metric contract does not match the active metric "
                "registry"
            )
        referenced_metrics = sorted(
            {
                item.metric
                for item in (
                    *problem.policy.objectives,
                    *problem.policy.constraints,
                )
            },
            key=lambda metric: metric.value,
        )
        missing = [
            metric.value
            for metric in referenced_metrics
            if metric not in BUILTIN_METRICS
        ]
        if missing:
            raise FormulationCompatibilityError(
                "one-hot placement has no exact plan evaluator for metrics "
                f"{missing}"
            )

        task_ids = tuple(
            task.task_id for task in problem.epoch.ready_tasks
        )
        if not task_ids:
            raise FormulationError(
                "one-hot placement requires at least one ready task"
            )
        candidate_options = []
        for task_id in task_ids:
            candidates = tuple(
                sorted(
                    (
                        candidate
                        for candidate in problem.candidates[task_id]
                        if candidate.feasible
                    ),
                    key=lambda item: (
                        item.node_id,
                        item.node_kind.value,
                    ),
                )
            )
            if not candidates:
                raise FormulationError(
                    f"task {task_id} has no feasible placement candidate"
                )
            candidate_options.append(candidates)

        return OneHotPlacementModel(
            problem_id=problem.problem_id,
            formulation_spec=self.spec,
            metric_contract_id=problem.metric_contract_id,
            ordered_task_ids=task_ids,
            candidate_options=tuple(candidate_options),
            objective_ids=tuple(
                item.objective_id for item in problem.policy.objectives
            ),
            constraint_ids=tuple(
                item.constraint_id for item in problem.policy.constraints
            ),
            referenced_metric_versions=tuple(
                (
                    metric.value,
                    BUILTIN_METRICS[metric].semantics_version,
                )
                for metric in referenced_metrics
            ),
            total_decisions=math.prod(
                len(options) for options in candidate_options
            ),
        )

    def decision(
        self,
        model: OneHotPlacementModel,
        selection: tuple[CandidateEstimate, ...],
    ) -> OneHotPlacementDecision:
        selected = tuple(selection)
        if len(selected) != len(model.ordered_task_ids):
            raise FormulationError(
                "one-hot decision must select one candidate per task"
            )
        for task_id, options, candidate in zip(
            model.ordered_task_ids,
            model.candidate_options,
            selected,
            strict=True,
        ):
            if candidate.task_id != task_id or candidate not in options:
                raise FormulationError(
                    f"one-hot decision contains an unknown candidate for "
                    f"task {task_id}"
                )
        return OneHotPlacementDecision(selected)

    def is_decision_feasible(
        self,
        problem: SchedulingProblem,
        model: OneHotPlacementModel,
        decision: OneHotPlacementDecision,
    ) -> bool:
        self._validate_model(problem, model)
        checked = self.decision(model, decision.selected_candidates)
        snapshots = problem.snapshot_by_id
        energy_by_node = defaultdict(float)
        for candidate in checked.selected_candidates:
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

    def materialize(
        self,
        problem: SchedulingProblem,
        model: CompiledFormulation,
        decision: object,
        *,
        optimizer_id: str,
    ) -> MaterializedSchedule:
        typed_model = self._typed_model(model)
        self._validate_model(problem, typed_model)
        if not isinstance(decision, OneHotPlacementDecision):
            raise TypeError(
                "one-hot placement requires OneHotPlacementDecision"
            )
        checked = self.decision(
            typed_model,
            decision.selected_candidates,
        )
        if not optimizer_id.strip():
            raise ValueError("optimizer_id must be non-blank")

        link_available = dict(problem.link_available_ms)
        assignments = []
        node_reservations = []
        reservations_by_node = defaultdict(list)
        for reservation in problem.existing_node_reservations:
            reservations_by_node[reservation.node_id].append(reservation)
        transfer_reservations = []

        for candidate in checked.selected_candidates:
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
                    optimizer_id=optimizer_id,
                    reason="one-hot placement decision",
                )
            )
            node_reservation = build_node_reservation(
                problem,
                materialized,
                compute_start_ms=compute_start,
                reservation_id=(
                    f"one-hot-node:{problem.epoch.epoch_id}:"
                    f"{candidate.task_id}:{candidate.node_id}"
                ),
            )
            node_reservations.append(node_reservation)
            reservations_by_node[candidate.node_id].append(
                node_reservation
            )

        return MaterializedSchedule(
            assignments=tuple(assignments),
            node_reservations=tuple(node_reservations),
            transfer_reservations=tuple(transfer_reservations),
        )

    def evaluate(
        self,
        problem: SchedulingProblem,
        model: CompiledFormulation,
        plan: SchedulingPlan,
    ) -> FormulationEvaluation:
        typed_model = self._typed_model(model)
        self._validate_model(problem, typed_model)
        objectives = evaluate_objectives(problem, plan)
        constraints = evaluate_constraints(problem, plan)
        if tuple(item.objective_id for item in objectives) != (
            typed_model.objective_ids
        ):
            raise FormulationCompatibilityError(
                "one-hot objective encoding does not cover the policy"
            )
        if tuple(item.constraint_id for item in constraints) != (
            typed_model.constraint_ids
        ):
            raise FormulationCompatibilityError(
                "one-hot constraint encoding does not cover the policy"
            )
        return FormulationEvaluation(
            objective_evaluations=objectives,
            constraint_evaluations=constraints,
            objective_key=objective_key(
                problem.policy,
                objectives,
                constraints,
            ),
        )

    def validate_plan_domain(
        self,
        problem: SchedulingProblem,
        model: CompiledFormulation,
        plan: SchedulingPlan,
    ) -> None:
        """Prove that a Plan is the canonical decode of one one-hot decision."""

        typed_model = self._typed_model(model)
        self._validate_model(problem, typed_model)
        if plan.deferred_task_ids:
            raise FormulationDomainError(
                "one-hot placement does not permit deferred tasks"
            )
        if tuple(item.task_id for item in plan.assignments) != (
            typed_model.ordered_task_ids
        ):
            raise FormulationDomainError(
                "one-hot placement requires exactly one assignment per "
                "compiled task in formulation order"
            )
        selected = []
        for task_id, options, assignment in zip(
            typed_model.ordered_task_ids,
            typed_model.candidate_options,
            plan.assignments,
            strict=True,
        ):
            candidate = next(
                (
                    item
                    for item in options
                    if item.node_id == assignment.target_node_id
                ),
                None,
            )
            if candidate is None:
                raise FormulationDomainError(
                    f"one-hot assignment for task {task_id} does not select "
                    "a compiled candidate"
                )
            selected.append(candidate)

        decision = self.decision(typed_model, tuple(selected))
        canonical = self.materialize(
            problem,
            typed_model,
            decision,
            optimizer_id=plan.optimizer_id,
        )
        actual_assignments = tuple(
            replace(item, reason="") for item in plan.assignments
        )
        expected_assignments = tuple(
            replace(item, reason="") for item in canonical.assignments
        )
        if actual_assignments != expected_assignments:
            raise FormulationDomainError(
                "one-hot assignments do not match the canonical materializer"
            )
        if plan.node_reservations != canonical.node_reservations:
            raise FormulationDomainError(
                "one-hot node reservations do not match the canonical "
                "materializer"
            )
        if plan.transfer_reservations != canonical.transfer_reservations:
            raise FormulationDomainError(
                "one-hot transfer reservations do not match the canonical "
                "materializer"
            )

    @staticmethod
    def _typed_model(
        model: CompiledFormulation,
    ) -> OneHotPlacementModel:
        if not isinstance(model, OneHotPlacementModel):
            raise TypeError(
                "one-hot placement requires OneHotPlacementModel"
            )
        return model

    def _validate_model(
        self,
        problem: SchedulingProblem,
        model: OneHotPlacementModel,
    ) -> None:
        if model.problem_id != problem.problem_id:
            raise FormulationCompatibilityError(
                "one-hot model belongs to a different problem"
            )
        if model.formulation_spec != self.spec:
            raise FormulationCompatibilityError(
                "one-hot model uses a different formulation spec"
            )
        if model.metric_contract_id != problem.metric_contract_id:
            raise FormulationCompatibilityError(
                "one-hot model uses a different metric contract"
            )


__all__ = [
    "ONE_HOT_PLACEMENT_SPEC",
    "OneHotPlacementDecision",
    "OneHotPlacementFormulation",
    "OneHotPlacementModel",
]
