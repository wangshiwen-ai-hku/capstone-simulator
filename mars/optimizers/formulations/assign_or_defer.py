"""Assign-one-or-defer formulation for a rolling-horizon ready batch."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from .one_hot import (
    OneHotPlacementDecision,
    OneHotPlacementFormulation,
    OneHotPlacementModel,
)


ASSIGN_OR_DEFER_SPEC = FormulationSpec(
    formulation_id="assign_or_defer",
    formulation_version="1",
    materializer_id="serial_transfer_earliest_resource",
    materializer_version="1",
    options={
        "assignment_cardinality": "zero_or_one",
        "allow_drop": False,
        "allow_defer": True,
        "allow_split": False,
        "allow_replication": False,
        "task_order": "epoch",
        "candidate_order": "node_id",
    },
)


@dataclass(frozen=True, slots=True)
class AssignOrDeferModel:
    """Compiled candidate groups with one explicit defer option per task."""

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
        object.__setattr__(self, "ordered_task_ids", tuple(self.ordered_task_ids))
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
            raise ValueError("assign-or-defer task ids and candidate groups must align")
        expected = math.prod(len(options) + 1 for options in self.candidate_options)
        if self.total_decisions != expected:
            raise ValueError("assign-or-defer total_decisions is inconsistent")


@dataclass(frozen=True, slots=True)
class AssignOrDeferDecision:
    """One candidate or an explicit defer marker for every ready task."""

    selections: tuple[CandidateEstimate | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selections", tuple(self.selections))


@dataclass(frozen=True, slots=True)
class AssignOrDeferFormulation:
    """Compile, decode, evaluate, and validate assign-or-defer decisions."""

    spec: FormulationSpec = field(default=ASSIGN_OR_DEFER_SPEC, init=False)
    _one_hot: OneHotPlacementFormulation = field(
        default_factory=OneHotPlacementFormulation,
        init=False,
        repr=False,
        compare=False,
    )

    def compile(self, problem: SchedulingProblem) -> AssignOrDeferModel:
        expected_metric_contract = metric_contract_id(problem.policy)
        if problem.metric_contract_id != expected_metric_contract:
            raise FormulationCompatibilityError(
                "problem metric contract does not match the active metric registry"
            )
        referenced_metrics = sorted(
            {
                item.metric
                for item in (*problem.policy.objectives, *problem.policy.constraints)
            },
            key=lambda metric: metric.value,
        )
        missing = [metric.value for metric in referenced_metrics if metric not in BUILTIN_METRICS]
        if missing:
            raise FormulationCompatibilityError(
                "assign-or-defer has no plan evaluator for metrics " f"{missing}"
            )
        task_ids = tuple(task.task_id for task in problem.epoch.ready_tasks)
        if not task_ids:
            raise FormulationError("assign-or-defer requires at least one ready task")
        candidate_options = tuple(
            tuple(
                sorted(
                    (
                        candidate
                        for candidate in problem.candidates[task_id]
                        if candidate.feasible
                    ),
                    key=lambda item: (item.node_id, item.node_kind.value),
                )
            )
            for task_id in task_ids
        )
        return AssignOrDeferModel(
            problem_id=problem.problem_id,
            formulation_spec=self.spec,
            metric_contract_id=problem.metric_contract_id,
            ordered_task_ids=task_ids,
            candidate_options=candidate_options,
            objective_ids=tuple(item.objective_id for item in problem.policy.objectives),
            constraint_ids=tuple(item.constraint_id for item in problem.policy.constraints),
            referenced_metric_versions=tuple(
                (metric.value, BUILTIN_METRICS[metric].semantics_version)
                for metric in referenced_metrics
            ),
            total_decisions=math.prod(len(options) + 1 for options in candidate_options),
        )

    def decision(
        self,
        model: AssignOrDeferModel,
        selections: tuple[CandidateEstimate | None, ...],
    ) -> AssignOrDeferDecision:
        selected = tuple(selections)
        if len(selected) != len(model.ordered_task_ids):
            raise FormulationError("assign-or-defer requires one decision per task")
        for task_id, options, candidate in zip(
            model.ordered_task_ids,
            model.candidate_options,
            selected,
            strict=True,
        ):
            if candidate is not None and (
                candidate.task_id != task_id or candidate not in options
            ):
                raise FormulationError(
                    f"assign-or-defer contains an unknown candidate for task {task_id}"
                )
        return AssignOrDeferDecision(selected)

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
        if not isinstance(decision, AssignOrDeferDecision):
            raise TypeError("assign-or-defer requires AssignOrDeferDecision")
        checked = self.decision(typed_model, decision.selections)
        selected = tuple(item for item in checked.selections if item is not None)
        deferred = tuple(
            task_id
            for task_id, item in zip(
                typed_model.ordered_task_ids,
                checked.selections,
                strict=True,
            )
            if item is None
        )
        if not selected:
            return MaterializedSchedule((), (), (), deferred)

        inner_model = OneHotPlacementModel(
            problem_id=problem.problem_id,
            formulation_spec=self._one_hot.spec,
            metric_contract_id=problem.metric_contract_id,
            ordered_task_ids=tuple(item.task_id for item in selected),
            candidate_options=tuple((item,) for item in selected),
            objective_ids=typed_model.objective_ids,
            constraint_ids=typed_model.constraint_ids,
            referenced_metric_versions=typed_model.referenced_metric_versions,
            total_decisions=1,
        )
        inner = self._one_hot.materialize(
            problem,
            inner_model,
            OneHotPlacementDecision(selected),
            optimizer_id=optimizer_id,
        )
        return MaterializedSchedule(
            assignments=tuple(
                replace(item, reason="assign-or-defer placement decision")
                for item in inner.assignments
            ),
            node_reservations=inner.node_reservations,
            transfer_reservations=inner.transfer_reservations,
            deferred_task_ids=deferred,
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
        if tuple(item.objective_id for item in objectives) != typed_model.objective_ids:
            raise FormulationCompatibilityError(
                "assign-or-defer objective encoding does not cover the policy"
            )
        if tuple(item.constraint_id for item in constraints) != typed_model.constraint_ids:
            raise FormulationCompatibilityError(
                "assign-or-defer constraint encoding does not cover the policy"
            )
        return FormulationEvaluation(
            objective_evaluations=objectives,
            constraint_evaluations=constraints,
            objective_key=objective_key(problem.policy, objectives, constraints),
        )

    def validate_plan_domain(
        self,
        problem: SchedulingProblem,
        model: CompiledFormulation,
        plan: SchedulingPlan,
    ) -> None:
        typed_model = self._typed_model(model)
        self._validate_model(problem, typed_model)
        assignments = {item.task_id: item for item in plan.assignments}
        deferred = set(plan.deferred_task_ids)
        if set(assignments) & deferred:
            raise FormulationDomainError("a task cannot be assigned and deferred")
        if set(assignments) | deferred != set(typed_model.ordered_task_ids):
            raise FormulationDomainError(
                "assign-or-defer must cover every compiled task exactly once"
            )
        selections = []
        for task_id, options in zip(
            typed_model.ordered_task_ids,
            typed_model.candidate_options,
            strict=True,
        ):
            if task_id in deferred:
                selections.append(None)
                continue
            assignment = assignments[task_id]
            candidate = next(
                (item for item in options if item.node_id == assignment.target_node_id),
                None,
            )
            if candidate is None:
                raise FormulationDomainError(
                    f"assignment for task {task_id} is not a compiled candidate"
                )
            selections.append(candidate)
        canonical = self.materialize(
            problem,
            typed_model,
            self.decision(typed_model, tuple(selections)),
            optimizer_id=plan.optimizer_id,
        )
        actual_assignments = tuple(replace(item, reason="") for item in plan.assignments)
        expected_assignments = tuple(replace(item, reason="") for item in canonical.assignments)
        if actual_assignments != expected_assignments:
            raise FormulationDomainError(
                "assign-or-defer assignments do not match its materializer"
            )
        if plan.node_reservations != canonical.node_reservations:
            raise FormulationDomainError(
                "assign-or-defer node reservations do not match its materializer"
            )
        if plan.transfer_reservations != canonical.transfer_reservations:
            raise FormulationDomainError(
                "assign-or-defer transfer reservations do not match its materializer"
            )
        if plan.deferred_task_ids != canonical.deferred_task_ids:
            raise FormulationDomainError(
                "assign-or-defer deferred ids do not match its decision"
            )

    @staticmethod
    def _typed_model(model: CompiledFormulation) -> AssignOrDeferModel:
        if not isinstance(model, AssignOrDeferModel):
            raise TypeError("assign-or-defer requires AssignOrDeferModel")
        return model

    def _validate_model(
        self,
        problem: SchedulingProblem,
        model: AssignOrDeferModel,
    ) -> None:
        if model.problem_id != problem.problem_id:
            raise FormulationCompatibilityError(
                "assign-or-defer model belongs to a different problem"
            )
        if model.formulation_spec != self.spec:
            raise FormulationCompatibilityError(
                "assign-or-defer model uses a different formulation spec"
            )
        if model.metric_contract_id != problem.metric_contract_id:
            raise FormulationCompatibilityError(
                "assign-or-defer model uses a different metric contract"
            )


__all__ = [
    "ASSIGN_OR_DEFER_SPEC",
    "AssignOrDeferDecision",
    "AssignOrDeferFormulation",
    "AssignOrDeferModel",
]
