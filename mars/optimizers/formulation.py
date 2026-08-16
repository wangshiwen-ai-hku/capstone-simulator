"""Versioned scheduling-formulation contracts and registry.

The shared :class:`SchedulingProblem` says what must be solved.  A formulation
compiles that domain contract into a decision model, while an optimizer decides
how to search that model.  Keeping these axes separate lets one formulation be
used by exhaustive, MILP, or distributed optimizers without leaking a solver's
objects into the portable Problem contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
import enum
import hashlib
import json
import math
from time import perf_counter
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from ..domain.execution import Assignment
from ..domain.transfer import TransferReservation
from .base import (
    PlannedResourceReservation,
    SchedulingPlan,
    SchedulingProblem,
    SolveStatus,
)
from .policy import ConstraintEvaluation, ObjectiveEvaluation
from .state import OptimizerSolveState, SolveTraceContext


FormulationOption = str | int | float | bool


class FormulationError(ValueError):
    """A Problem cannot be represented or decoded by a formulation."""


class FormulationCompatibilityError(FormulationError):
    """An optimizer, policy, or metric contract is not supported exactly."""


class FormulationDomainError(FormulationError):
    """A returned Plan does not belong to its declared formulation domain."""


def formulation_failure_status(error: Exception) -> SolveStatus:
    """Classify formulation failures consistently at every solve entry point."""

    if isinstance(
        error,
        (FormulationCompatibilityError, FormulationDomainError),
    ):
        return SolveStatus.ERROR
    if isinstance(error, FormulationError):
        return SolveStatus.INFEASIBLE
    if isinstance(error, TimeoutError):
        return SolveStatus.TIME_LIMIT
    return SolveStatus.ERROR


@dataclass(frozen=True, slots=True)
class FormulationSpec:
    """Serializable identity and configuration for one formulation."""

    formulation_id: str
    formulation_version: str
    materializer_id: str
    materializer_version: str
    options: Mapping[str, FormulationOption] = field(default_factory=dict)
    schema_version: str = "mars.formulation-spec.v1"
    formulation_digest: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = (
            self.schema_version,
            self.formulation_id,
            self.formulation_version,
            self.materializer_id,
            self.materializer_version,
        )
        if any(not value.strip() for value in identifiers):
            raise ValueError("formulation identifiers must be non-blank")
        normalized: dict[str, FormulationOption] = {}
        for key, value in self.options.items():
            name = str(key).strip()
            if not name:
                raise ValueError("formulation option names must be non-blank")
            if name in normalized:
                raise ValueError(
                    "formulation option names must remain unique after "
                    "normalization"
                )
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError(
                    "formulation options must be scalar JSON values"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("formulation options must be finite")
            normalized[name] = value
        object.__setattr__(self, "options", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "formulation_digest",
            _contract_digest(
                {
                    "schema_version": self.schema_version,
                    "formulation_id": self.formulation_id,
                    "formulation_version": self.formulation_version,
                    "materializer_id": self.materializer_id,
                    "materializer_version": self.materializer_version,
                    "options": normalized,
                }
            ),
        )

    def __hash__(self) -> int:
        """Hash the immutable canonical contract rather than its mapping."""

        return hash((self.schema_version, self.formulation_digest))

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "formulation_id": self.formulation_id,
            "formulation_version": self.formulation_version,
            "materializer_id": self.materializer_id,
            "materializer_version": self.materializer_version,
            "options": dict(self.options),
            "formulation_digest": self.formulation_digest,
        }


@dataclass(frozen=True, slots=True)
class FormulationEvaluation:
    """Canonical policy evaluation of one decoded formulation decision."""

    objective_evaluations: tuple[ObjectiveEvaluation, ...]
    constraint_evaluations: tuple[ConstraintEvaluation, ...]
    objective_key: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_evaluations",
            tuple(self.objective_evaluations),
        )
        object.__setattr__(
            self,
            "constraint_evaluations",
            tuple(self.constraint_evaluations),
        )
        object.__setattr__(self, "objective_key", tuple(self.objective_key))

    @property
    def has_hard_violation(self) -> bool:
        return any(
            item.hard and not item.satisfied
            for item in self.constraint_evaluations
        )


@dataclass(frozen=True, slots=True)
class MaterializedSchedule:
    """Executable plan fields decoded from one formulation decision."""

    assignments: tuple[Assignment, ...]
    node_reservations: tuple[PlannedResourceReservation, ...]
    transfer_reservations: tuple[TransferReservation, ...]
    deferred_task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        object.__setattr__(
            self,
            "node_reservations",
            tuple(self.node_reservations),
        )
        object.__setattr__(
            self,
            "transfer_reservations",
            tuple(self.transfer_reservations),
        )
        object.__setattr__(
            self,
            "deferred_task_ids",
            tuple(self.deferred_task_ids),
        )


@runtime_checkable
class CompiledFormulation(Protocol):
    """Minimum identity carried by a process-local compiled model."""

    problem_id: str
    formulation_spec: FormulationSpec
    metric_contract_id: str


@runtime_checkable
class SchedulingFormulation(Protocol):
    """Replaceable compiler and decoder for a SchedulingProblem."""

    spec: FormulationSpec

    def compile(self, problem: SchedulingProblem) -> CompiledFormulation: ...

    def materialize(
        self,
        problem: SchedulingProblem,
        model: CompiledFormulation,
        decision: object,
        *,
        optimizer_id: str,
    ) -> MaterializedSchedule: ...

    def evaluate(
        self,
        problem: SchedulingProblem,
        model: CompiledFormulation,
        plan: SchedulingPlan,
    ) -> FormulationEvaluation: ...

    def validate_plan_domain(
        self,
        problem: SchedulingProblem,
        model: CompiledFormulation,
        plan: SchedulingPlan,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SchedulingSolveRequest:
    """Data-only identity for one Problem/formulation/optimizer combination."""

    problem: SchedulingProblem = field(repr=False)
    formulation_spec: FormulationSpec
    optimizer_id: str
    optimizer_version: str
    optimizer_config_digest: str = "default"
    schema_version: str = "mars.scheduling-solve-request.v1"
    solve_request_id: str = field(init=False)
    continuation_contract_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, SchedulingProblem):
            raise TypeError("solve request problem must be SchedulingProblem")
        if not isinstance(self.formulation_spec, FormulationSpec):
            raise TypeError(
                "solve request formulation_spec must be FormulationSpec"
            )
        identifiers = (
            self.schema_version,
            self.optimizer_id,
            self.optimizer_version,
            self.optimizer_config_digest,
        )
        if any(not value.strip() for value in identifiers):
            raise ValueError("solve request identifiers must be non-blank")
        digest = _contract_digest(
            {
                "schema_version": self.schema_version,
                "problem_id": self.problem.problem_id,
                "metric_contract_id": self.problem.metric_contract_id,
                "formulation": self.formulation_spec.as_dict(),
                "optimizer_id": self.optimizer_id,
                "optimizer_version": self.optimizer_version,
                "optimizer_config_digest": self.optimizer_config_digest,
            }
        )
        object.__setattr__(
            self,
            "solve_request_id",
            f"{self.problem.epoch.epoch_id}:solve-request:{digest}",
        )
        continuation_digest = _contract_digest(
            {
                "schema_version": "mars.optimizer-continuation-contract.v1",
                "problem_schema_version": self.problem.schema_version,
                "snapshot_schema_version": self.problem.snapshot.schema_version,
                "policy": self.problem.policy,
                "metric_contract_id": self.problem.metric_contract_id,
                "formulation": self.formulation_spec.as_dict(),
                "optimizer_id": self.optimizer_id,
                "optimizer_version": self.optimizer_version,
                "optimizer_config_digest": self.optimizer_config_digest,
                "deterministic": self.problem.solve_limits.deterministic,
                "random_seed": self.problem.solve_limits.random_seed,
            }
        )
        object.__setattr__(
            self,
            "continuation_contract_id",
            f"continuation-contract:{continuation_digest}",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "solve_request_id": self.solve_request_id,
            "continuation_contract_id": self.continuation_contract_id,
            "problem_id": self.problem.problem_id,
            "metric_contract_id": self.problem.metric_contract_id,
            "formulation": self.formulation_spec.as_dict(),
            "optimizer_id": self.optimizer_id,
            "optimizer_version": self.optimizer_version,
            "optimizer_config_digest": self.optimizer_config_digest,
        }


@dataclass(frozen=True, slots=True)
class PreparedSolve:
    """Process-local compiled solve request passed to a formulated optimizer."""

    request: SchedulingSolveRequest
    formulation: SchedulingFormulation = field(repr=False)
    model: CompiledFormulation = field(repr=False)
    compilation_elapsed_ms: float = 0.0
    solve_deadline_monotonic: float | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.request, SchedulingSolveRequest):
            raise TypeError("prepared solve requires SchedulingSolveRequest")
        if not isinstance(self.formulation, SchedulingFormulation):
            raise TypeError(
                "prepared solve formulation must implement SchedulingFormulation"
            )
        if not isinstance(self.model, CompiledFormulation):
            raise TypeError(
                "prepared solve model must implement CompiledFormulation"
            )
        if self.formulation.spec != self.request.formulation_spec:
            raise FormulationCompatibilityError(
                "compiled formulation spec does not match the solve request"
            )
        if self.model.formulation_spec != self.request.formulation_spec:
            raise FormulationCompatibilityError(
                "compiled model spec does not match the solve request"
            )
        if self.model.problem_id != self.request.problem.problem_id:
            raise FormulationCompatibilityError(
                "compiled model problem_id does not match the solve request"
            )
        if (
            self.model.metric_contract_id
            != self.request.problem.metric_contract_id
        ):
            raise FormulationCompatibilityError(
                "compiled model metric contract does not match the problem"
            )
        if (
            not math.isfinite(self.compilation_elapsed_ms)
            or self.compilation_elapsed_ms < 0.0
        ):
            raise ValueError(
                "formulation compilation elapsed time must be non-negative"
            )
        if self.solve_deadline_monotonic is not None and (
            not math.isfinite(self.solve_deadline_monotonic)
            or self.solve_deadline_monotonic < 0.0
        ):
            raise ValueError(
                "prepared solve deadline must be a non-negative finite time"
            )

    @property
    def problem(self) -> SchedulingProblem:
        return self.request.problem

    def time_limit_reached(
        self,
        *,
        now_monotonic: float,
        solve_started_monotonic: float,
    ) -> bool:
        """Check both the request-local and orchestration-wide deadlines."""

        return bool(
            (
                self.solve_deadline_monotonic is not None
                and now_monotonic >= self.solve_deadline_monotonic
            )
            or (
                now_monotonic - solve_started_monotonic
            )
            * 1000.0
            >= self.problem.solve_limits.solve_budget_ms
        )


@runtime_checkable
class FormulatedOptimizer(Protocol):
    """Optional optimizer extension for compiled formulation models."""

    optimizer_id: str
    optimizer_version: str
    optimizer_config_digest: str
    supported_formulation_ids: frozenset[str]

    def supports_formulation(self, spec: FormulationSpec) -> bool: ...

    def solve_formulated(self, prepared: PreparedSolve) -> SchedulingPlan: ...


@runtime_checkable
class StatefulFormulatedOptimizer(FormulatedOptimizer, Protocol):
    """Formulated optimizer extension with caller-owned trace/state."""

    optimizer_id: str
    optimizer_version: str
    optimizer_config_digest: str
    supported_formulation_ids: frozenset[str]

    def supports_formulation(self, spec: FormulationSpec) -> bool: ...

    def solve_formulated_with_state(
        self,
        prepared: PreparedSolve,
        state: OptimizerSolveState,
        *,
        context: SolveTraceContext | None = None,
    ) -> SchedulingPlan: ...


class FormulationRegistry:
    """Explicit registry for independently selectable formulations."""

    def __init__(self) -> None:
        self._formulations: dict[str, SchedulingFormulation] = {}

    def register(
        self,
        formulation: SchedulingFormulation,
        *,
        aliases: tuple[str, ...] = (),
        replace: bool = False,
    ) -> None:
        if not isinstance(formulation, SchedulingFormulation):
            raise TypeError(
                "formulation must implement SchedulingFormulation"
            )
        if not isinstance(formulation.spec, FormulationSpec):
            raise TypeError("formulation spec must be a FormulationSpec")
        names = (formulation.spec.formulation_id, *aliases)
        if any(not name.strip() for name in names):
            raise ValueError("formulation ids and aliases must be non-blank")
        if len(names) != len(set(names)):
            raise ValueError("formulation ids and aliases must be unique")
        collisions = [name for name in names if name in self._formulations]
        if collisions and not replace:
            raise ValueError(
                f"formulation ids already registered: {sorted(collisions)}"
            )
        if replace:
            for name, current in tuple(self._formulations.items()):
                if (
                    current.spec.formulation_id
                    == formulation.spec.formulation_id
                ):
                    self._formulations[name] = formulation
        for name in names:
            self._formulations[name] = formulation

    def resolve(
        self,
        formulation: str | SchedulingFormulation,
    ) -> SchedulingFormulation:
        if isinstance(formulation, str):
            try:
                return self._formulations[formulation]
            except KeyError as exc:
                raise KeyError(
                    f"unknown formulation {formulation!r}; available="
                    f"{sorted(self._formulations)}"
                ) from exc
        if not isinstance(formulation, SchedulingFormulation):
            raise TypeError(
                "formulation must be a registered id or SchedulingFormulation"
            )
        if not isinstance(formulation.spec, FormulationSpec):
            raise TypeError("formulation spec must be a FormulationSpec")
        return formulation

    def extend(
        self,
        other: FormulationRegistry,
        *,
        replace: bool = False,
    ) -> None:
        collisions = set(self._formulations) & set(other._formulations)
        if collisions and not replace:
            raise ValueError(
                "formulation registries overlap: "
                f"{sorted(collisions)}"
            )
        self._formulations.update(other._formulations)
        if replace:
            # Keep every pre-existing alias bound to the newly selected
            # canonical implementation. Without this normalization, extending
            # a registry with v2 could leave an old alias resolving to v1.
            canonical = {
                name: formulation
                for name, formulation in self._formulations.items()
                if name == formulation.spec.formulation_id
            }
            for name, formulation in tuple(self._formulations.items()):
                replacement = canonical.get(
                    formulation.spec.formulation_id
                )
                if replacement is not None:
                    self._formulations[name] = replacement

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._formulations))

    def specs(self) -> tuple[FormulationSpec, ...]:
        canonical = {
            name: formulation.spec
            for name, formulation in self._formulations.items()
            if name == formulation.spec.formulation_id
        }
        return tuple(canonical[key] for key in sorted(canonical))


def prepare_solve(
    problem: SchedulingProblem,
    optimizer: object,
    formulation: SchedulingFormulation,
) -> PreparedSolve:
    """Compile a Problem and bind deterministic solve-request identity."""

    request = build_solve_request(problem, optimizer, formulation)
    return compile_solve_request(request, formulation)


def build_solve_request(
    problem: SchedulingProblem,
    optimizer: object,
    formulation: SchedulingFormulation,
) -> SchedulingSolveRequest:
    """Bind deterministic request identity before model compilation."""

    optimizer_id = str(getattr(optimizer, "optimizer_id", ""))
    optimizer_version = str(getattr(optimizer, "optimizer_version", ""))
    optimizer_config_digest = str(
        getattr(optimizer, "optimizer_config_digest", "")
    )
    return SchedulingSolveRequest(
        problem=problem,
        formulation_spec=formulation.spec,
        optimizer_id=optimizer_id,
        optimizer_version=optimizer_version,
        optimizer_config_digest=optimizer_config_digest,
    )


def compile_solve_request(
    request: SchedulingSolveRequest,
    formulation: SchedulingFormulation,
    *,
    solve_deadline_monotonic: float | None = None,
) -> PreparedSolve:
    """Compile an identified request into its process-local model."""

    if request.formulation_spec != formulation.spec:
        raise FormulationCompatibilityError(
            "solve request does not match the selected formulation"
        )
    compile_started = perf_counter()
    model = formulation.compile(request.problem)
    compile_finished = perf_counter()
    compilation_elapsed_ms = (compile_finished - compile_started) * 1000.0
    if (
        solve_deadline_monotonic is not None
        and compile_finished >= solve_deadline_monotonic
    ):
        raise TimeoutError(
            "shared scheduling-epoch solve budget expired during formulation "
            "compilation"
        )
    return PreparedSolve(
        request=request,
        formulation=formulation,
        model=model,
        compilation_elapsed_ms=compilation_elapsed_ms,
        solve_deadline_monotonic=solve_deadline_monotonic,
    )


def built_in_formulation_registry() -> FormulationRegistry:
    from .formulations.assign_or_defer import AssignOrDeferFormulation
    from .formulations.one_hot import OneHotPlacementFormulation

    registry = FormulationRegistry()
    registry.register(OneHotPlacementFormulation())
    registry.register(AssignOrDeferFormulation())
    return registry


def _contract_digest(value: object) -> str:
    encoded = json.dumps(
        _canonical_contract_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _canonical_contract_value(value: object) -> object:
    if isinstance(value, enum.Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_contract_value(
                getattr(value, item.name)
            )
            for item in fields(value)
            if item.init
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_contract_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_contract_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_contract_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
    return value


__all__ = [
    "CompiledFormulation",
    "FormulatedOptimizer",
    "FormulationCompatibilityError",
    "FormulationDomainError",
    "FormulationError",
    "FormulationEvaluation",
    "FormulationRegistry",
    "FormulationSpec",
    "MaterializedSchedule",
    "PreparedSolve",
    "SchedulingFormulation",
    "SchedulingSolveRequest",
    "StatefulFormulatedOptimizer",
    "build_solve_request",
    "built_in_formulation_registry",
    "compile_solve_request",
    "formulation_failure_status",
    "prepare_solve",
]
