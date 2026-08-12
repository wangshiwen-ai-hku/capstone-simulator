"""Externally owned state and trace contracts for optimizer solves."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import enum
import math
from types import MappingProxyType
from typing import Generic, Mapping, Protocol, TypeVar, runtime_checkable

from .base import SchedulingPlan, SchedulingProblem, SolveStatus


ContinuationPayloadT_co = TypeVar(
    "ContinuationPayloadT_co",
    covariant=True,
)


class SolveTracePhase(str, enum.Enum):
    """Lifecycle phases recorded for one optimizer invocation."""

    STARTED = "started"
    ITERATION = "iteration"
    INCUMBENT = "incumbent"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SolveTraceContext:
    """Stable correlation data shared by entries from one solve invocation."""

    solve_id: str
    frame_index: int
    problem_id: str
    snapshot_id: str
    epoch_id: str
    policy_id: str
    policy_version: str
    optimizer_id: str
    optimizer_version: str
    work_unit: str
    solve_budget_ms: float
    max_iterations: int

    def __post_init__(self) -> None:
        identifiers = (
            self.solve_id,
            self.problem_id,
            self.snapshot_id,
            self.epoch_id,
            self.policy_id,
            self.policy_version,
            self.optimizer_id,
            self.work_unit,
        )
        if any(not value.strip() for value in identifiers):
            raise ValueError("solve trace identifiers must be non-blank")
        if self.frame_index < 1 or self.max_iterations < 0:
            raise ValueError("solve trace counters are out of range")
        if (
            not math.isfinite(self.solve_budget_ms)
            or self.solve_budget_ms <= 0.0
        ):
            raise ValueError("solve trace budget must be positive")


@dataclass(frozen=True)
class SolveTraceEntry:
    """One immutable, serializable observation from a solve lifecycle."""

    sequence: int
    context: SolveTraceContext
    phase: SolveTracePhase
    iteration: int = 0
    elapsed_ms: float = 0.0
    solve_status: SolveStatus | None = None
    termination_reason: str = ""
    has_incumbent: bool = False
    evaluated_work_units: int = 0
    total_work_units: int | None = None
    objective_key: tuple[float, ...] = ()
    objective_components: Mapping[str, float] = field(default_factory=dict)
    selected_targets: Mapping[str, str] = field(default_factory=dict)
    details: Mapping[str, float | int | str | bool] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("solve trace sequence must be positive")
        if not isinstance(self.context, SolveTraceContext):
            raise TypeError("solve trace context must be SolveTraceContext")
        if not isinstance(self.phase, SolveTracePhase):
            raise TypeError("solve trace phase must be SolveTracePhase")
        if self.solve_status is not None and not isinstance(
            self.solve_status,
            SolveStatus,
        ):
            raise TypeError("solve trace status must be SolveStatus")
        if self.iteration < 0 or self.evaluated_work_units < 0:
            raise ValueError("solve trace counters must be non-negative")
        if self.total_work_units is not None and self.total_work_units < 0:
            raise ValueError("total_work_units must be non-negative")
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0.0:
            raise ValueError("solve trace elapsed_ms must be non-negative")
        object.__setattr__(self, "objective_key", tuple(self.objective_key))
        object.__setattr__(
            self,
            "objective_components",
            MappingProxyType(dict(self.objective_components)),
        )
        object.__setattr__(
            self,
            "selected_targets",
            MappingProxyType(dict(self.selected_targets)),
        )
        object.__setattr__(
            self,
            "details",
            MappingProxyType(dict(self.details)),
        )

    def as_dict(self) -> dict[str, object]:
        """Return the stable wire/log projection of this entry."""

        return {
            "sequence": self.sequence,
            "solve_id": self.context.solve_id,
            "frame_index": self.context.frame_index,
            "problem_id": self.context.problem_id,
            "snapshot_id": self.context.snapshot_id,
            "epoch_id": self.context.epoch_id,
            "policy_id": self.context.policy_id,
            "policy_version": self.context.policy_version,
            "optimizer_id": self.context.optimizer_id,
            "optimizer_version": self.context.optimizer_version,
            "work_unit": self.context.work_unit,
            "solve_budget_ms": self.context.solve_budget_ms,
            "max_iterations": self.context.max_iterations,
            "phase": self.phase.value,
            "iteration": self.iteration,
            "elapsed_ms": self.elapsed_ms,
            "solve_status": (
                self.solve_status.value if self.solve_status is not None else ""
            ),
            "termination_reason": self.termination_reason,
            "has_incumbent": self.has_incumbent,
            "evaluated_work_units": self.evaluated_work_units,
            "total_work_units": self.total_work_units,
            "objective_key": list(self.objective_key),
            "objective_components": dict(self.objective_components),
            "selected_targets": dict(self.selected_targets),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class OptimizerContinuation(Generic[ContinuationPayloadT_co]):
    """Typed wrapper for optimizer-specific cross-frame warm-start data.

    Payloads should be immutable, versioned dataclasses or JSON-like values so
    archived continuation snapshots remain reproducible.
    """

    optimizer_id: str
    schema_version: str
    updated_problem_id: str
    payload: ContinuationPayloadT_co = field(repr=False)
    iteration: int = 0
    objective_key: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.optimizer_id.strip() or not self.schema_version.strip():
            raise ValueError(
                "optimizer continuation ids and schema version must be non-blank"
            )
        if not self.updated_problem_id.strip():
            raise ValueError("continuation updated_problem_id must be non-blank")
        if self.iteration < 0:
            raise ValueError("continuation iteration must be non-negative")
        object.__setattr__(self, "objective_key", tuple(self.objective_key))
        if any(not math.isfinite(value) for value in self.objective_key):
            raise ValueError("continuation objective_key must be finite")

    def as_dict(self) -> dict[str, object]:
        return {
            "optimizer_id": self.optimizer_id,
            "schema_version": self.schema_version,
            "updated_problem_id": self.updated_problem_id,
            "iteration": self.iteration,
            "objective_key": list(self.objective_key),
            "payload": _serializable_state(self.payload),
        }


@dataclass
class OptimizerSolveState:
    """Caller-owned trace and continuation state spanning frames/iterations.

    Optimizers append immutable trace entries and may store a typed continuation
    payload for a future frame. The coordinator owns this mutable container; an
    optimizer instance therefore does not retain workflow history between calls.
    """

    session_id: str
    schema_version: str = "mars.optimizer-solve-state.v1"
    trace_entries: list[SolveTraceEntry] = field(default_factory=list)
    continuations: dict[str, OptimizerContinuation[object]] = field(
        default_factory=dict
    )
    continuation_history: list[OptimizerContinuation[object]] = field(
        default_factory=list
    )
    _frame_by_problem: dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _latest_context: dict[tuple[str, str], SolveTraceContext] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _next_solve_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.schema_version.strip():
            raise ValueError(
                "solve state session_id and schema_version must be non-blank"
            )
        restored = list(self.trace_entries)
        self.trace_entries = []
        for entry in restored:
            if not isinstance(entry, SolveTraceEntry):
                raise TypeError("trace_entries must contain SolveTraceEntry")
            self.trace_entries.append(entry)
            self._frame_by_problem.setdefault(
                entry.context.problem_id,
                entry.context.frame_index,
            )
            self._latest_context[
                (entry.context.problem_id, entry.context.optimizer_id)
            ] = entry.context
        self._next_solve_index = len(
            {entry.context.solve_id for entry in self.trace_entries}
        )
        if any(
            not isinstance(continuation, OptimizerContinuation)
            or key != continuation.optimizer_id
            for key, continuation in self.continuations.items()
        ):
            raise TypeError(
                "continuations must map optimizer ids to matching "
                "OptimizerContinuation values"
            )
        if any(
            not isinstance(continuation, OptimizerContinuation)
            for continuation in self.continuation_history
        ):
            raise TypeError(
                "continuation_history must contain OptimizerContinuation"
            )

    @property
    def entries(self) -> tuple[SolveTraceEntry, ...]:
        return tuple(self.trace_entries)

    def begin(
        self,
        problem: SchedulingProblem,
        *,
        optimizer_id: str,
        optimizer_version: str = "",
        work_unit: str = "iteration",
    ) -> SolveTraceContext:
        """Start one solve attempt and record its initial trace entry."""

        if not optimizer_id.strip() or not work_unit.strip():
            raise ValueError("optimizer_id and work_unit must be non-blank")
        frame_index = self._frame_by_problem.setdefault(
            problem.problem_id,
            len(self._frame_by_problem) + 1,
        )
        self._next_solve_index += 1
        context = SolveTraceContext(
            solve_id=f"{self.session_id}:solve:{self._next_solve_index}",
            frame_index=frame_index,
            problem_id=problem.problem_id,
            snapshot_id=problem.snapshot.snapshot_id,
            epoch_id=problem.epoch.epoch_id,
            policy_id=problem.policy.policy_id,
            policy_version=problem.policy.version,
            optimizer_id=optimizer_id,
            optimizer_version=optimizer_version,
            work_unit=work_unit,
            solve_budget_ms=problem.solve_limits.solve_budget_ms,
            max_iterations=problem.solve_limits.max_iterations,
        )
        self._latest_context[(problem.problem_id, optimizer_id)] = context
        self.record(context, SolveTracePhase.STARTED)
        return context

    def record(
        self,
        context: SolveTraceContext,
        phase: SolveTracePhase,
        **values: object,
    ) -> SolveTraceEntry:
        """Append an immutable observation to this session."""

        if not isinstance(context, SolveTraceContext):
            raise TypeError("solve trace context must be SolveTraceContext")
        if not isinstance(phase, SolveTracePhase):
            raise TypeError("solve trace phase must be SolveTracePhase")

        entry = SolveTraceEntry(
            sequence=len(self.trace_entries) + 1,
            context=context,
            phase=phase,
            **values,
        )
        self.trace_entries.append(entry)
        self._latest_context[(context.problem_id, context.optimizer_id)] = context
        return entry

    def latest_context(
        self,
        problem_id: str,
        optimizer_id: str,
    ) -> SolveTraceContext | None:
        """Return the latest invocation context for one problem and solver."""

        return self._latest_context.get((problem_id, optimizer_id))

    def set_continuation(
        self,
        continuation: OptimizerContinuation[object],
    ) -> None:
        """Replace one optimizer's typed warm-start state."""

        _serializable_state(continuation.payload)
        self.continuations[continuation.optimizer_id] = continuation
        self.continuation_history.append(continuation)

    def continuation_for(
        self,
        optimizer_id: str,
    ) -> OptimizerContinuation[object] | None:
        return self.continuations.get(optimizer_id)

    def terminal_entries(
        self,
        optimizer_id: str | None = None,
    ) -> tuple[SolveTraceEntry, ...]:
        terminal = {
            SolveTracePhase.COMPLETED,
            SolveTracePhase.FAILED,
            SolveTracePhase.REJECTED,
            SolveTracePhase.VALIDATED,
        }
        return tuple(
            entry
            for entry in self.trace_entries
            if entry.phase in terminal
            and (
                optimizer_id is None
                or entry.context.optimizer_id == optimizer_id
            )
        )

    def invocation_summaries(
        self,
        optimizer_id: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Return one compatibility-friendly row per solve invocation."""

        grouped: dict[str, list[SolveTraceEntry]] = {}
        for entry in self.trace_entries:
            if (
                optimizer_id is not None
                and entry.context.optimizer_id != optimizer_id
            ):
                continue
            grouped.setdefault(entry.context.solve_id, []).append(entry)

        summaries = []
        for entries in grouped.values():
            context = entries[0].context
            latest = entries[-1]
            terminal = next(
                (
                    entry
                    for entry in reversed(entries)
                    if entry.phase
                    in {
                        SolveTracePhase.FAILED,
                        SolveTracePhase.REJECTED,
                        SolveTracePhase.VALIDATED,
                        SolveTracePhase.COMPLETED,
                    }
                ),
                latest,
            )
            summary = {
                **dict(terminal.details),
                "solve_id": context.solve_id,
                "frame_index": context.frame_index,
                "problem_id": context.problem_id,
                "snapshot_id": context.snapshot_id,
                "epoch_id": context.epoch_id,
                "policy_id": context.policy_id,
                "policy_version": context.policy_version,
                "optimizer_id": context.optimizer_id,
                "optimizer_version": context.optimizer_version,
                "work_unit": context.work_unit,
                "terminal_phase": terminal.phase.value,
                "ready_task_count": int(
                    terminal.details.get("ready_task_count", 0)
                ),
                "solve_status": (
                    terminal.solve_status.value
                    if terminal.solve_status is not None
                    else ""
                ),
                "termination_reason": terminal.termination_reason,
                "has_incumbent": terminal.has_incumbent,
                "solve_elapsed_ms": terminal.elapsed_ms,
                "iteration_count": terminal.iteration,
                "evaluated_work_units": terminal.evaluated_work_units,
                "total_work_units": terminal.total_work_units,
                "solve_budget_ms": context.solve_budget_ms,
                "max_iterations": context.max_iterations,
                "objective_value": (
                    terminal.objective_key[0]
                    if terminal.objective_key
                    else 0.0
                ),
                "objective_key": list(terminal.objective_key),
                "objective_components": dict(
                    terminal.objective_components
                ),
                "assignments": dict(terminal.selected_targets),
            }
            if context.work_unit == "placement_combination":
                summary.update(
                    {
                        "enumerated_combinations": (
                            terminal.evaluated_work_units
                        ),
                        "total_combinations": terminal.total_work_units,
                        "placement_search_exhaustive": bool(
                            terminal.solve_status is SolveStatus.OPTIMAL
                        ),
                    }
                )
            summaries.append(summary)
        return tuple(summaries)

    def as_dict(self) -> dict[str, object]:
        """Serialize the complete auditable trace and continuation snapshots."""

        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "frame_count": len(self._frame_by_problem),
            "solve_count": self._next_solve_index,
            "trace_entries": [entry.as_dict() for entry in self.trace_entries],
            "invocation_summaries": list(self.invocation_summaries()),
            "continuations": {
                optimizer_id: continuation.as_dict()
                for optimizer_id, continuation in self.continuations.items()
            },
            "continuation_history": [
                continuation.as_dict()
                for continuation in self.continuation_history
            ],
        }


@runtime_checkable
class StatefulOptimizer(Protocol):
    """Optional optimizer extension for trace and cross-frame state access."""

    optimizer_id: str

    def solve_with_state(
        self,
        problem: SchedulingProblem,
        state: OptimizerSolveState,
        *,
        context: SolveTraceContext | None = None,
    ) -> SchedulingPlan: ...


def _serializable_state(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _serializable_state(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _serializable_state(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_serializable_state(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("optimizer state cannot contain non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "optimizer state payload must be a dataclass or JSON-like value"
    )


__all__ = [
    "OptimizerContinuation",
    "OptimizerSolveState",
    "SolveTraceContext",
    "SolveTraceEntry",
    "SolveTracePhase",
    "StatefulOptimizer",
]
