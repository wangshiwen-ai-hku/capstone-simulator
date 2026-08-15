"""Externally owned state and trace contracts for optimizer solves."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import enum
import math
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Generic,
    Mapping,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from .base import SchedulingPlan, SchedulingProblem, SolveStatus

if TYPE_CHECKING:
    from .formulation import SchedulingSolveRequest


ContinuationPayloadT_co = TypeVar(
    "ContinuationPayloadT_co",
    covariant=True,
)
ContinuationKey = tuple[str, ...]


class SolveTracePhase(str, enum.Enum):
    """Lifecycle phases recorded for one optimizer invocation."""

    STARTED = "started"
    ITERATION = "iteration"
    INCUMBENT = "incumbent"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    FALLBACK = "fallback"


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
    optimizer_config_digest: str = ""
    solve_request_id: str = ""
    continuation_contract_id: str = ""
    metric_contract_id: str = ""
    formulation_id: str = ""
    formulation_version: str = ""
    formulation_digest: str = ""

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
        formulation_identity = (
            self.solve_request_id,
            self.continuation_contract_id,
            self.metric_contract_id,
            self.formulation_id,
            self.formulation_version,
            self.formulation_digest,
            self.optimizer_config_digest,
        )
        if any(formulation_identity) and not all(formulation_identity):
            raise ValueError(
                "solve trace formulation identity must be provided together"
            )


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
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 1
        ):
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
        if (
            not isinstance(self.iteration, int)
            or isinstance(self.iteration, bool)
            or not isinstance(self.evaluated_work_units, int)
            or isinstance(self.evaluated_work_units, bool)
            or self.iteration < 0
            or self.evaluated_work_units < 0
        ):
            raise ValueError("solve trace counters must be non-negative")
        if self.total_work_units is not None and (
            not isinstance(self.total_work_units, int)
            or isinstance(self.total_work_units, bool)
            or self.total_work_units < 0
        ):
            raise ValueError("total_work_units must be non-negative")
        if (
            not isinstance(self.elapsed_ms, (int, float))
            or isinstance(self.elapsed_ms, bool)
            or not math.isfinite(self.elapsed_ms)
            or self.elapsed_ms < 0.0
        ):
            raise ValueError("solve trace elapsed_ms must be non-negative")
        if not isinstance(self.termination_reason, str):
            raise TypeError("solve trace termination_reason must be a string")
        if not isinstance(self.has_incumbent, bool):
            raise TypeError("solve trace has_incumbent must be bool")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in self.objective_key
        ):
            raise ValueError("solve trace objective_key must be finite")
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for key, value in self.objective_components.items()
        ):
            raise ValueError(
                "solve trace objective components must be named finite numbers"
            )
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            for key, value in self.selected_targets.items()
        ):
            raise ValueError(
                "solve trace selected targets must map names to strings"
            )
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, (float, int, str, bool))
            or isinstance(value, float)
            and not math.isfinite(value)
            for key, value in self.details.items()
        ):
            raise ValueError(
                "solve trace details must contain named finite scalars"
            )
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
            "optimizer_config_digest": self.context.optimizer_config_digest,
            "solve_request_id": self.context.solve_request_id,
            "continuation_contract_id": (
                self.context.continuation_contract_id
            ),
            "metric_contract_id": self.context.metric_contract_id,
            "formulation_id": self.context.formulation_id,
            "formulation_version": self.context.formulation_version,
            "formulation_digest": self.context.formulation_digest,
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
    optimizer_version: str = ""
    optimizer_config_digest: str = ""
    source_solve_request_id: str = ""
    continuation_contract_id: str = ""
    metric_contract_id: str = ""
    formulation_id: str = ""
    formulation_version: str = ""
    formulation_digest: str = ""

    def __post_init__(self) -> None:
        if not self.optimizer_id.strip() or not self.schema_version.strip():
            raise ValueError(
                "optimizer continuation ids and schema version must be non-blank"
            )
        if not self.updated_problem_id.strip():
            raise ValueError("continuation updated_problem_id must be non-blank")
        if self.iteration < 0:
            raise ValueError("continuation iteration must be non-negative")
        formulation_identity = (
            self.optimizer_version,
            self.optimizer_config_digest,
            self.source_solve_request_id,
            self.continuation_contract_id,
            self.metric_contract_id,
            self.formulation_id,
            self.formulation_version,
            self.formulation_digest,
        )
        if any(formulation_identity) and not all(formulation_identity):
            raise ValueError(
                "continuation formulation identity must be provided together"
            )
        object.__setattr__(self, "objective_key", tuple(self.objective_key))
        if any(not math.isfinite(value) for value in self.objective_key):
            raise ValueError("continuation objective_key must be finite")

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "optimizer_id": self.optimizer_id,
            "schema_version": self.schema_version,
            "updated_problem_id": self.updated_problem_id,
            "iteration": self.iteration,
            "objective_key": list(self.objective_key),
            "payload": _serializable_state(self.payload),
        }
        # Preserve the v1 projection for legacy optimizer continuations while
        # binding every formulated continuation to its complete solve contract.
        if self.formulation_id:
            data.update(
                {
                    "optimizer_version": self.optimizer_version,
                    "optimizer_config_digest": self.optimizer_config_digest,
                    "source_solve_request_id": self.source_solve_request_id,
                    "continuation_contract_id": self.continuation_contract_id,
                    "metric_contract_id": self.metric_contract_id,
                    "formulation_id": self.formulation_id,
                    "formulation_version": self.formulation_version,
                    "formulation_digest": self.formulation_digest,
                }
            )
        return data


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
    continuations: dict[ContinuationKey, OptimizerContinuation[object]] = field(
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
    _latest_context: dict[tuple[str, str, str], SolveTraceContext] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _next_frame_index: int = field(default=0, init=False, repr=False)
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
                (
                    entry.context.problem_id,
                    entry.context.optimizer_id,
                    entry.context.solve_request_id,
                )
            ] = entry.context
        self._next_frame_index = max(
            (
                entry.context.frame_index
                for entry in self.trace_entries
            ),
            default=0,
        )
        solve_ids = {
            entry.context.solve_id for entry in self.trace_entries
        }
        prefix = f"{self.session_id}:solve:"
        restored_indices = [
            int(solve_id.removeprefix(prefix))
            for solve_id in solve_ids
            if solve_id.startswith(prefix)
            and solve_id.removeprefix(prefix).isdigit()
        ]
        self._next_solve_index = max(
            (len(solve_ids), *restored_indices),
            default=0,
        )
        normalized_continuations = {}
        for key, continuation in self.continuations.items():
            if not isinstance(continuation, OptimizerContinuation):
                raise TypeError(
                    "continuations must contain OptimizerContinuation values"
                )
            expected = _continuation_key(continuation)
            # Accept the legacy constructor shape while normalizing all
            # in-memory keys to collision-free tagged tuples.
            if key != expected and not (
                isinstance(key, str)
                and not continuation.formulation_id
                and key == continuation.optimizer_id
            ):
                raise TypeError(
                    "continuation key does not match its solve contract"
                )
            if expected in normalized_continuations:
                raise ValueError(
                    "continuation keys normalize to the same solve contract"
                )
            normalized_continuations[expected] = _snapshot_continuation(
                continuation
            )
        self.continuations = normalized_continuations
        normalized_history = []
        for continuation in self.continuation_history:
            if not isinstance(continuation, OptimizerContinuation):
                raise TypeError(
                    "continuation_history must contain OptimizerContinuation"
                )
            normalized_history.append(_snapshot_continuation(continuation))
        self.continuation_history = normalized_history

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
        solve_request: "SchedulingSolveRequest | None" = None,
    ) -> SolveTraceContext:
        """Start one solve attempt and record its initial trace entry."""

        if not optimizer_id.strip() or not work_unit.strip():
            raise ValueError("optimizer_id and work_unit must be non-blank")
        if solve_request is not None and (
            solve_request.problem.problem_id != problem.problem_id
            or solve_request.problem != problem
            or solve_request.optimizer_id != optimizer_id
            or solve_request.optimizer_version != optimizer_version
        ):
            raise ValueError(
                "solve request identity does not match the trace context"
            )
        frame_index = self._frame_by_problem.get(problem.problem_id)
        if frame_index is None:
            self._next_frame_index += 1
            frame_index = self._next_frame_index
            self._frame_by_problem[problem.problem_id] = frame_index
        existing_solve_ids = {
            entry.context.solve_id for entry in self.trace_entries
        }
        while True:
            self._next_solve_index += 1
            solve_id = (
                f"{self.session_id}:solve:{self._next_solve_index}"
            )
            if solve_id not in existing_solve_ids:
                break
        context = SolveTraceContext(
            solve_id=solve_id,
            frame_index=frame_index,
            problem_id=problem.problem_id,
            snapshot_id=problem.snapshot.snapshot_id,
            epoch_id=problem.epoch.epoch_id,
            policy_id=problem.policy.policy_id,
            policy_version=problem.policy.version,
            optimizer_id=optimizer_id,
            optimizer_version=optimizer_version,
            optimizer_config_digest=(
                solve_request.optimizer_config_digest
                if solve_request is not None
                else ""
            ),
            work_unit=work_unit,
            solve_budget_ms=problem.solve_limits.solve_budget_ms,
            max_iterations=problem.solve_limits.max_iterations,
            solve_request_id=(
                solve_request.solve_request_id
                if solve_request is not None
                else ""
            ),
            continuation_contract_id=(
                solve_request.continuation_contract_id
                if solve_request is not None
                else ""
            ),
            metric_contract_id=(
                problem.metric_contract_id
                if solve_request is not None
                else ""
            ),
            formulation_id=(
                solve_request.formulation_spec.formulation_id
                if solve_request is not None
                else ""
            ),
            formulation_version=(
                solve_request.formulation_spec.formulation_version
                if solve_request is not None
                else ""
            ),
            formulation_digest=(
                solve_request.formulation_spec.formulation_digest
                if solve_request is not None
                else ""
            ),
        )
        self._latest_context[
            (problem.problem_id, optimizer_id, context.solve_request_id)
        ] = context
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
        self._latest_context[
            (
                context.problem_id,
                context.optimizer_id,
                context.solve_request_id,
            )
        ] = context
        return entry

    def latest_context(
        self,
        problem_id: str,
        optimizer_id: str,
        solve_request_id: str = "",
    ) -> SolveTraceContext | None:
        """Return the latest invocation context for one problem and solver."""

        exact = self._latest_context.get(
            (problem_id, optimizer_id, solve_request_id)
        )
        if exact is not None or solve_request_id:
            return exact
        return next(
            (
                context
                for (candidate_problem, candidate_optimizer, _), context
                in reversed(tuple(self._latest_context.items()))
                if candidate_problem == problem_id
                and candidate_optimizer == optimizer_id
            ),
            None,
        )

    def set_continuation(
        self,
        continuation: OptimizerContinuation[object],
    ) -> None:
        """Replace one optimizer's typed warm-start state."""

        if not isinstance(continuation, OptimizerContinuation):
            raise TypeError(
                "continuation must be an OptimizerContinuation"
            )
        continuation = _snapshot_continuation(continuation)
        self.continuations[
            _continuation_key(continuation)
        ] = continuation
        self.continuation_history.append(continuation)

    def continuation_for(
        self,
        optimizer_id: str,
        *,
        optimizer_version: str = "",
        optimizer_config_digest: str = "",
        continuation_contract_id: str = "",
        formulation_id: str = "",
        formulation_version: str = "",
        formulation_digest: str = "",
        metric_contract_id: str = "",
    ) -> OptimizerContinuation[object] | None:
        if not formulation_id:
            return self.continuations.get(("legacy", optimizer_id))
        return self.continuations.get(
            _continuation_key_parts(
                optimizer_id,
                optimizer_version,
                optimizer_config_digest,
                continuation_contract_id,
                formulation_id,
                formulation_version,
                formulation_digest,
                metric_contract_id,
            )
        )

    def continuation_for_request(
        self,
        request: "SchedulingSolveRequest",
    ) -> OptimizerContinuation[object] | None:
        """Return only warm state compatible with the requested solve stack."""

        from .formulation import SchedulingSolveRequest

        if not isinstance(request, SchedulingSolveRequest):
            raise TypeError(
                "continuation lookup requires SchedulingSolveRequest"
            )
        spec = request.formulation_spec
        return self.continuation_for(
            request.optimizer_id,
            optimizer_version=request.optimizer_version,
            optimizer_config_digest=request.optimizer_config_digest,
            continuation_contract_id=request.continuation_contract_id,
            formulation_id=spec.formulation_id,
            formulation_version=spec.formulation_version,
            formulation_digest=spec.formulation_digest,
            metric_contract_id=request.problem.metric_contract_id,
        )

    def terminal_entries(
        self,
        optimizer_id: str | None = None,
    ) -> tuple[SolveTraceEntry, ...]:
        terminal = {
            SolveTracePhase.COMPLETED,
            SolveTracePhase.FAILED,
            SolveTracePhase.REJECTED,
            SolveTracePhase.FALLBACK,
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
                        SolveTracePhase.FALLBACK,
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
                "optimizer_config_digest": context.optimizer_config_digest,
                "solve_request_id": context.solve_request_id,
                "continuation_contract_id": (
                    context.continuation_contract_id
                ),
                "metric_contract_id": context.metric_contract_id,
                "formulation_id": context.formulation_id,
                "formulation_version": context.formulation_version,
                "formulation_digest": context.formulation_digest,
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
                formulation_exhausted = bool(
                    terminal.details.get("formulation_exhausted", False)
                )
                summary.update(
                    {
                        "enumerated_combinations": (
                            terminal.evaluated_work_units
                        ),
                        "total_combinations": terminal.total_work_units,
                        "formulation_exhausted": formulation_exhausted,
                        "placement_search_exhaustive": formulation_exhausted,
                    }
                )
            summaries.append(summary)
        return tuple(summaries)

    def as_dict(self) -> dict[str, object]:
        """Serialize the complete auditable trace and continuation snapshots."""

        serialized_continuations = {
            _serialized_continuation_key(key): continuation.as_dict()
            for key, continuation in self.continuations.items()
        }
        if len(serialized_continuations) != len(self.continuations):
            raise ValueError(
                "optimizer continuation archive keys must be unique"
            )

        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "frame_count": len(self._frame_by_problem),
            "solve_count": len(
                {entry.context.solve_id for entry in self.trace_entries}
            ),
            "trace_entries": [entry.as_dict() for entry in self.trace_entries],
            "invocation_summaries": list(self.invocation_summaries()),
            "continuations": serialized_continuations,
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


def _continuation_key(
    continuation: OptimizerContinuation[object],
) -> ContinuationKey:
    if not continuation.formulation_id:
        return ("legacy", continuation.optimizer_id)
    return _continuation_key_parts(
        continuation.optimizer_id,
        continuation.optimizer_version,
        continuation.optimizer_config_digest,
        continuation.continuation_contract_id,
        continuation.formulation_id,
        continuation.formulation_version,
        continuation.formulation_digest,
        continuation.metric_contract_id,
    )


def _continuation_key_parts(
    optimizer_id: str,
    optimizer_version: str,
    optimizer_config_digest: str,
    continuation_contract_id: str,
    formulation_id: str,
    formulation_version: str,
    formulation_digest: str,
    metric_contract_id: str,
) -> ContinuationKey:
    identifiers = (
        optimizer_id,
        optimizer_version,
        optimizer_config_digest,
        continuation_contract_id,
        formulation_id,
        formulation_version,
        formulation_digest,
        metric_contract_id,
    )
    if any(not value.strip() for value in identifiers):
        raise ValueError(
            "formulated continuation lookup requires complete identity"
        )
    return ("formulated", *identifiers)


def _serialized_continuation_key(key: ContinuationKey) -> str:
    """Return an unambiguous JSON-object key for an in-memory tuple key."""

    if len(key) == 2 and key[0] == "legacy":
        return key[1]
    if not key or key[0] != "formulated":
        raise ValueError("unknown optimizer continuation key kind")
    # Length-prefixed fields are readable and reversible without reserving a
    # delimiter that plug-in identifiers would otherwise have to forbid.
    return "formulated:" + "".join(
        f"{len(value)}:{value}" for value in key[1:]
    )


def _serializable_state(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _serializable_state(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, enum.Enum):
        return _serializable_state(value.value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(
                "optimizer state mapping keys must be strings"
            )
        return {
            key: _serializable_state(item)
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


def _freeze_state(value: object) -> object:
    """Take an immutable recursive snapshot for auditable continuation state."""

    if is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__", None)
        if params is None or not params.frozen:
            raise TypeError(
                "optimizer state dataclass payloads must be frozen"
            )
        if any(not item.init for item in fields(value)):
            raise TypeError(
                "optimizer state dataclass fields must all use init=True"
            )
        return replace(
            value,
            **{
                item.name: _freeze_state(getattr(value, item.name))
                for item in fields(value)
            },
        )
    if isinstance(value, enum.Enum):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("optimizer state mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze_state(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_state(item) for item in value)
    return value


def _snapshot_continuation(
    continuation: OptimizerContinuation[object],
) -> OptimizerContinuation[object]:
    """Validate and freeze one independent continuation archive snapshot."""

    _serializable_state(continuation.payload)
    return replace(
        continuation,
        payload=_freeze_state(continuation.payload),
    )


__all__ = [
    "OptimizerContinuation",
    "OptimizerSolveState",
    "SolveTraceContext",
    "SolveTraceEntry",
    "SolveTracePhase",
    "StatefulOptimizer",
]
