"""Central MARS coordinator and workflow event loop."""

from __future__ import annotations

import asyncio
import heapq
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from random import SystemRandom
from time import perf_counter
from typing import Iterable

from .dag import TaskManager, resolve_task_input_bindings
from .domain.artifact import (
    ArtifactRef,
    InputArtifactBinding,
    artifacts_from_bindings,
)
from .domain.execution import Assignment, ExecutionMode
from .domain.task import (
    TaskInstance,
    TaskState,
    resolved_placement_constraints,
)
from .domain.topology import (
    LinkSnapshot,
    LinkSpec,
    NodeSpec,
)
from .domain.transfer import TransferReservation
from .domain.workflow import FailurePolicy, WorkflowSpec
from .network import synthesize_legacy_full_mesh
from .optimizers import (
    FormulationRegistry,
    Optimizer,
    OptimizerRegistry,
    OptimizerSolveState,
    PlannedResourceReservation,
    SchedulingEpoch,
    SchedulingFormulation,
    SchedulingPlan,
    SolveLimits,
)
from .profiling import ProfileCatalog, profile_catalog_from_workloads
from .runtime import (
    AttemptCompletion,
    DispatchAck,
    DispatchCommand,
    RuntimeInventory,
    RuntimePort,
)
from .scheduler import critical_path, plan_scheduling_epoch
from .synthetic_workloads import (
    SyntheticWorkloadCatalog,
    load_default_synthetic_workloads,
)

@dataclass(frozen=True)
class RuntimeEvent:
    sequence: int
    time_ms: float
    event_type: str
    message: str
    workflow_id: str
    task_id: str = ""
    attempt_id: str = ""
    agent_id: str = ""


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    attempt_no: int
    state: str
    target_node_id: str
    mode: str
    start_time_ms: float
    finish_time_ms: float
    compute_time_ms: float
    communication_time_ms: float
    transferred_mb: float
    energy_j: float
    input_artifact_ids: tuple[str, ...]
    error_code: str = ""


@dataclass(frozen=True)
class CoordinatorReport:
    workflow: dict[str, object]
    metrics: dict[str, float | int]
    task_results: tuple[dict[str, object], ...]
    agents: tuple[dict[str, object], ...]
    data_edges: tuple[dict[str, str], ...]
    events: tuple[RuntimeEvent, ...]
    logs: tuple[str, ...]
    scheduling_plans: tuple[SchedulingPlan, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "workflow": _report_data(self.workflow),
            "metrics": _report_data(self.metrics),
            "task_results": [
                _report_data(item) for item in self.task_results
            ],
            "agents": [_report_data(item) for item in self.agents],
            "data_edges": [
                _report_data(item) for item in self.data_edges
            ],
            "events": [asdict(event) for event in self.events],
            "logs": list(self.logs),
        }


def _report_data(value: object) -> object:
    """Return detached data from mutable or read-only report containers."""

    if isinstance(value, Mapping):
        return {
            key: _report_data(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_report_data(item) for item in value]
    if isinstance(value, tuple):
        items = tuple(_report_data(item) for item in value)
        return items if type(value) is tuple else list(items)
    return value


@dataclass(frozen=True)
class _ActiveAttempt:
    task: TaskInstance
    attempt_id: str
    attempt_no: int
    assignment: Assignment
    resource_reservation: PlannedResourceReservation
    transfer_reservations: tuple[TransferReservation, ...]
    input_bindings: tuple[InputArtifactBinding, ...]
    dispatch_id: str
    completion_future: asyncio.Task[AttemptCompletion]
    completion_watermark_ms: float | None = None
    injected_error_code: str = ""

    @property
    def input_artifacts(self) -> tuple[ArtifactRef, ...]:
        return artifacts_from_bindings(self.input_bindings)


class CentralCoordinator:
    """Execute one DAG through the RuntimePort boundary."""

    def __init__(
        self,
        runtime: RuntimePort,
        *,
        workload_catalog: SyntheticWorkloadCatalog | None = None,
        profile_catalog: ProfileCatalog | None = None,
        link_specs: Iterable[LinkSpec] | None = None,
        link_snapshots: Iterable[LinkSnapshot] | None = None,
        optimizer_registry: OptimizerRegistry | None = None,
        formulation_registry: FormulationRegistry | None = None,
        fallback_optimizer: str | Optimizer | None = "heuristic",
    ) -> None:
        if not isinstance(runtime, RuntimePort):
            raise TypeError("runtime must implement RuntimePort")
        if (link_specs is None) != (link_snapshots is None):
            raise ValueError(
                "link_specs and link_snapshots must both be provided or omitted"
            )
        self.runtime = runtime
        self.workload_catalog = (
            workload_catalog or load_default_synthetic_workloads()
        )
        self.profile_catalog = (
            profile_catalog
            if profile_catalog is not None
            else profile_catalog_from_workloads(self.workload_catalog)
        )
        self._configured_link_specs = (
            None if link_specs is None else tuple(link_specs)
        )
        self._configured_link_snapshots = (
            None if link_snapshots is None else tuple(link_snapshots)
        )
        self.optimizer_registry = optimizer_registry
        self.formulation_registry = formulation_registry
        self.fallback_optimizer = fallback_optimizer
        self._events: list[RuntimeEvent] = []
        self._sequence = 0
        self._started = False
        self._run_started = False
        self._optimizer_solve_state: OptimizerSolveState | None = None
        self._runtime_loop: asyncio.AbstractEventLoop | None = None
        self._runtime_view: dict[str, object] = {
            "scheduler_id": "mars-central",
            "status": "initializing",
            "agent_count": 0,
            "agents": [],
        }

    def describe(self) -> dict[str, object]:
        """Return the latest cached view without touching the runtime adapter."""

        return deepcopy(self._runtime_view)

    @property
    def optimizer_solve_state(self) -> OptimizerSolveState | None:
        """Expose the workflow solve trace, including after a failed run."""

        return self._optimizer_solve_state

    async def describe_async(self) -> dict[str, object]:
        return self.describe()

    async def initialize_async(self) -> dict[str, object]:
        self._bind_runtime_loop()
        if not self._started:
            inventory = await self.runtime.start(0.0)
            self._started = True
            self._update_runtime_view(
                inventory,
                await self.runtime.describe(0.0),
            )
        return self.describe()

    def run(
        self,
        workflow: WorkflowSpec,
        *,
        algorithm: str = "dag_deadline",
        formulation: str | SchedulingFormulation | None = None,
        seed: int = 7,
        max_attempts: int = 2,
        fail_first_task_ids: Iterable[str] = (),
        deterministic: bool = True,
        solve_limits: SolveLimits | None = None,
    ) -> CoordinatorReport:
        """Run one workflow from synchronous application code."""

        return asyncio.run(
            self.run_async(
                workflow,
                algorithm=algorithm,
                formulation=formulation,
                seed=seed,
                max_attempts=max_attempts,
                fail_first_task_ids=fail_first_task_ids,
                deterministic=deterministic,
                solve_limits=solve_limits,
            )
        )

    async def run_async(
        self,
        workflow: WorkflowSpec,
        *,
        algorithm: str = "dag_deadline",
        formulation: str | SchedulingFormulation | None = None,
        seed: int = 7,
        max_attempts: int = 2,
        fail_first_task_ids: Iterable[str] = (),
        deterministic: bool = True,
        solve_limits: SolveLimits | None = None,
    ) -> CoordinatorReport:
        """Run the single event loop shared by local and remote adapters."""

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if self._run_started:
            raise RuntimeError(
                "CentralCoordinator is one-shot; create a new instance per workflow run"
            )
        self._bind_runtime_loop()
        self._run_started = True
        self._events = []
        self._sequence = 0

        current_time_ms = 0.0
        failed_once = frozenset(fail_first_task_ids)
        execution_seed = (
            seed
            if deterministic
            else SystemRandom().randrange(0, 2**63)
        )
        solve_limits = replace(
            solve_limits or SolveLimits(),
            deterministic=deterministic,
            random_seed=execution_seed,
        )
        optimizer_solve_state = OptimizerSolveState(
            session_id=f"workflow:{workflow.workflow_id}",
        )
        self._optimizer_solve_state = optimizer_solve_state
        if self._started:
            inventory = await self.runtime.inventory(current_time_ms)
        else:
            inventory = await self.runtime.start(current_time_ms)
            self._started = True
        self._update_runtime_view(
            inventory,
            await self.runtime.describe(current_time_ms),
        )
        heartbeat_by_id = {
            heartbeat.agent_id: heartbeat
            for heartbeat in inventory.heartbeats
        }
        for node in inventory.nodes:
            heartbeat = heartbeat_by_id[node.node_id]
            self._emit(
                current_time_ms,
                "agent_registered",
                (
                    f"{node.node_id} registered and heartbeat "
                    f"{heartbeat.sequence} received"
                ),
                workflow.workflow_id,
                agent_id=node.node_id,
            )

        manager = TaskManager()
        index = manager.submit(workflow)
        critical_ids, critical_path_ms, critical_tail = critical_path(
            workflow.tasks,
            index,
        )
        if self._configured_link_specs is None:
            resolved_link_specs, resolved_link_snapshots = (
                synthesize_legacy_full_mesh(
                    inventory.nodes,
                    inventory.snapshots.values(),
                )
            )
        else:
            resolved_link_specs = self._configured_link_specs
            resolved_link_snapshots = (
                self._configured_link_snapshots or ()
            )
        profiles = self.profile_catalog

        node_specs = {node.node_id: node for node in inventory.nodes}
        completion_time: dict[str, float] = {}
        attempts_by_task: dict[str, list[AttemptRecord]] = {
            task.task_id: [] for task in workflow.tasks
        }
        next_attempt_no = {
            task.task_id: 1 for task in workflow.tasks
        }
        failed_nodes_by_task: dict[str, set[str]] = {
            task.task_id: set() for task in workflow.tasks
        }
        pending_retries: set[str] = set()
        target_by_task: dict[str, str] = {}
        mode_by_task: dict[str, str] = {}
        active_by_dispatch: dict[str, _ActiveAttempt] = {}
        active_resource_reservations: dict[
            str, PlannedResourceReservation
        ] = {}
        queued_completions: set[str] = set()
        completion_heap: list[
            tuple[float, int, str, AttemptCompletion]
        ] = []
        completion_sequence = 0
        epoch_sequence = 0
        link_available_ms = {
            link.link_id: 0.0 for link in resolved_link_specs
        }
        transferred_mb = 0.0
        transfer_time_ms = 0.0
        total_energy_j = 0.0
        retry_successes = 0
        total_solver_time_ms = 0.0
        max_solver_time_ms = 0.0
        optimizer_counts: Counter[str] = Counter()
        formulation_counts: Counter[str] = Counter()
        policy_counts: Counter[str] = Counter()
        solve_status_counts: Counter[str] = Counter()
        termination_reason_counts: Counter[str] = Counter()
        fallback_count = 0
        scheduling_plans: list[SchedulingPlan] = []

        self._emit(
            current_time_ms,
            "workflow_accepted",
            (
                f"workflow {workflow.workflow_id} accepted with "
                f"{len(workflow.tasks)} tasks"
            ),
            workflow.workflow_id,
        )

        def ready_at(task: TaskInstance) -> float:
            return max(
                task.arrival_time_ms,
                max(
                    (
                        completion_time.get(parent_id, 0.0)
                        for parent_id in index.parents[task.task_id]
                    ),
                    default=0.0,
                ),
            )

        async def plan_tasks(
            tasks: tuple[TaskInstance, ...],
            *,
            excluded_node_ids: dict[
                str, frozenset[str]
            ] | None = None,
            epoch_kind: str = "runtime",
        ) -> tuple[
            SchedulingPlan,
            dict[str, tuple[InputArtifactBinding, ...]],
            dict[str, float],
        ]:
            nonlocal epoch_sequence, inventory, node_specs
            nonlocal total_solver_time_ms, max_solver_time_ms
            nonlocal fallback_count
            inventory = await self.runtime.inventory(current_time_ms)
            node_specs = {
                node.node_id: node for node in inventory.nodes
            }
            epoch_sequence += 1
            epoch = SchedulingEpoch(
                epoch_id=(
                    f"{workflow.workflow_id}:{epoch_kind}-epoch:"
                    f"{epoch_sequence}"
                ),
                now_ms=current_time_ms,
                ready_tasks=tasks,
            )
            bindings = {
                task.task_id: resolve_task_input_bindings(
                    manager,
                    task.task_id,
                )
                for task in tasks
            }
            ready_times = {
                task.task_id: max(
                    current_time_ms,
                    ready_at(task),
                )
                for task in tasks
            }
            active_attempt_ids = {
                active.attempt_id
                for active in active_by_dispatch.values()
            }
            carry_in = tuple(
                reservation
                for attempt_id, reservation
                in active_resource_reservations.items()
                if attempt_id in active_attempt_ids
            )
            planning_started = perf_counter()
            plan = plan_scheduling_epoch(
                epoch,
                optimizer=algorithm,
                formulation=formulation,
                node_specs=node_specs,
                node_snapshots=inventory.snapshots,
                input_artifact_bindings=bindings,
                ready_time_ms=ready_times,
                node_available_ms={
                    node_id: current_time_ms
                    for node_id in node_specs
                },
                link_specs=resolved_link_specs,
                link_snapshots=resolved_link_snapshots,
                link_available_ms=link_available_ms,
                existing_node_reservations=carry_in,
                critical_tail_ms={
                    task.task_id: critical_tail[task.task_id]
                    for task in tasks
                },
                profiles=profiles,
                excluded_node_ids=excluded_node_ids,
                solve_limits=solve_limits,
                registry=self.optimizer_registry,
                formulation_registry=self.formulation_registry,
                fallback_optimizer=self.fallback_optimizer,
                solve_state=optimizer_solve_state,
            )
            scheduling_plans.append(plan)
            planning_elapsed_ms = (
                perf_counter() - planning_started
            ) * 1000.0
            total_solver_time_ms += planning_elapsed_ms
            max_solver_time_ms = max(
                max_solver_time_ms,
                planning_elapsed_ms,
            )
            optimizer_counts[plan.optimizer_id] += 1
            formulation_counts[
                plan.formulation_id or "unformulated"
            ] += 1
            policy_counts[plan.policy_id] += 1
            solve_status_counts[plan.solve_status.value] += 1
            termination_reason_counts[
                plan.termination_reason or "unspecified"
            ] += 1
            if "fallback_optimizer" in plan.diagnostics:
                fallback_count += 1
            if not plan.assignments:
                raise RuntimeError(
                    "optimizer deferred every ready task; the runtime "
                    "requires at least one committable assignment per epoch"
                )
            if plan.deferred_task_ids:
                raise RuntimeError(
                    "the coordinator does not commit partial plans with "
                    "deferred ready tasks"
                )
            self._emit(
                current_time_ms,
                "scheduling_epoch_planned",
                (
                    f"{epoch.epoch_id} planned {len(tasks)} ready tasks "
                    f"with optimizer {plan.optimizer_id} and policy "
                    f"{plan.policy_id}"
                ),
                workflow.workflow_id,
            )
            return plan, bindings, ready_times

        def schedule_retry(
            task: TaskInstance,
            attempt_no: int,
            *,
            dispatch_rejected: bool = False,
        ) -> bool:
            placement = resolved_placement_constraints(task)
            retry_allowed = (
                dispatch_rejected
                or placement.idempotent
                and not placement.stateful
            )
            if attempt_no >= max_attempts or not retry_allowed:
                if attempt_no < max_attempts and not dispatch_rejected:
                    self._emit(
                        current_time_ms,
                        "retry_suppressed",
                        (
                            f"{task.task_id} is stateful or "
                            "non-idempotent; automatic retry suppressed"
                        ),
                        workflow.workflow_id,
                        task_id=task.task_id,
                    )
                return False
            pending_retries.add(task.task_id)
            self._emit(
                current_time_ms,
                "retry_scheduled",
                (
                    f"{task.task_id} retry "
                    f"{next_attempt_no[task.task_id]} scheduled"
                ),
                workflow.workflow_id,
                task_id=task.task_id,
            )
            return True

        async def dispatch_assignment(
            plan: SchedulingPlan,
            assignment: Assignment,
            bindings_by_task: dict[
                str, tuple[InputArtifactBinding, ...]
            ],
        ) -> None:
            nonlocal completion_sequence
            task = manager.get(assignment.task_id)
            if manager.state_of(task.task_id) is TaskState.READY:
                manager.mark_running(task.task_id)
            attempt_no = next_attempt_no[task.task_id]
            next_attempt_no[task.task_id] += 1
            attempt_id = (
                f"{workflow.workflow_id}:{task.task_id}:"
                f"attempt:{attempt_no}"
            )
            input_bindings = bindings_by_task[task.task_id]
            input_artifacts = artifacts_from_bindings(input_bindings)

            if not assignment.target_node_id:
                event_time = max(
                    current_time_ms,
                    task.arrival_time_ms,
                    assignment.estimated_finish_ms,
                )
                attempts_by_task[task.task_id].append(
                    AttemptRecord(
                        attempt_id=attempt_id,
                        attempt_no=attempt_no,
                        state=TaskState.DROPPED.value,
                        target_node_id="",
                        mode=ExecutionMode.DROP.value,
                        start_time_ms=event_time,
                        finish_time_ms=event_time,
                        compute_time_ms=0.0,
                        communication_time_ms=0.0,
                        transferred_mb=0.0,
                        energy_j=0.0,
                        input_artifact_ids=tuple(
                            item.artifact_id
                            for item in input_artifacts
                        ),
                        error_code="no_feasible_agent",
                    )
                )
                target_by_task[task.task_id] = ""
                mode_by_task[task.task_id] = ExecutionMode.DROP.value
                manager.complete(
                    task.task_id,
                    ok=False,
                    finished_time_ms=event_time,
                    dropped=True,
                    error_code="no_feasible_agent",
                )
                completion_time[task.task_id] = event_time
                self._emit(
                    event_time,
                    "task_dropped",
                    f"{task.task_id} has no feasible agent",
                    workflow.workflow_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                )
                return

            injected_error_code = ""
            injected_failure = (
                attempt_no == 1
                and task.task_id in failed_once
            )
            if injected_failure:
                injected_error_code = (
                    "injected_first_attempt_failure"
                )

            resource_reservation = next(
                reservation
                for reservation in plan.node_reservations
                if reservation.task_id == task.task_id
            )
            transfer_reservations = tuple(
                reservation
                for reservation in plan.transfer_reservations
                if reservation.task_id == task.task_id
            )
            try:
                ack = await self.runtime.dispatch(
                    DispatchCommand(
                        attempt_id=attempt_id,
                        attempt_no=attempt_no,
                        task=task,
                        assignment=assignment,
                        resource_reservation=resource_reservation,
                        transfer_reservations=transfer_reservations,
                        input_artifact_bindings=input_bindings,
                        problem_id=plan.problem_id,
                        solve_request_id=plan.solve_request_id,
                        snapshot_id=plan.snapshot_id,
                        policy_id=plan.policy_id,
                        policy_version=plan.policy_version,
                        seed=execution_seed,
                        inject_failure=injected_failure,
                    )
                )
            except BaseException:
                await _best_effort_cancel(
                    self.runtime,
                    attempt_id,
                    "dispatch_failed",
                    current_time_ms,
                )
                raise

            if not ack.accepted:
                failed_nodes_by_task[task.task_id].add(
                    assignment.target_node_id
                )
                reject_reason = (
                    ack.error_code or "dispatch_rejected"
                )
                attempts_by_task[task.task_id].append(
                    AttemptRecord(
                        attempt_id=attempt_id,
                        attempt_no=attempt_no,
                        state=TaskState.FAILED.value,
                        target_node_id=assignment.target_node_id,
                        mode=assignment.execution_mode.value,
                        start_time_ms=current_time_ms,
                        finish_time_ms=current_time_ms,
                        compute_time_ms=0.0,
                        communication_time_ms=0.0,
                        transferred_mb=0.0,
                        energy_j=0.0,
                        input_artifact_ids=tuple(
                            item.artifact_id
                            for item in input_artifacts
                        ),
                        error_code=reject_reason,
                    )
                )
                target_by_task[task.task_id] = (
                    assignment.target_node_id
                )
                mode_by_task[task.task_id] = (
                    assignment.execution_mode.value
                )
                self._emit(
                    current_time_ms,
                    "dispatch_rejected",
                    (
                        f"{assignment.target_node_id} rejected "
                        f"{task.task_id}: {reject_reason}"
                    ),
                    workflow.workflow_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    agent_id=assignment.target_node_id,
                )
                if schedule_retry(
                    task,
                    attempt_no,
                    dispatch_rejected=True,
                ):
                    return
                manager.complete(
                    task.task_id,
                    ok=False,
                    finished_time_ms=current_time_ms,
                    dropped=True,
                    error_code=reject_reason,
                )
                completion_time[task.task_id] = current_time_ms
                return

            try:
                _validate_dispatch_ack(
                    ack,
                    attempt_id=attempt_id,
                    task_id=task.task_id,
                    agent_id=assignment.target_node_id,
                    require_schedule=(
                        self.runtime.capabilities.virtual_time
                    ),
                )
            except BaseException:
                await _best_effort_cancel(
                    self.runtime,
                    attempt_id,
                    "dispatch_ack_mismatch",
                    current_time_ms,
                )
                raise

            for reservation in transfer_reservations:
                for link_id in reservation.path_link_ids:
                    link_available_ms[link_id] = max(
                        link_available_ms.get(link_id, 0.0),
                        reservation.finish_ms,
                    )
            completion_future = asyncio.create_task(
                self.runtime.receive_completion(ack.dispatch_id)
            )
            committed_reservation = resource_reservation
            if ack.scheduled_start_ms is not None:
                assert ack.scheduled_finish_ms is not None
                committed_reservation = replace(
                    resource_reservation,
                    start_ms=ack.scheduled_start_ms,
                    finish_ms=ack.scheduled_finish_ms,
                )
            active = _ActiveAttempt(
                task=task,
                attempt_id=attempt_id,
                attempt_no=attempt_no,
                assignment=assignment,
                resource_reservation=committed_reservation,
                transfer_reservations=transfer_reservations,
                input_bindings=input_bindings,
                dispatch_id=ack.dispatch_id,
                completion_future=completion_future,
                completion_watermark_ms=ack.scheduled_finish_ms,
                injected_error_code=injected_error_code,
            )
            active_by_dispatch[ack.dispatch_id] = active
            active_resource_reservations[attempt_id] = (
                committed_reservation
            )
            self._emit(
                current_time_ms,
                "attempt_dispatched",
                (
                    f"{task.task_id} attempt {attempt_no} "
                    f"dispatched to {ack.agent_id}"
                ),
                workflow.workflow_id,
                task_id=task.task_id,
                attempt_id=attempt_id,
                agent_id=ack.agent_id,
            )

        async def harvest_completions() -> None:
            nonlocal completion_sequence
            for dispatch_id, active in tuple(
                active_by_dispatch.items()
            ):
                if (
                    dispatch_id in queued_completions
                    or not active.completion_future.done()
                ):
                    continue
                try:
                    completion = active.completion_future.result()
                    _validate_completion(
                        completion,
                        dispatch_id=dispatch_id,
                        attempt_id=active.attempt_id,
                        task_id=active.task.task_id,
                        agent_id=active.assignment.target_node_id,
                        expected_start_ms=(
                            active.resource_reservation.start_ms
                            if active.completion_watermark_ms is not None
                            else None
                        ),
                        expected_finish_ms=(
                            active.completion_watermark_ms
                        ),
                    )
                except BaseException:
                    raise
                completion_sequence += 1
                queued_completions.add(dispatch_id)
                active_resource_reservations[
                    active.attempt_id
                ] = replace(
                    active.resource_reservation,
                    start_ms=completion.started_time_ms,
                    finish_ms=completion.finished_time_ms,
                )
                heapq.heappush(
                    completion_heap,
                    (
                        completion.finished_time_ms,
                        completion_sequence,
                        dispatch_id,
                        completion,
                    ),
                )

        def unresolved_completion_watermark_ms() -> float:
            """Earliest event time that an unreceived virtual result may hold."""

            if not self.runtime.capabilities.virtual_time:
                return float("inf")
            return min(
                (
                    active.completion_watermark_ms
                    for dispatch_id, active
                    in active_by_dispatch.items()
                    if (
                        dispatch_id not in queued_completions
                        and active.completion_watermark_ms is not None
                    )
                ),
                default=float("inf"),
            )

        def process_completion(
            dispatch_id: str,
            execution: AttemptCompletion,
        ) -> None:
            nonlocal transferred_mb
            nonlocal transfer_time_ms
            nonlocal total_energy_j
            nonlocal retry_successes
            active = active_by_dispatch.pop(dispatch_id)
            queued_completions.discard(dispatch_id)
            active_resource_reservations.pop(
                active.attempt_id,
                None,
            )
            task = active.task
            input_artifacts = active.input_artifacts
            attempt_transfer_mb = sum(
                artifact.size_mb
                for artifact in input_artifacts
                if artifact.node_id != execution.agent_id
            )
            if (
                not input_artifacts
                and execution.agent_id != task.source_node_id
            ):
                attempt_transfer_mb = task.spec.input_size_mb
            error_code = (
                ""
                if execution.ok
                else active.injected_error_code
                or execution.error_code
                or "execution_failed"
            )
            attempt_start_ms = execution.started_time_ms
            if active.transfer_reservations:
                attempt_start_ms = min(
                    attempt_start_ms,
                    *(
                        reservation.start_ms
                        for reservation
                        in active.transfer_reservations
                    ),
                )
            attempt = AttemptRecord(
                attempt_id=active.attempt_id,
                attempt_no=active.attempt_no,
                state=(
                    TaskState.SUCCEEDED.value
                    if execution.ok
                    else TaskState.FAILED.value
                ),
                target_node_id=execution.agent_id,
                mode=active.assignment.execution_mode.value,
                start_time_ms=round(
                    attempt_start_ms,
                    4,
                ),
                finish_time_ms=round(
                    execution.finished_time_ms,
                    4,
                ),
                compute_time_ms=round(
                    execution.compute_time_ms,
                    4,
                ),
                communication_time_ms=round(
                    active.assignment.communication_ms,
                    4,
                ),
                transferred_mb=round(
                    attempt_transfer_mb,
                    6,
                ),
                energy_j=round(execution.energy_j, 6),
                input_artifact_ids=tuple(
                    artifact.artifact_id
                    for artifact in input_artifacts
                ),
                error_code=error_code,
            )
            attempts_by_task[task.task_id].append(attempt)
            transferred_mb += attempt_transfer_mb
            transfer_time_ms += (
                active.assignment.communication_ms
            )
            total_energy_j += execution.energy_j
            target_by_task[task.task_id] = execution.agent_id
            mode_by_task[task.task_id] = (
                active.assignment.execution_mode.value
            )

            if execution.ok:
                manager.complete(
                    task.task_id,
                    ok=True,
                    finished_time_ms=execution.finished_time_ms,
                    outputs=execution.outputs,
                )
                completion_time[task.task_id] = (
                    execution.finished_time_ms
                )
                if active.attempt_no > 1:
                    retry_successes += 1
                self._emit(
                    execution.finished_time_ms,
                    "attempt_succeeded",
                    (
                        f"{task.task_id} attempt "
                        f"{active.attempt_no} completed on "
                        f"{execution.agent_id}"
                    ),
                    workflow.workflow_id,
                    task_id=task.task_id,
                    attempt_id=active.attempt_id,
                    agent_id=execution.agent_id,
                )
                for output in execution.outputs:
                    self._emit(
                        execution.finished_time_ms,
                        "artifact_published",
                        (
                            f"{output.artifact_id} published from "
                            f"{output.producer_port}"
                        ),
                        workflow.workflow_id,
                        task_id=task.task_id,
                        attempt_id=active.attempt_id,
                        agent_id=execution.agent_id,
                    )
                return

            failed_nodes_by_task[task.task_id].add(
                execution.agent_id
            )
            self._emit(
                execution.finished_time_ms,
                "attempt_failed",
                (
                    f"{task.task_id} attempt {active.attempt_no} "
                    f"failed on {execution.agent_id}"
                ),
                workflow.workflow_id,
                task_id=task.task_id,
                attempt_id=active.attempt_id,
                agent_id=execution.agent_id,
            )
            if schedule_retry(task, active.attempt_no):
                return
            manager.complete(
                task.task_id,
                ok=False,
                finished_time_ms=execution.finished_time_ms,
                timed_out=(
                    execution.finished_time_ms
                    > task.deadline_time_ms
                ),
                error_code=error_code,
            )
            completion_time[task.task_id] = (
                execution.finished_time_ms
            )

        async def cancel_active(reason: str) -> None:
            active_items = tuple(active_by_dispatch.values())
            for active in active_items:
                if not active.completion_future.done():
                    active.completion_future.cancel()
                await _best_effort_cancel(
                    self.runtime,
                    active.attempt_id,
                    reason,
                    current_time_ms,
                )
            if active_items:
                await asyncio.gather(
                    *(
                        active.completion_future
                        for active in active_items
                    ),
                    return_exceptions=True,
                )

        try:
            while manager.unresolved():
                if active_by_dispatch:
                    await asyncio.sleep(0)
                    await harvest_completions()

                processed_completion = False
                completion_watermark_ms = (
                    unresolved_completion_watermark_ms()
                )
                while (
                    completion_heap
                    and completion_heap[0][0]
                    <= current_time_ms + 1e-9
                    and (
                        completion_watermark_ms == float("inf")
                        or completion_heap[0][0]
                        < completion_watermark_ms - 1e-9
                    )
                ):
                    (
                        _,
                        _,
                        dispatch_id,
                        execution,
                    ) = heapq.heappop(completion_heap)
                    process_completion(dispatch_id, execution)
                    processed_completion = True
                if processed_completion:
                    continue

                fail_fast_waiting = (
                    workflow.failure_policy
                    is FailurePolicy.FAIL_FAST
                    and bool(active_by_dispatch)
                )
                if pending_retries and not fail_fast_waiting:
                    task_id = min(pending_retries)
                    pending_retries.remove(task_id)
                    task = manager.get(task_id)
                    exclusions = {
                        task_id: frozenset(
                            failed_nodes_by_task[task_id]
                        )
                    }
                    plan, bindings, _ = await plan_tasks(
                        (task,),
                        excluded_node_ids=exclusions,
                        epoch_kind="retry",
                    )
                    assignment = plan.assignments[0]
                    if (
                        not assignment.target_node_id
                        and failed_nodes_by_task[task_id]
                    ):
                        plan, bindings, _ = await plan_tasks(
                            (task,),
                            epoch_kind="retry-fallback",
                        )
                        assignment = plan.assignments[0]
                    await dispatch_assignment(
                        plan,
                        assignment,
                        bindings,
                    )
                    continue

                ready = manager.ready()
                arrived = tuple(
                    sorted(
                        (
                            task
                            for task in ready
                            if ready_at(task)
                            <= current_time_ms + 1e-9
                        ),
                        key=lambda item: item.task_id,
                    )
                )
                if arrived and not fail_fast_waiting:
                    plan, bindings, _ = await plan_tasks(arrived)
                    ordered_assignments = sorted(
                        plan.assignments,
                        key=lambda item: (
                            item.estimated_start_ms,
                            item.estimated_finish_ms,
                            item.task_id,
                        ),
                    )
                    if (
                        workflow.failure_policy
                        is FailurePolicy.FAIL_FAST
                    ):
                        drops = [
                            item
                            for item in ordered_assignments
                            if not item.target_node_id
                        ]
                        ordered_assignments = [
                            drops[0]
                            if drops
                            else ordered_assignments[0]
                        ]
                    for assignment in ordered_assignments:
                        await dispatch_assignment(
                            plan,
                            assignment,
                            bindings,
                        )
                    continue

                next_arrival_ms = float("inf")
                if not fail_fast_waiting:
                    next_arrival_ms = min(
                        (ready_at(task) for task in ready),
                        default=float("inf"),
                    )
                next_completion_ms = (
                    completion_heap[0][0]
                    if completion_heap
                    else float("inf")
                )
                completion_watermark_ms = (
                    unresolved_completion_watermark_ms()
                )
                next_available_event_ms = min(
                    next_completion_ms,
                    next_arrival_ms,
                )
                if (
                    next_available_event_ms != float("inf")
                    and (
                        completion_watermark_ms == float("inf")
                        or next_available_event_ms
                        < completion_watermark_ms - 1e-9
                    )
                ):
                    current_time_ms = max(
                        current_time_ms,
                        next_available_event_ms,
                    )
                    continue

                pending_receiver_items = [
                    active
                    for dispatch_id, active
                    in active_by_dispatch.items()
                    if dispatch_id not in queued_completions
                ]
                if pending_receiver_items:
                    if completion_watermark_ms != float("inf"):
                        pending_receiver_items = [
                            active
                            for active in pending_receiver_items
                            if (
                                active.completion_watermark_ms is not None
                                and abs(
                                    active.completion_watermark_ms
                                    - completion_watermark_ms
                                )
                                <= 1e-9
                            )
                        ]
                    await asyncio.wait(
                        [
                            active.completion_future
                            for active in pending_receiver_items
                        ],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    await harvest_completions()
                    continue
                raise RuntimeError(
                    "workflow is unresolved but has neither ready "
                    "tasks nor active attempts"
                )
        except BaseException:
            await cancel_active("coordinator_run_failed")
            raise

        progress = manager.progress(critical_ids)
        task_results: list[dict[str, object]] = []
        for task_id in index.topological_order:
            task = manager.get(task_id)
            outputs = manager.artifacts_for(task_id)
            task_attempts = attempts_by_task[task_id]
            task_results.append(
                {
                    "task_id": task_id,
                    "task_name": task.name,
                    "task_type": task.spec.task_type,
                    "task_class": task.spec.task_class.value,
                    "state": manager.state_of(task_id).value,
                    "source_node_id": task.source_node_id,
                    "target_node_id": target_by_task.get(
                        task_id,
                        "",
                    ),
                    "mode": mode_by_task.get(task_id, ""),
                    "dependencies": list(index.parents[task_id]),
                    "attempt_count": len(task_attempts),
                    "attempts": [
                        asdict(attempt)
                        for attempt in task_attempts
                    ],
                    "outputs": [
                        asdict(output) for output in outputs
                    ],
                }
            )

        makespan_ms = max(
            completion_time.values(),
            default=current_time_ms,
        )
        states = Counter(
            str(item["state"]) for item in task_results
        )
        attempt_count = sum(
            len(items) for items in attempts_by_task.values()
        )
        retry_count = sum(
            max(0, len(items) - 1)
            for items in attempts_by_task.values()
        )
        succeeded = states[TaskState.SUCCEEDED.value]
        skipped = states[TaskState.SKIPPED.value]
        deadline_missed_by_task = {
            task_id: bool(
                attempts_by_task[task_id]
                and (
                    attempts_by_task[task_id][-1].finish_time_ms
                    > manager.get(task_id).deadline_time_ms
                    or manager.state_of(task_id) is TaskState.TIMEOUT
                )
            )
            for task_id in index.topological_order
        }
        executed_task_ids = {
            task_id
            for task_id, attempts in attempts_by_task.items()
            if any(attempt.compute_time_ms > 0.0 for attempt in attempts)
        }
        deadline_misses = sum(deadline_missed_by_task.values())
        executed_deadline_misses = sum(
            deadline_missed_by_task[task_id]
            for task_id in executed_task_ids
        )
        required_tasks_on_time = sum(
            manager.state_of(task_id) is TaskState.SUCCEEDED
            and not deadline_missed_by_task[task_id]
            for task_id in index.topological_order
        )
        offloaded = sum(
            item["mode"] == ExecutionMode.EDGE.value
            for item in task_results
        )
        safety_violations = sum(
            _violates_safety_contract(
                manager.get(str(item["task_id"])),
                str(item["target_node_id"]),
                node_specs,
            )
            for item in task_results
        )
        metrics: dict[str, float | int] = {
            "task_count": len(task_results),
            "succeeded_task_count": succeeded,
            "failed_task_count": len(task_results) - succeeded,
            "success_rate": round(
                succeeded / max(1, len(task_results)),
                4,
            ),
            # Keep the legacy all-task rate for API compatibility. The two
            # explicit metrics below avoid making skipped work look timely.
            "deadline_miss_rate": round(
                deadline_misses / max(1, len(task_results)),
                4,
            ),
            "executed_deadline_miss_rate": round(
                executed_deadline_misses
                / max(1, len(executed_task_ids)),
                4,
            ),
            "required_task_on_time_rate": round(
                required_tasks_on_time / max(1, len(task_results)),
                4,
            ),
            "skipped_task_count": skipped,
            "attempt_count": attempt_count,
            "retry_count": retry_count,
            "retry_success_count": retry_successes,
            "transferred_mb": round(transferred_mb, 6),
            "transfer_time_ms": round(transfer_time_ms, 4),
            "total_energy_j": round(total_energy_j, 6),
            "total_solver_time_ms": round(
                total_solver_time_ms,
                6,
            ),
            "max_solver_time_ms": round(max_solver_time_ms, 6),
            "scheduling_epoch_count": epoch_sequence,
            "fallback_count": fallback_count,
            "makespan_ms": round(makespan_ms, 4),
            "edge_offload_ratio": round(
                offloaded / max(1, len(task_results)),
                4,
            ),
            "safety_violation_count": safety_violations,
            "critical_path_ms": round(critical_path_ms, 4),
        }
        final_inventory = await self.runtime.inventory(makespan_ms)
        agent_report = await self.runtime.describe(makespan_ms)
        self._update_runtime_view(
            final_inventory,
            agent_report,
        )
        return CoordinatorReport(
            workflow={
                "workflow_id": workflow.workflow_id,
                "state": progress.state.value,
                "failure_policy": workflow.failure_policy.value,
                "state_counts": dict(states),
                "critical_path": list(critical_ids),
                "topological_order": list(index.topological_order),
                "levels": index.levels,
                "scheduling": {
                    "requested_algorithm": algorithm,
                    "requested_formulation": (
                        formulation
                        if isinstance(formulation, str)
                        else formulation.spec.formulation_id
                        if formulation is not None
                        else ""
                    ),
                    "effective_optimizers": dict(optimizer_counts),
                    "effective_formulations": dict(formulation_counts),
                    "effective_policies": dict(policy_counts),
                    "solve_statuses": dict(solve_status_counts),
                    "termination_reasons": dict(
                        termination_reason_counts
                    ),
                    "fallback_count": fallback_count,
                    "requested_seed": seed,
                    "deterministic": deterministic,
                    "execution_seed": execution_seed,
                    "solve_limits": {
                        "solve_budget_ms": solve_limits.solve_budget_ms,
                        "max_iterations": solve_limits.max_iterations,
                        "deterministic": solve_limits.deterministic,
                        "random_seed": solve_limits.random_seed,
                    },
                    "optimizer_solve_state": (
                        optimizer_solve_state.as_dict()
                    ),
                },
            },
            metrics=metrics,
            task_results=tuple(task_results),
            agents=agent_report,
            data_edges=tuple(
                asdict(edge) for edge in workflow.data_edges
            ),
            events=tuple(self._events),
            logs=tuple(
                event.message for event in self._events
            ),
            scheduling_plans=tuple(scheduling_plans),
        )

    def _update_runtime_view(
        self,
        inventory: RuntimeInventory,
        agents: tuple[dict[str, object], ...],
    ) -> None:
        self._runtime_view = {
            "scheduler_id": "mars-central",
            "status": "online",
            "agent_count": len(inventory.nodes),
            "agents": list(agents),
        }

    def _bind_runtime_loop(self) -> None:
        current_loop = asyncio.get_running_loop()
        if self._runtime_loop is None:
            self._runtime_loop = current_loop
            return
        if self._runtime_loop is not current_loop:
            raise RuntimeError(
                "CentralCoordinator and its RuntimePort must remain "
                "on one event loop"
            )

    def _emit(
        self,
        time_ms: float,
        event_type: str,
        message: str,
        workflow_id: str,
        *,
        task_id: str = "",
        attempt_id: str = "",
        agent_id: str = "",
    ) -> None:
        self._sequence += 1
        self._events.append(
            RuntimeEvent(
                sequence=self._sequence,
                time_ms=round(time_ms, 4),
                event_type=event_type,
                message=message,
                workflow_id=workflow_id,
                task_id=task_id,
                attempt_id=attempt_id,
                agent_id=agent_id,
            )
        )


async def _best_effort_cancel(
    runtime: RuntimePort,
    attempt_id: str,
    reason: str,
    now_ms: float,
) -> None:
    try:
        await runtime.cancel(attempt_id, reason, now_ms)
    except BaseException:
        pass


def _validate_dispatch_ack(
    ack: DispatchAck,
    *,
    attempt_id: str,
    task_id: str,
    agent_id: str,
    require_schedule: bool = False,
) -> None:
    expected = (attempt_id, task_id, agent_id)
    actual = (ack.attempt_id, ack.task_id, ack.agent_id)
    if not ack.dispatch_id or actual != expected:
        raise RuntimeError(
            "runtime returned mismatched dispatch acknowledgement: "
            f"{actual!r}"
        )
    if require_schedule and ack.scheduled_finish_ms is None:
        raise RuntimeError(
            "virtual-time runtime omitted its authoritative dispatch schedule"
        )


def _validate_completion(
    completion: AttemptCompletion,
    *,
    dispatch_id: str,
    attempt_id: str,
    task_id: str,
    agent_id: str,
    expected_start_ms: float | None = None,
    expected_finish_ms: float | None = None,
) -> None:
    expected = (dispatch_id, attempt_id, task_id, agent_id)
    actual = (
        completion.dispatch_id,
        completion.attempt_id,
        completion.task_id,
        completion.agent_id,
    )
    if actual != expected:
        raise RuntimeError(
            f"runtime returned mismatched completion: {actual!r}"
        )
    if (
        expected_start_ms is not None
        and abs(completion.started_time_ms - expected_start_ms) > 1e-6
    ):
        raise RuntimeError(
            "runtime completion start does not match its dispatch schedule"
        )
    if (
        expected_finish_ms is not None
        and abs(completion.finished_time_ms - expected_finish_ms) > 1e-6
    ):
        raise RuntimeError(
            "runtime completion finish does not match its dispatch schedule"
        )


def _violates_safety_contract(
    task: TaskInstance,
    target_node_id: str,
    node_specs: dict[str, NodeSpec],
) -> bool:
    constraints = resolved_placement_constraints(task)
    if not constraints.safety_required or not target_node_id:
        return False
    target = node_specs.get(target_node_id)
    return (
        target is None
        or not target.safety_capable
        or bool(constraints.pinned_node_id)
        and target_node_id != constraints.pinned_node_id
    )
