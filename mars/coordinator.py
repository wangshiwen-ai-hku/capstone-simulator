"""Central MARS coordinator for local multi-agent workflow exercises."""

from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Iterable

from .dag import TaskManager, resolve_task_inputs
from .models import (
    Assignment,
    ExecutionMode,
    LinkSnapshot,
    LinkSpec,
    NodeKind,
    NodeSpec,
    TaskInstance,
    TaskState,
    WorkflowSpec,
    resolved_placement_constraints,
)
from .network import synthesize_legacy_full_mesh
from .optimizers import (
    OptimizerRegistry,
    SchedulingEpoch,
)
from .profiling import ExecutionProfile, ProfileCatalog
from .runtime import (
    AttemptCompletion,
    DispatchAck,
    DispatchCommand,
    RuntimeInventory,
    RuntimePort,
)
from .scheduler import critical_path, plan_scheduling_epoch
from .synthetic_workloads import (
    ExecutionTarget,
    SyntheticSampler,
    SyntheticWorkloadCatalog,
    UnsupportedTargetError,
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

    def as_dict(self) -> dict[str, object]:
        return {
            "workflow": self.workflow,
            "metrics": self.metrics,
            "task_results": list(self.task_results),
            "agents": list(self.agents),
            "data_edges": list(self.data_edges),
            "events": [asdict(event) for event in self.events],
            "logs": list(self.logs),
        }


class CentralCoordinator:
    """Execute one DAG through a single asynchronous runtime boundary."""

    def __init__(
        self,
        runtime: RuntimePort,
        *,
        workload_catalog: SyntheticWorkloadCatalog | None = None,
        link_specs: Iterable[LinkSpec] | None = None,
        link_snapshots: Iterable[LinkSnapshot] | None = None,
        optimizer_registry: OptimizerRegistry | None = None,
    ) -> None:
        if not isinstance(runtime, RuntimePort):
            raise TypeError("runtime must implement RuntimePort")
        self.runtime = runtime
        self.workload_catalog = workload_catalog or load_default_synthetic_workloads()
        if (link_specs is None) != (link_snapshots is None):
            raise ValueError(
                "link_specs and link_snapshots must both be provided or omitted"
            )
        self._configured_link_specs = (
            None if link_specs is None else tuple(link_specs)
        )
        self._configured_link_snapshots = (
            None if link_snapshots is None else tuple(link_snapshots)
        )
        self.optimizer_registry = optimizer_registry
        self._events: list[RuntimeEvent] = []
        self._sequence = 0
        self._started = False
        self._run_started = False
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
        seed: int = 7,
        max_attempts: int = 2,
        fail_first_task_ids: Iterable[str] = (),
        deterministic: bool = True,
    ) -> CoordinatorReport:
        """Synchronous adapter used by the background worker and tests."""

        return asyncio.run(
            self.run_async(
                workflow,
                algorithm=algorithm,
                seed=seed,
                max_attempts=max_attempts,
                fail_first_task_ids=fail_first_task_ids,
                deterministic=deterministic,
            )
        )

    async def run_async(
        self,
        workflow: WorkflowSpec,
        *,
        algorithm: str = "dag_deadline",
        seed: int = 7,
        max_attempts: int = 2,
        fail_first_task_ids: Iterable[str] = (),
        deterministic: bool = True,
    ) -> CoordinatorReport:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
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
            heartbeat.agent_id: heartbeat for heartbeat in inventory.heartbeats
        }
        for node in inventory.nodes:
            heartbeat = heartbeat_by_id[node.node_id]
            self._emit(
                current_time_ms,
                "agent_registered",
                f"{node.node_id} registered and heartbeat {heartbeat.sequence} received",
                workflow.workflow_id,
                agent_id=node.node_id,
            )

        manager = TaskManager()
        index = manager.submit(workflow)
        critical_ids, critical_path_ms, critical_tail = critical_path(workflow.tasks, index)
        node_specs = {node.node_id: node for node in inventory.nodes}
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
        profiles = _profile_catalog(self.workload_catalog)
        sampler = SyntheticSampler(
            self.workload_catalog,
            seed=seed,
            deterministic=deterministic,
        )
        completion_time: dict[str, float] = {}
        attempts_by_task: dict[str, list[AttemptRecord]] = {
            task.task_id: [] for task in workflow.tasks
        }
        target_by_task: dict[str, str] = {}
        mode_by_task: dict[str, str] = {}
        transferred_mb = 0.0
        transfer_time_ms = 0.0
        total_energy_j = 0.0
        retry_successes = 0
        epoch_sequence = 0

        self._emit(
            current_time_ms,
            "workflow_accepted",
            f"workflow {workflow.workflow_id} accepted with {len(workflow.tasks)} tasks",
            workflow.workflow_id,
        )

        while manager.unresolved():
            ready = manager.ready()
            if not ready:
                raise RuntimeError("workflow is unresolved but no task is ready")
            arrived = [task for task in ready if task.arrival_time_ms <= current_time_ms]
            if not arrived:
                current_time_ms = min(task.arrival_time_ms for task in ready)
                arrived = [task for task in ready if task.arrival_time_ms <= current_time_ms]
            inventory = await self.runtime.inventory(current_time_ms)
            node_specs = {node.node_id: node for node in inventory.nodes}
            snapshots = inventory.snapshots
            epoch_sequence += 1
            epoch_tasks = tuple(
                sorted(arrived, key=lambda item: item.task_id)
            )
            artifacts_by_task = {
                item.task_id: resolve_task_inputs(manager, item.task_id)
                for item in epoch_tasks
            }
            ready_times = {
                item.task_id: max(
                    current_time_ms,
                    item.arrival_time_ms,
                    max(
                        (
                            completion_time.get(parent, 0.0)
                            for parent in index.parents[item.task_id]
                        ),
                        default=0.0,
                    ),
                )
                for item in epoch_tasks
            }
            epoch = SchedulingEpoch(
                epoch_id=(
                    f"{workflow.workflow_id}:runtime-epoch:{epoch_sequence}"
                ),
                now_ms=current_time_ms,
                ready_tasks=epoch_tasks,
            )
            batch_plan = plan_scheduling_epoch(
                epoch,
                optimizer=algorithm,
                node_specs=node_specs,
                node_snapshots=snapshots,
                parent_artifacts=artifacts_by_task,
                ready_time_ms=ready_times,
                node_available_ms={
                    node_id: current_time_ms for node_id in node_specs
                },
                link_specs=resolved_link_specs,
                link_snapshots=resolved_link_snapshots,
                critical_tail_ms=critical_tail,
                profiles=profiles,
                registry=self.optimizer_registry,
            )
            if not batch_plan.assignments:
                raise RuntimeError(
                    "optimizer deferred every ready task; the runtime "
                    "requires at least one committable assignment per epoch"
                )
            initial_assignment = min(
                batch_plan.assignments,
                key=lambda item: (
                    item.estimated_start_ms,
                    item.estimated_finish_ms,
                    item.task_id,
                ),
            )
            task = manager.get(initial_assignment.task_id)
            manager.mark_running(task.task_id)
            input_artifacts = artifacts_by_task[task.task_id]
            current_time_ms = ready_times[task.task_id]
            self._emit(
                current_time_ms,
                "scheduling_epoch_planned",
                (
                    f"{epoch.epoch_id} considered {len(epoch_tasks)} ready "
                    f"tasks with {batch_plan.optimizer_id}"
                ),
                workflow.workflow_id,
                task_id=task.task_id,
            )
            failed_nodes: set[str] = set()
            task_finished = False

            for attempt_no in range(1, max_attempts + 1):
                attempt_id = f"{workflow.workflow_id}:{task.task_id}:attempt:{attempt_no}"
                inventory = await self.runtime.inventory(current_time_ms)
                node_specs = {node.node_id: node for node in inventory.nodes}
                snapshots = inventory.snapshots
                if attempt_no == 1:
                    assignment = initial_assignment
                else:
                    retry_epoch = SchedulingEpoch(
                        epoch_id=(
                            f"{workflow.workflow_id}:"
                            f"{task.task_id}:retry:{attempt_no}"
                        ),
                        now_ms=current_time_ms,
                        ready_tasks=(task,),
                    )
                    retry_plan = plan_scheduling_epoch(
                        retry_epoch,
                        optimizer=algorithm,
                        node_specs=node_specs,
                        node_snapshots=snapshots,
                        parent_artifacts={
                            task.task_id: input_artifacts
                        },
                        ready_time_ms={
                            task.task_id: current_time_ms
                        },
                        node_available_ms={
                            node_id: current_time_ms
                            for node_id in node_specs
                        },
                        link_specs=resolved_link_specs,
                        link_snapshots=resolved_link_snapshots,
                        critical_tail_ms={
                            task.task_id: critical_tail[task.task_id]
                        },
                        profiles=profiles,
                        excluded_node_ids={
                            task.task_id: frozenset(failed_nodes)
                        },
                        registry=self.optimizer_registry,
                    )
                    assignment = retry_plan.assignments[0]
                if not assignment.target_node_id and failed_nodes:
                    fallback_epoch = SchedulingEpoch(
                        epoch_id=(
                            f"{workflow.workflow_id}:"
                            f"{task.task_id}:retry-fallback:{attempt_no}"
                        ),
                        now_ms=current_time_ms,
                        ready_tasks=(task,),
                    )
                    fallback_plan = plan_scheduling_epoch(
                        fallback_epoch,
                        optimizer=algorithm,
                        node_specs=node_specs,
                        node_snapshots=snapshots,
                        parent_artifacts={
                            task.task_id: input_artifacts
                        },
                        ready_time_ms={
                            task.task_id: current_time_ms
                        },
                        node_available_ms={
                            node_id: current_time_ms
                            for node_id in node_specs
                        },
                        link_specs=resolved_link_specs,
                        link_snapshots=resolved_link_snapshots,
                        critical_tail_ms={
                            task.task_id: critical_tail[task.task_id]
                        },
                        profiles=profiles,
                        registry=self.optimizer_registry,
                    )
                    assignment = fallback_plan.assignments[0]
                if not assignment.target_node_id:
                    attempts_by_task[task.task_id].append(
                        AttemptRecord(
                            attempt_id,
                            attempt_no,
                            TaskState.DROPPED.value,
                            "",
                            ExecutionMode.DROP.value,
                            current_time_ms,
                            current_time_ms,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            tuple(item.artifact_id for item in input_artifacts),
                            "no_feasible_agent",
                        )
                    )
                    manager.complete(
                        task.task_id,
                        ok=False,
                        finished_time_ms=current_time_ms,
                        dropped=True,
                        error_code="no_feasible_agent",
                    )
                    completion_time[task.task_id] = current_time_ms
                    self._emit(
                        current_time_ms,
                        "task_dropped",
                        f"{task.task_id} has no feasible agent",
                        workflow.workflow_id,
                        task_id=task.task_id,
                        attempt_id=attempt_id,
                    )
                    task_finished = True
                    break

                sampled_failure = False
                error_code = ""
                try:
                    target = (
                        ExecutionTarget.EDGE
                        if node_specs[assignment.target_node_id].kind is NodeKind.EDGE
                        else ExecutionTarget.ORIN
                    )
                    sample = sampler.sample(task.spec.task_type, target)
                    assignment = replace(
                        assignment,
                        estimated_start_ms=current_time_ms,
                        estimated_finish_ms=(
                            current_time_ms + assignment.communication_ms + sample.latency_ms
                        ),
                        compute_ms=sample.latency_ms,
                        energy_j=sample.energy_j,
                    )
                    sampled_failure = sample.failed
                    if sampled_failure:
                        error_code = "synthetic_profile_failure"
                except (KeyError, UnsupportedTargetError):
                    assignment = replace(
                        assignment,
                        estimated_start_ms=current_time_ms,
                        estimated_finish_ms=(
                            current_time_ms
                            + assignment.communication_ms
                            + assignment.compute_ms
                        ),
                    )

                injected_failure = attempt_no == 1 and task.task_id in failed_once
                if injected_failure:
                    error_code = "injected_first_attempt_failure"
                try:
                    ack = await self.runtime.dispatch(
                        DispatchCommand(
                            attempt_id=attempt_id,
                            attempt_no=attempt_no,
                            task=task,
                            assignment=assignment,
                            input_artifacts=input_artifacts,
                            seed=seed,
                            inject_failure=injected_failure or sampled_failure,
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
                    failed_nodes.add(assignment.target_node_id)
                    reject_reason = ack.error_code or "dispatch_rejected"
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
                        f"{assignment.target_node_id} rejected {task.task_id}: {reject_reason}",
                        workflow.workflow_id,
                        task_id=task.task_id,
                        attempt_id=attempt_id,
                        agent_id=assignment.target_node_id,
                    )
                    if (
                        len(failed_nodes) < len(node_specs)
                        and attempt_no < max_attempts
                    ):
                        continue
                    manager.complete(
                        task.task_id,
                        ok=False,
                        finished_time_ms=current_time_ms,
                        dropped=True,
                        error_code=reject_reason,
                    )
                    completion_time[task.task_id] = current_time_ms
                    task_finished = True
                    break

                try:
                    _validate_dispatch_ack(
                        ack,
                        attempt_id=attempt_id,
                        task_id=task.task_id,
                        agent_id=assignment.target_node_id,
                    )
                except BaseException:
                    await _best_effort_cancel(
                        self.runtime,
                        attempt_id,
                        "dispatch_ack_mismatch",
                        current_time_ms,
                    )
                    raise
                self._emit(
                    current_time_ms,
                    "attempt_dispatched",
                    f"{task.task_id} attempt {attempt_no} dispatched to {ack.agent_id}",
                    workflow.workflow_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    agent_id=ack.agent_id,
                )
                try:
                    execution = await self.runtime.receive_completion(ack.dispatch_id)
                    _validate_completion(
                        execution,
                        dispatch_id=ack.dispatch_id,
                        attempt_id=attempt_id,
                        task_id=task.task_id,
                        agent_id=ack.agent_id,
                    )
                except BaseException:
                    await _best_effort_cancel(
                        self.runtime,
                        attempt_id,
                        "completion_receive_failed",
                        current_time_ms,
                    )
                    raise
                finish_time_ms = execution.finished_time_ms
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
                attempt = AttemptRecord(
                    attempt_id=attempt_id,
                    attempt_no=attempt_no,
                    state=(TaskState.SUCCEEDED.value if execution.ok else TaskState.FAILED.value),
                    target_node_id=execution.agent_id,
                    mode=assignment.execution_mode.value,
                    start_time_ms=round(current_time_ms, 4),
                    finish_time_ms=round(finish_time_ms, 4),
                    compute_time_ms=round(execution.compute_time_ms, 4),
                    communication_time_ms=round(assignment.communication_ms, 4),
                    transferred_mb=round(attempt_transfer_mb, 6),
                    energy_j=round(execution.energy_j, 6),
                    input_artifact_ids=tuple(
                        artifact.artifact_id for artifact in input_artifacts
                    ),
                    error_code="" if execution.ok else (error_code or execution.error_code),
                )
                attempts_by_task[task.task_id].append(attempt)
                transferred_mb += attempt_transfer_mb
                transfer_time_ms += assignment.communication_ms
                total_energy_j += execution.energy_j
                current_time_ms = finish_time_ms
                target_by_task[task.task_id] = execution.agent_id
                mode_by_task[task.task_id] = assignment.execution_mode.value

                if execution.ok:
                    manager.complete(
                        task.task_id,
                        ok=True,
                        finished_time_ms=finish_time_ms,
                        outputs=execution.outputs,
                    )
                    completion_time[task.task_id] = finish_time_ms
                    if attempt_no > 1:
                        retry_successes += 1
                    self._emit(
                        finish_time_ms,
                        "attempt_succeeded",
                        f"{task.task_id} attempt {attempt_no} completed on {execution.agent_id}",
                        workflow.workflow_id,
                        task_id=task.task_id,
                        attempt_id=attempt_id,
                        agent_id=execution.agent_id,
                    )
                    for output in execution.outputs:
                        self._emit(
                            finish_time_ms,
                            "artifact_published",
                            f"{output.artifact_id} published from {output.producer_port}",
                            workflow.workflow_id,
                            task_id=task.task_id,
                            attempt_id=attempt_id,
                            agent_id=execution.agent_id,
                        )
                    task_finished = True
                    break

                self._emit(
                    finish_time_ms,
                    "attempt_failed",
                    f"{task.task_id} attempt {attempt_no} failed on {execution.agent_id}",
                    workflow.workflow_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    agent_id=execution.agent_id,
                )
                failed_nodes.add(execution.agent_id)
                if attempt_no < max_attempts:
                    self._emit(
                        finish_time_ms,
                        "retry_scheduled",
                        f"{task.task_id} retry {attempt_no + 1} scheduled",
                        workflow.workflow_id,
                        task_id=task.task_id,
                        attempt_id=attempt_id,
                    )
                    continue
                manager.complete(
                    task.task_id,
                    ok=False,
                    finished_time_ms=finish_time_ms,
                    timed_out=finish_time_ms > task.deadline_time_ms,
                    error_code=attempt.error_code or "execution_failed",
                )
                completion_time[task.task_id] = finish_time_ms
                task_finished = True

            if not task_finished:
                raise RuntimeError(f"task {task.task_id} left the retry loop unresolved")

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
                    "target_node_id": target_by_task.get(task_id, ""),
                    "mode": mode_by_task.get(task_id, ""),
                    "dependencies": list(index.parents[task_id]),
                    "attempt_count": len(task_attempts),
                    "attempts": [asdict(attempt) for attempt in task_attempts],
                    "outputs": [asdict(output) for output in outputs],
                }
            )

        makespan_ms = max(completion_time.values(), default=current_time_ms)
        states = Counter(item["state"] for item in task_results)
        attempt_count = sum(len(items) for items in attempts_by_task.values())
        retry_count = sum(max(0, len(items) - 1) for items in attempts_by_task.values())
        succeeded = states[TaskState.SUCCEEDED.value]
        offloaded = sum(item["mode"] == ExecutionMode.EDGE.value for item in task_results)
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
            "success_rate": round(succeeded / max(1, len(task_results)), 4),
            "attempt_count": attempt_count,
            "retry_count": retry_count,
            "retry_success_count": retry_successes,
            "transferred_mb": round(transferred_mb, 6),
            "transfer_time_ms": round(transfer_time_ms, 4),
            "total_energy_j": round(total_energy_j, 6),
            "makespan_ms": round(makespan_ms, 4),
            "edge_offload_ratio": round(offloaded / max(1, len(task_results)), 4),
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
            },
            metrics=metrics,
            task_results=tuple(task_results),
            agents=agent_report,
            data_edges=tuple(asdict(edge) for edge in workflow.data_edges),
            events=tuple(self._events),
            logs=tuple(event.message for event in self._events),
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
                "CentralCoordinator and its RuntimePort must remain on one event loop"
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
) -> None:
    expected = (attempt_id, task_id, agent_id)
    actual = (ack.attempt_id, ack.task_id, ack.agent_id)
    if not ack.dispatch_id or actual != expected:
        raise RuntimeError(
            f"runtime returned mismatched dispatch acknowledgement: {actual!r}"
        )


def _validate_completion(
    completion: AttemptCompletion,
    *,
    dispatch_id: str,
    attempt_id: str,
    task_id: str,
    agent_id: str,
) -> None:
    expected = (dispatch_id, attempt_id, task_id, agent_id)
    actual = (
        completion.dispatch_id,
        completion.attempt_id,
        completion.task_id,
        completion.agent_id,
    )
    if actual != expected:
        raise RuntimeError(f"runtime returned mismatched completion: {actual!r}")


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


def _profile_catalog(catalog: SyntheticWorkloadCatalog) -> ProfileCatalog:
    profiles: list[ExecutionProfile] = []
    for workload in catalog:
        for target in ExecutionTarget:
            profile = workload.profile_for(target)
            profiles.append(
                ExecutionProfile(
                    task_type=workload.task_type,
                    task_class=workload.task_class,
                    node_kind=(NodeKind.ROBOT if target is ExecutionTarget.ORIN else NodeKind.EDGE),
                    model_variant=workload.model_variant,
                    input_shape="synthetic",
                    precision="synthetic",
                    batch_size=1,
                    p50_ms=profile.latency.p50_ms,
                    p95_ms=profile.latency.p95_ms,
                    p99_ms=profile.latency.p99_ms,
                    throughput_per_s=(
                        1000.0 * profile.max_concurrency / profile.latency.p50_ms
                    ),
                    peak_memory_mb=profile.resources.memory_mb,
                    energy_j=profile.energy_j.typical,
                    output_size_mb=profile.output_size_mb.typical,
                    supported=profile.supported,
                    provenance="synthetic_workload_catalog",
                )
            )
    return ProfileCatalog(profiles)
