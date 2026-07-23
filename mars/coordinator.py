"""Central MARS coordinator for local multi-agent workflow exercises."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Iterable

from .agents import AgentSession
from .dag import TaskManager
from .models import (
    ArtifactRef,
    Assignment,
    ExecutionMode,
    NodeKind,
    TaskInstance,
    TaskState,
    WorkflowSpec,
)
from .profiling import ExecutionProfile, ProfileCatalog
from .scheduler import choose_assignment, critical_path
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
    """Register agents and execute one DAG through the central control plane.

    The implementation advances a virtual clock rather than sleeping. Agent
    sessions still receive explicit reservations, inputs, completions, and
    resource-release calls. The session contract is independent of its process
    or network implementation.
    """

    def __init__(
        self,
        agents: Iterable[AgentSession],
        *,
        workload_catalog: SyntheticWorkloadCatalog | None = None,
    ) -> None:
        self.agents = tuple(agents)
        self.agent_by_id = {agent.node_spec.node_id: agent for agent in self.agents}
        if not self.agents:
            raise ValueError("at least one agent is required")
        if len(self.agent_by_id) != len(self.agents):
            raise ValueError("agent node ids must be unique")
        self.workload_catalog = workload_catalog or load_default_synthetic_workloads()
        self._events: list[RuntimeEvent] = []
        self._sequence = 0

    def describe(self) -> dict[str, object]:
        return {
            "scheduler_id": "mars-central",
            "status": "online",
            "agent_count": len(self.agents),
            "agents": [agent.describe(0.0) for agent in self.agents],
        }

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
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self._events = []
        self._sequence = 0
        current_time_ms = 0.0
        failed_once = frozenset(fail_first_task_ids)

        for agent in self.agents:
            agent.reset()
            agent.register(current_time_ms)
            heartbeat = agent.heartbeat(current_time_ms)
            self._emit(
                current_time_ms,
                "agent_registered",
                f"{agent.node_spec.node_id} registered and heartbeat {heartbeat.sequence} received",
                workflow.workflow_id,
                agent_id=agent.node_spec.node_id,
            )

        manager = TaskManager()
        index = manager.submit(workflow)
        critical_ids, critical_path_ms, critical_tail = critical_path(workflow.tasks, index)
        node_specs = {agent.node_spec.node_id: agent.node_spec for agent in self.agents}
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
            task = min(
                arrived,
                key=lambda item: (
                    item.deadline_time_ms - critical_tail[item.task_id],
                    -item.priority,
                    item.task_id,
                ),
            )
            manager.mark_running(task.task_id)
            input_artifacts = _input_artifacts(manager, task.task_id)
            parent_finished = max(
                (completion_time.get(parent, 0.0) for parent in index.parents[task.task_id]),
                default=0.0,
            )
            current_time_ms = max(current_time_ms, task.arrival_time_ms, parent_finished)
            failed_nodes: set[str] = set()
            task_finished = False

            for attempt_no in range(1, max_attempts + 1):
                attempt_id = f"{workflow.workflow_id}:{task.task_id}:attempt:{attempt_no}"
                node_available = {node_id: current_time_ms for node_id in node_specs}
                for node_id in failed_nodes:
                    node_available[node_id] = current_time_ms + 1_000_000_000.0
                snapshots = {
                    agent.node_spec.node_id: agent.snapshot for agent in self.agents
                }
                assignment = choose_assignment(
                    task,
                    algorithm=algorithm,
                    ready_time_ms=current_time_ms,
                    node_available=node_available,
                    node_specs=node_specs,
                    node_snapshots=snapshots,
                    parent_artifacts=input_artifacts,
                    critical_tail_ms=critical_tail[task.task_id],
                    profiles=profiles,
                    excluded_node_ids=frozenset(failed_nodes),
                )
                if not assignment.target_node_id and failed_nodes:
                    assignment = choose_assignment(
                        task,
                        algorithm=algorithm,
                        ready_time_ms=current_time_ms,
                        node_available={
                            node_id: current_time_ms for node_id in node_specs
                        },
                        node_specs=node_specs,
                        node_snapshots=snapshots,
                        parent_artifacts=input_artifacts,
                        critical_tail_ms=critical_tail[task.task_id],
                        profiles=profiles,
                    )
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

                agent = self.agent_by_id[assignment.target_node_id]
                can_execute, reject_reason = agent.can_execute(task)
                reservation = agent.reserve(
                    task,
                    attempt_id,
                    assignment.estimated_start_ms,
                ) if can_execute else None
                if reservation is None:
                    failed_nodes.add(assignment.target_node_id)
                    self._emit(
                        current_time_ms,
                        "dispatch_rejected",
                        f"{assignment.target_node_id} rejected {task.task_id}: {reject_reason or 'resources_unavailable'}",
                        workflow.workflow_id,
                        task_id=task.task_id,
                        attempt_id=attempt_id,
                        agent_id=assignment.target_node_id,
                    )
                    if (
                        len(failed_nodes) < len(self.agents)
                        and attempt_no < max_attempts
                    ):
                        continue
                    manager.complete(
                        task.task_id,
                        ok=False,
                        finished_time_ms=current_time_ms,
                        dropped=True,
                        error_code=reject_reason or "resources_unavailable",
                    )
                    task_finished = True
                    break

                sampled_failure = False
                error_code = ""
                try:
                    target = (
                        ExecutionTarget.EDGE
                        if agent.node_spec.kind is NodeKind.EDGE
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
                self._emit(
                    current_time_ms,
                    "attempt_dispatched",
                    f"{task.task_id} attempt {attempt_no} dispatched to {agent.node_spec.node_id}",
                    workflow.workflow_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    agent_id=agent.node_spec.node_id,
                )
                execution = agent.execute(
                    task,
                    assignment,
                    reservation,
                    input_artifacts,
                    seed=seed,
                    attempt_no=attempt_no,
                    inject_failure=injected_failure or sampled_failure,
                )
                finish_time_ms = (
                    current_time_ms
                    + assignment.communication_ms
                    + execution.compute_time_ms
                )
                agent.release(
                    reservation.reservation_id,
                    finish_time_ms,
                    ok=execution.ok,
                )
                agent.heartbeat(finish_time_ms)
                attempt_transfer_mb = sum(
                    artifact.size_mb
                    for artifact in input_artifacts
                    if artifact.node_id != agent.node_spec.node_id
                )
                if (
                    not input_artifacts
                    and agent.node_spec.node_id != task.source_node_id
                ):
                    attempt_transfer_mb = task.spec.input_size_mb
                attempt = AttemptRecord(
                    attempt_id=attempt_id,
                    attempt_no=attempt_no,
                    state=(TaskState.SUCCEEDED.value if execution.ok else TaskState.FAILED.value),
                    target_node_id=agent.node_spec.node_id,
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
                target_by_task[task.task_id] = agent.node_spec.node_id
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
                        f"{task.task_id} attempt {attempt_no} completed on {agent.node_spec.node_id}",
                        workflow.workflow_id,
                        task_id=task.task_id,
                        attempt_id=attempt_id,
                        agent_id=agent.node_spec.node_id,
                    )
                    for output in execution.outputs:
                        self._emit(
                            finish_time_ms,
                            "artifact_published",
                            f"{output.artifact_id} published from {output.producer_port}",
                            workflow.workflow_id,
                            task_id=task.task_id,
                            attempt_id=attempt_id,
                            agent_id=agent.node_spec.node_id,
                        )
                    task_finished = True
                    break

                self._emit(
                    finish_time_ms,
                    "attempt_failed",
                    f"{task.task_id} attempt {attempt_no} failed on {agent.node_spec.node_id}",
                    workflow.workflow_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    agent_id=agent.node_spec.node_id,
                )
                failed_nodes.add(agent.node_spec.node_id)
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
            item["task_class"] == "local_safety"
            and bool(item["target_node_id"])
            and item["target_node_id"] != item["source_node_id"]
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
            agents=tuple(agent.describe(makespan_ms) for agent in self.agents),
            data_edges=tuple(asdict(edge) for edge in workflow.data_edges),
            events=tuple(self._events),
            logs=tuple(event.message for event in self._events),
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


def _input_artifacts(manager: TaskManager, task_id: str) -> tuple[ArtifactRef, ...]:
    task = manager.get(task_id)
    artifacts = list(manager.input_artifacts_for(task_id))
    typed_parents = {
        edge.producer_task for edge in manager.index.incoming_edges[task_id]
    }
    for parent in manager.index.parents[task_id]:
        if parent not in typed_parents:
            artifacts.extend(manager.artifacts_for(parent))
    bound_ports = {
        edge.consumer_port for edge in manager.index.incoming_edges[task_id]
    }
    if task.spec.input_ports:
        unbound_count = sum(
            port.name not in bound_ports for port in task.spec.input_ports
        )
        external_size_mb = (
            task.spec.input_size_mb
            * unbound_count
            / len(task.spec.input_ports)
        )
    else:
        external_size_mb = task.spec.input_size_mb if not artifacts else 0.0
    if external_size_mb > 0:
        artifacts.append(
            ArtifactRef(
                artifact_id=f"input:{task.workflow_id}:{task.task_id}",
                producer_task_id="",
                node_id=task.source_node_id,
                size_mb=external_size_mb,
                uri=f"source://{task.source_node_id}/{task.workflow_id}/{task.task_id}",
                producer_port="external_input",
                message_type="external_input_batch",
            )
        )
    return tuple(artifacts)


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
