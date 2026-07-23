"""DAG validation and MARS workflow lifecycle state management."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from .models import (
    ArtifactRef,
    DataEdge,
    FailurePolicy,
    TaskCompletion,
    TaskInstance,
    TaskState,
    TERMINAL_STATES,
    WorkflowProgress,
    WorkflowSpec,
    WorkflowState,
)


class DagValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DagIndex:
    topological_order: tuple[str, ...]
    parents: dict[str, tuple[str, ...]]
    children: dict[str, tuple[str, ...]]
    levels: dict[str, int]
    data_edges: tuple[DataEdge, ...]
    incoming_edges: dict[str, tuple[DataEdge, ...]]
    outgoing_edges: dict[str, tuple[DataEdge, ...]]


def validate_workflow(workflow: WorkflowSpec) -> DagIndex:
    """Validate the complete workflow atomically and return its graph index."""
    if not workflow.workflow_id.strip():
        raise DagValidationError("workflow_id must be non-empty")
    if not workflow.tasks:
        raise DagValidationError("workflow must contain at least one task")

    task_by_id: dict[str, TaskInstance] = {}
    for task in workflow.tasks:
        if not task.task_id.strip():
            raise DagValidationError("task_id must be non-empty")
        if task.task_id in task_by_id:
            raise DagValidationError(f"duplicate task_id: {task.task_id}")
        if task.workflow_id != workflow.workflow_id:
            raise DagValidationError(
                f"task {task.task_id} belongs to {task.workflow_id}, expected {workflow.workflow_id}"
            )
        _validate_ports(task)
        task_by_id[task.task_id] = task

    incoming_edge_lists: dict[str, list[DataEdge]] = {
        task_id: [] for task_id in task_by_id
    }
    outgoing_edge_lists: dict[str, list[DataEdge]] = {
        task_id: [] for task_id in task_by_id
    }
    seen_edges: set[tuple[str, str, str, str]] = set()
    bound_inputs: set[tuple[str, str]] = set()
    for edge in workflow.data_edges:
        _validate_data_edge(edge, task_by_id)
        identity = (
            edge.producer_task,
            edge.producer_port,
            edge.consumer_task,
            edge.consumer_port,
        )
        if identity in seen_edges:
            raise DagValidationError(
                "duplicate data edge: "
                f"{edge.producer_task}.{edge.producer_port} -> "
                f"{edge.consumer_task}.{edge.consumer_port}"
            )
        seen_edges.add(identity)
        consumer_input = (edge.consumer_task, edge.consumer_port)
        if consumer_input in bound_inputs:
            raise DagValidationError(
                "consumer input port has multiple producers: "
                f"{edge.consumer_task}.{edge.consumer_port}"
            )
        bound_inputs.add(consumer_input)
        outgoing_edge_lists[edge.producer_task].append(edge)
        incoming_edge_lists[edge.consumer_task].append(edge)

    parents: dict[str, tuple[str, ...]] = {}
    children_lists: dict[str, list[str]] = {task_id: [] for task_id in task_by_id}
    indegree: dict[str, int] = {}
    for task in workflow.tasks:
        deps = tuple(task.dependency_task_ids)
        if len(deps) != len(set(deps)):
            raise DagValidationError(f"task {task.task_id} has duplicate dependencies")
        if task.task_id in deps:
            raise DagValidationError(f"task {task.task_id} depends on itself")
        missing = [dep for dep in deps if dep not in task_by_id]
        if missing:
            raise DagValidationError(
                f"task {task.task_id} references missing dependencies: {', '.join(missing)}"
            )
        edge_parents = tuple(
            edge.producer_task for edge in incoming_edge_lists[task.task_id]
        )
        all_parents = tuple(dict.fromkeys((*deps, *edge_parents)))
        parents[task.task_id] = all_parents
        indegree[task.task_id] = len(all_parents)
        for parent in all_parents:
            children_lists[parent].append(task.task_id)

    queue = deque(task.task_id for task in workflow.tasks if indegree[task.task_id] == 0)
    order: list[str] = []
    levels: dict[str, int] = {}
    while queue:
        task_id = queue.popleft()
        order.append(task_id)
        levels[task_id] = (
            0 if not parents[task_id] else 1 + max(levels[parent] for parent in parents[task_id])
        )
        for child in children_lists[task_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(order) != len(task_by_id):
        cyclic = sorted(task_id for task_id, degree in indegree.items() if degree > 0)
        raise DagValidationError(f"workflow contains a cycle involving: {', '.join(cyclic)}")

    return DagIndex(
        topological_order=tuple(order),
        parents=parents,
        children={task_id: tuple(items) for task_id, items in children_lists.items()},
        levels=levels,
        data_edges=workflow.data_edges,
        incoming_edges={
            task_id: tuple(items) for task_id, items in incoming_edge_lists.items()
        },
        outgoing_edges={
            task_id: tuple(items) for task_id, items in outgoing_edge_lists.items()
        },
    )


def _validate_ports(task: TaskInstance) -> None:
    for direction, ports in (
        ("input", task.spec.input_ports),
        ("output", task.spec.output_ports),
    ):
        names: set[str] = set()
        for port in ports:
            if not port.name.strip():
                raise DagValidationError(
                    f"task {task.task_id} has an empty {direction} port name"
                )
            if not port.message_type.strip():
                raise DagValidationError(
                    f"task {task.task_id} port {port.name} has an empty message_type"
                )
            if port.name in names:
                raise DagValidationError(
                    f"task {task.task_id} has duplicate {direction} port: {port.name}"
                )
            names.add(port.name)


def _validate_data_edge(
    edge: DataEdge,
    task_by_id: dict[str, TaskInstance],
) -> None:
    if edge.producer_task not in task_by_id:
        raise DagValidationError(
            f"data edge references missing producer task: {edge.producer_task}"
        )
    if edge.consumer_task not in task_by_id:
        raise DagValidationError(
            f"data edge references missing consumer task: {edge.consumer_task}"
        )
    if not edge.message_type.strip():
        raise DagValidationError("data edge message_type must be non-empty")

    producer_ports = {
        port.name: port.message_type
        for port in task_by_id[edge.producer_task].spec.output_ports
    }
    consumer_ports = {
        port.name: port.message_type
        for port in task_by_id[edge.consumer_task].spec.input_ports
    }
    if edge.producer_port not in producer_ports:
        raise DagValidationError(
            f"task {edge.producer_task} has no output port: {edge.producer_port}"
        )
    if edge.consumer_port not in consumer_ports:
        raise DagValidationError(
            f"task {edge.consumer_task} has no input port: {edge.consumer_port}"
        )
    producer_type = producer_ports[edge.producer_port]
    consumer_type = consumer_ports[edge.consumer_port]
    if len({producer_type, edge.message_type, consumer_type}) != 1:
        raise DagValidationError(
            "data edge type mismatch: "
            f"{edge.producer_task}.{edge.producer_port} ({producer_type}) -> "
            f"{edge.consumer_task}.{edge.consumer_port} ({consumer_type}), "
            f"edge declares {edge.message_type}"
        )


class TaskManager:
    """Store DAG readiness, results, artifacts, and failure propagation state."""

    def __init__(self) -> None:
        self._workflow: WorkflowSpec | None = None
        self._index: DagIndex | None = None
        self._tasks: dict[str, TaskInstance] = {}
        self._states: dict[str, TaskState] = {}
        self._artifacts: dict[tuple[str, str], ArtifactRef] = {}
        self._completions: dict[str, TaskCompletion] = {}

    @property
    def workflow(self) -> WorkflowSpec:
        if self._workflow is None:
            raise RuntimeError("no workflow submitted")
        return self._workflow

    @property
    def index(self) -> DagIndex:
        if self._index is None:
            raise RuntimeError("no workflow submitted")
        return self._index

    def submit(self, workflow: WorkflowSpec) -> DagIndex:
        if self._workflow is not None:
            raise RuntimeError("TaskManager accepts one workflow; create a manager per workflow")
        index = validate_workflow(workflow)
        self._workflow = workflow
        self._index = index
        self._tasks = {task.task_id: task for task in workflow.tasks}
        self._states = {
            task.task_id: TaskState.READY if not index.parents[task.task_id] else TaskState.BLOCKED
            for task in workflow.tasks
        }
        return index

    def get(self, task_id: str) -> TaskInstance:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task_id: {task_id}") from exc

    def state_of(self, task_id: str) -> TaskState:
        self.get(task_id)
        return self._states[task_id]

    def ready(self) -> list[TaskInstance]:
        return [
            self._tasks[task_id]
            for task_id in self.index.topological_order
            if self._states[task_id] is TaskState.READY
        ]

    def mark_running(self, task_id: str) -> None:
        if self.state_of(task_id) is not TaskState.READY:
            raise ValueError(f"task {task_id} is not ready")
        self._states[task_id] = TaskState.RUNNING

    def complete(
        self,
        task_id: str,
        *,
        ok: bool,
        finished_time_ms: float,
        artifact: ArtifactRef | None = None,
        outputs: tuple[ArtifactRef, ...] = (),
        timed_out: bool = False,
        dropped: bool = False,
        error_code: str = "",
    ) -> tuple[list[str], list[str]]:
        """Record one terminal result and return (newly_ready, newly_skipped)."""
        current = self.state_of(task_id)
        if current in TERMINAL_STATES:
            return [], []
        if current is not TaskState.RUNNING:
            raise ValueError(f"task {task_id} must be running before completion")

        normalized_outputs = self._normalize_outputs(task_id, artifact, outputs) if ok else ()

        if ok:
            state = TaskState.SUCCEEDED
        elif timed_out:
            state = TaskState.TIMEOUT
        elif dropped:
            state = TaskState.DROPPED
        else:
            state = TaskState.FAILED
        self._states[task_id] = state
        if state is TaskState.SUCCEEDED:
            for output in normalized_outputs:
                self._artifacts[(task_id, output.producer_port)] = output
        self._completions[task_id] = TaskCompletion(
            task_id=task_id,
            ok=ok,
            state=state,
            finished_time_ms=finished_time_ms,
            artifact=normalized_outputs[0] if len(normalized_outputs) == 1 else None,
            error_code=error_code,
            outputs=normalized_outputs,
        )

        if state is TaskState.SUCCEEDED:
            released: list[str] = []
            for child in self.index.children[task_id]:
                if self._states[child] is not TaskState.BLOCKED:
                    continue
                if all(self._states[parent] is TaskState.SUCCEEDED for parent in self.index.parents[child]):
                    self._states[child] = TaskState.READY
                    released.append(child)
            return released, []

        if self.workflow.failure_policy is FailurePolicy.FAIL_FAST:
            candidates = [
                candidate
                for candidate in self.index.topological_order
                if self._states[candidate] not in TERMINAL_STATES
            ]
        else:
            candidates = self._descendants(task_id)
        skipped: list[str] = []
        for candidate in candidates:
            if self._states[candidate] in {TaskState.BLOCKED, TaskState.READY, TaskState.RUNNING}:
                self._states[candidate] = TaskState.SKIPPED
                skipped.append(candidate)
        return [], skipped

    def artifact_for(
        self,
        task_id: str,
        producer_port: str | None = None,
    ) -> ArtifactRef | None:
        """Return one output, preserving the legacy single-artifact lookup.

        Callers handling multi-output tasks must name a producer port or use
        :meth:`artifacts_for`.
        """
        self.get(task_id)
        if producer_port is not None:
            return self._artifacts.get((task_id, producer_port))
        outputs = self.artifacts_for(task_id)
        if len(outputs) > 1:
            raise ValueError(
                f"task {task_id} has multiple outputs; specify producer_port"
            )
        return outputs[0] if outputs else None

    def artifacts_for(
        self,
        task_id: str,
        producer_port: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        self.get(task_id)
        completion = self._completions.get(task_id)
        if completion is None:
            return ()
        if producer_port is None:
            return completion.outputs
        artifact = self._artifacts.get((task_id, producer_port))
        return (artifact,) if artifact is not None else ()

    def input_artifacts_for(self, task_id: str) -> tuple[ArtifactRef, ...]:
        """Resolve typed data-edge inputs, retaining shared refs for fan-out."""
        self.get(task_id)
        resolved: list[ArtifactRef] = []
        for edge in self.index.incoming_edges[task_id]:
            artifact = self._artifacts.get((edge.producer_task, edge.producer_port))
            if artifact is None:
                raise RuntimeError(
                    "input artifact is not available: "
                    f"{edge.producer_task}.{edge.producer_port}"
                )
            resolved.append(artifact)
        return tuple(resolved)

    def completion_of(self, task_id: str) -> TaskCompletion | None:
        return self._completions.get(task_id)

    def unresolved(self) -> list[TaskInstance]:
        return [task for task_id, task in self._tasks.items() if self._states[task_id] not in TERMINAL_STATES]

    def workflow_state(self) -> WorkflowState:
        states = tuple(self._states.values())
        if states and all(state is TaskState.SUCCEEDED for state in states):
            return WorkflowState.SUCCEEDED
        if states and all(state in TERMINAL_STATES for state in states):
            return WorkflowState.FAILED
        if any(state is TaskState.RUNNING for state in states):
            return WorkflowState.RUNNING
        return WorkflowState.ACCEPTED

    def progress(self, critical_path: tuple[str, ...] = ()) -> WorkflowProgress:
        counts = Counter(state.value for state in self._states.values())
        return WorkflowProgress(
            workflow_id=self.workflow.workflow_id,
            state=self.workflow_state(),
            total_tasks=len(self._tasks),
            state_counts=dict(counts),
            ready_task_ids=tuple(task.task_id for task in self.ready()),
            critical_path=critical_path,
        )

    def _descendants(self, task_id: str) -> list[str]:
        seen: set[str] = set()
        queue = deque(self.index.children[task_id])
        while queue:
            child = queue.popleft()
            if child in seen:
                continue
            seen.add(child)
            queue.extend(self.index.children[child])
        return [candidate for candidate in self.index.topological_order if candidate in seen]
    def _normalize_outputs(
        self,
        task_id: str,
        artifact: ArtifactRef | None,
        outputs: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        if artifact is not None and outputs:
            if len(outputs) != 1 or outputs[0] != artifact:
                raise ValueError("artifact and outputs describe different task outputs")
        normalized = tuple(outputs) if outputs else ((artifact,) if artifact is not None else ())

        declared_ports = {
            port.name: port.message_type for port in self.get(task_id).spec.output_ports
        }
        seen_ports: set[str] = set()
        for output in normalized:
            if output.producer_task_id != task_id:
                raise ValueError(
                    f"artifact {output.artifact_id} belongs to "
                    f"{output.producer_task_id}, expected {task_id}"
                )
            if output.producer_port in seen_ports:
                raise ValueError(
                    f"task {task_id} produced duplicate output port: {output.producer_port}"
                )
            seen_ports.add(output.producer_port)
            if declared_ports:
                expected_type = declared_ports.get(output.producer_port)
                if expected_type is None:
                    raise ValueError(
                        f"task {task_id} produced undeclared output port: "
                        f"{output.producer_port}"
                    )
                if output.message_type != expected_type:
                    raise ValueError(
                        f"artifact {output.artifact_id} type mismatch for "
                        f"{task_id}.{output.producer_port}: expected {expected_type}, "
                        f"got {output.message_type or '<empty>'}"
                    )

        required_ports = {
            edge.producer_port for edge in self.index.outgoing_edges[task_id]
        }
        missing = sorted(required_ports - seen_ports)
        if missing:
            raise ValueError(
                f"task {task_id} completion is missing required output ports: "
                f"{', '.join(missing)}"
            )
        return normalized


def resolve_task_inputs(
    manager: TaskManager,
    task_id: str,
) -> tuple[ArtifactRef, ...]:
    """Resolve every materialized and external input for one ready task.

    Typed ``DataEdge`` bindings select the producer artifact for a consumer
    port. Legacy dependency-only edges contribute all producer artifacts.
    Unbound declared input ports represent source data and receive a
    proportional share of ``input_size_mb``.
    """

    task = manager.get(task_id)
    artifacts = list(manager.input_artifacts_for(task_id))
    typed_parents = {
        edge.producer_task
        for edge in manager.index.incoming_edges[task_id]
    }
    for parent in manager.index.parents[task_id]:
        if parent not in typed_parents:
            artifacts.extend(manager.artifacts_for(parent))

    bound_ports = {
        edge.consumer_port
        for edge in manager.index.incoming_edges[task_id]
    }
    if task.spec.input_ports:
        unbound_count = sum(
            port.name not in bound_ports
            for port in task.spec.input_ports
        )
        external_size_mb = (
            task.spec.input_size_mb
            * unbound_count
            / len(task.spec.input_ports)
        )
    else:
        external_size_mb = (
            task.spec.input_size_mb if not artifacts else 0.0
        )

    if external_size_mb > 0:
        artifacts.append(
            ArtifactRef(
                artifact_id=(
                    f"input:{task.workflow_id}:{task.task_id}"
                ),
                producer_task_id="",
                node_id=task.source_node_id,
                size_mb=external_size_mb,
                uri=(
                    f"source://{task.source_node_id}/"
                    f"{task.workflow_id}/{task.task_id}"
                ),
                producer_port="external_input",
                message_type="external_input_batch",
            )
        )
    return tuple(artifacts)
