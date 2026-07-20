"""DAG validation and authoritative MARS workflow lifecycle management."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from .models import (
    ArtifactRef,
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
        task_by_id[task.task_id] = task

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
        parents[task.task_id] = deps
        indegree[task.task_id] = len(deps)
        for parent in deps:
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
    )


class TaskManager:
    """Single source of truth for DAG readiness, results and failure propagation."""

    def __init__(self) -> None:
        self._workflow: WorkflowSpec | None = None
        self._index: DagIndex | None = None
        self._tasks: dict[str, TaskInstance] = {}
        self._states: dict[str, TaskState] = {}
        self._artifacts: dict[str, ArtifactRef] = {}
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

        if ok:
            state = TaskState.SUCCEEDED
        elif timed_out:
            state = TaskState.TIMEOUT
        elif dropped:
            state = TaskState.DROPPED
        else:
            state = TaskState.FAILED
        self._states[task_id] = state
        if artifact is not None and state is TaskState.SUCCEEDED:
            self._artifacts[task_id] = artifact
        self._completions[task_id] = TaskCompletion(
            task_id=task_id,
            ok=ok,
            state=state,
            finished_time_ms=finished_time_ms,
            artifact=artifact if ok else None,
            error_code=error_code,
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

    def artifact_for(self, task_id: str) -> ArtifactRef | None:
        return self._artifacts.get(task_id)

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
