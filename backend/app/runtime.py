"""Process-local runtime service for the three-agent MARS demonstration."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

from mars.coordinator import CentralCoordinator, CoordinatorReport
from mars.runtime import InProcessRuntime

from .mars_adapter import build_node_snapshots, build_node_specs, build_workflow
from .scene_generator import build_deterministic_scene
from .schemas import GenerateSceneRequest, RuntimeWorkflowRequest


@dataclass
class RuntimeRun:
    run_id: str
    workflow_id: str
    status: str
    future: Future[CoordinatorReport] | None = None
    result: dict[str, object] | None = None
    error: str = ""


class LocalRuntimeService:
    """Own one central scheduler and its process-local runtime adapter."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mars-runtime")
        self._coordinator: CentralCoordinator | None = None
        self._runs: dict[str, RuntimeRun] = {}

    def bootstrap(self) -> dict[str, object]:
        with self._lock:
            if self._coordinator is None:
                scene = build_deterministic_scene(
                    GenerateSceneRequest(
                        robot_count=2,
                        edge_count=1,
                        use_llm=False,
                        seed=7,
                    )
                )
                coordinator = CentralCoordinator(_runtime_for_scene(scene))
                asyncio.run(coordinator.initialize_async())
                self._coordinator = coordinator
            return self._runtime_view_locked()

    def status(self) -> dict[str, object]:
        self.bootstrap()
        with self._lock:
            self._refresh_all_locked()
            return self._runtime_view_locked()

    def submit(self, request: RuntimeWorkflowRequest) -> dict[str, object]:
        _validate_demo_topology(request)
        coordinator = CentralCoordinator(_runtime_for_scene(request.scene))
        workflow = build_workflow(request.scene)
        failure_ids: tuple[str, ...] = ()
        if request.inject_first_failure:
            selected = next(
                (
                    task.task_id
                    for task in workflow.tasks
                    if task.spec.task_type == request.failure_task_type
                ),
                workflow.tasks[0].task_id,
            )
            failure_ids = (selected,)

        run_id = f"run_{uuid4().hex[:12]}"
        run = RuntimeRun(run_id, workflow.workflow_id, "accepted")
        with self._lock:
            self._coordinator = coordinator
            self._runs[run_id] = run
            run.status = "running"
            run.future = self._executor.submit(
                coordinator.run,
                workflow,
                algorithm=request.algorithm,
                seed=request.seed,
                max_attempts=request.max_attempts,
                fail_first_task_ids=failure_ids,
                deterministic=request.deterministic,
            )
        return {
            "run_id": run_id,
            "workflow_id": workflow.workflow_id,
            "status": "running",
        }

    def get_run(self, run_id: str) -> dict[str, object] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            self._refresh_locked(run)
            return {
                "run_id": run.run_id,
                "workflow_id": run.workflow_id,
                "status": run.status,
                "result": run.result,
                "error": run.error,
            }

    def events(self, run_id: str, after_sequence: int = 0) -> dict[str, object] | None:
        payload = self.get_run(run_id)
        if payload is None:
            return None
        result = payload.get("result") or {}
        events = [
            event
            for event in result.get("events", [])
            if int(event["sequence"]) > after_sequence
        ]
        return {
            "run_id": run_id,
            "status": payload["status"],
            "events": events,
            "next_sequence": max(
                [after_sequence, *(int(event["sequence"]) for event in events)]
            ),
        }

    def _refresh_all_locked(self) -> None:
        for run in self._runs.values():
            self._refresh_locked(run)

    def _refresh_locked(self, run: RuntimeRun) -> None:
        if run.future is None or not run.future.done() or run.status in {"succeeded", "failed"}:
            return
        try:
            report = run.future.result()
        except Exception as exc:  # surfaced through the run status endpoint
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            return
        run.result = report.as_dict()
        run.status = (
            "succeeded"
            if report.workflow.get("state") == "succeeded"
            else "failed"
        )

    def _runtime_view_locked(self) -> dict[str, object]:
        coordinator = self._coordinator
        assert coordinator is not None
        view = coordinator.describe()
        view.update(
            {
                "runtime": "process_local_virtual_time",
                "topology": {
                    "central_schedulers": 1,
                    "orin_agents": 2,
                    "edge_agents": 1,
                },
                "run_count": len(self._runs),
            }
        )
        return view


def _runtime_for_scene(scene) -> InProcessRuntime:
    specs = build_node_specs(scene)
    return InProcessRuntime(
        specs,
        build_node_snapshots(scene),
        max_concurrency={
            spec.node_id: 2 if spec.kind.value == "robot" else 4
            for spec in specs
        },
    )


def _validate_demo_topology(request: RuntimeWorkflowRequest) -> None:
    robots = [node for node in request.scene.nodes if node.kind == "robot"]
    edges = [node for node in request.scene.nodes if node.kind == "edge"]
    clouds = [node for node in request.scene.nodes if node.kind == "cloud"]
    if len(robots) != 2 or len(edges) != 1 or clouds:
        raise ValueError(
            "the local runtime demo requires exactly two Orin robot nodes and one edge node"
        )


runtime_service = LocalRuntimeService()
