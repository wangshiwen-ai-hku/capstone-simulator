"""Process-local runtime service for MARS robot and edge agents."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Lock
from uuid import uuid4

from evals.workflow import evaluate_run_artifact
from mars.coordinator import CentralCoordinator, CoordinatorReport
from mars.domain.topology import LinkSnapshot
from mars.optimizers import OptimizerRegistry
from mars.run_artifact import RunArtifact, build_run_artifact
from mars.runtime import InProcessRuntime

from .mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from .scene_generator import build_deterministic_scene
from .schemas import GenerateSceneRequest, RuntimeWorkflowRequest
from .scheduling import SchedulingConfiguration, configure_scheduling
from .trace_archive import TraceSession


@dataclass
class RuntimeRun:
    run_id: str
    workflow_id: str
    status: str
    future: Future[CoordinatorReport] | None = None
    artifact: RunArtifact | None = None
    result: dict[str, object] | None = None
    error: str = ""
    trace_session: TraceSession | None = None
    trace_archived: bool = False


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
                coordinator = coordinator_for_scene(scene)
                asyncio.run(coordinator.initialize_async())
                self._coordinator = coordinator
            return self._runtime_view_locked()

    def status(self) -> dict[str, object]:
        self.bootstrap()
        with self._lock:
            self._refresh_all_locked()
            return self._runtime_view_locked()

    def submit(
        self,
        request: RuntimeWorkflowRequest,
        *,
        trace_session: TraceSession | None = None,
    ) -> dict[str, object]:
        _validate_runtime_topology(request)
        scheduling = configure_scheduling(
            request.algorithm,
            request.optimizer_options,
            formulation=request.formulation,
            legacy_beta=request.model_dump(include={"beta"}).get("beta"),
        )
        coordinator = coordinator_for_scene(
            request.scene,
            optimizer_registry=scheduling.registry,
            fallback_optimizer=scheduling.fallback_optimizer,
        )
        workflow = build_workflow(request.scene)
        failure_ids: tuple[str, ...] = ()
        if request.inject_first_failure:
            selected = next(
                (
                    task.task_id
                    for task in workflow.tasks
                    if task.spec.task_type == request.failure_task_type
                ),
                None,
            )
            if selected is None:
                raise ValueError(
                    "failure injection target task type is not present in "
                    f"the workflow: {request.failure_task_type}"
                )
            failure_ids = (selected,)

        run_id = f"run_{uuid4().hex[:12]}"
        run = RuntimeRun(
            run_id,
            workflow.workflow_id,
            "accepted",
            trace_session=trace_session,
        )
        if trace_session is not None:
            trace_session.write_json(
                "run.json",
                {"run_id": run_id, "workflow_id": workflow.workflow_id},
            )
        with self._lock:
            self._runs[run_id] = run
            future = self._executor.submit(
                self._execute_run,
                run_id,
                coordinator,
                workflow,
                request,
                failure_ids,
                scheduling,
            )
            run.future = future
        # Finalization cannot depend on a client polling the status endpoint.
        # Register outside the lock because add_done_callback() executes
        # synchronously when an exceptionally fast future is already done.
        future.add_done_callback(
            lambda _future: self._finalize_run(run_id)
        )
        payload = {
            "run_id": run_id,
            "workflow_id": workflow.workflow_id,
            "status": "accepted",
        }
        if trace_session is not None:
            payload["trace_id"] = trace_session.trace_id
            payload["trace_directory"] = str(trace_session.directory)
        return payload

    def _execute_run(
        self,
        run_id: str,
        coordinator: CentralCoordinator,
        workflow,
        request: RuntimeWorkflowRequest,
        failure_ids: tuple[str, ...],
        scheduling: SchedulingConfiguration,
    ) -> CoordinatorReport:
        with self._lock:
            run = self._runs[run_id]
            run.status = "running"
            self._coordinator = coordinator
        report = coordinator.run(
            workflow,
            algorithm=request.algorithm,
            formulation=scheduling.formulation,
            seed=request.seed,
            max_attempts=request.max_attempts,
            fail_first_task_ids=failure_ids,
            deterministic=request.deterministic,
        )
        artifact = build_run_artifact(
            run_id=run_id,
            workflow=workflow,
            node_specs=build_node_specs(request.scene),
            node_snapshots=build_node_snapshots(request.scene),
            link_specs=build_link_specs(request.scene),
            link_snapshots=build_link_snapshots(request.scene),
            profiles=coordinator.profile_catalog.profiles,
            raw_report=report,
            algorithm=request.algorithm,
            formulation=scheduling.formulation,
            seed=request.seed,
            deterministic=request.deterministic,
            max_attempts=request.max_attempts,
            network_jitter=0.0,
            resource_noise=0.04,
        )
        # Preserve the completed factual run before any post-run evaluator
        # runs.  Evaluation is deliberately downstream of the artifact and a
        # failure there must not discard otherwise complete execution evidence.
        with self._lock:
            self._runs[run_id].artifact = artifact
        evaluation = evaluate_run_artifact(
            artifact,
            weights=scheduling.evaluation_weights,
        )
        return replace(
            report,
            metrics={**report.metrics, **evaluation.as_dict()},
            workflow={
                **report.workflow,
                "requested_algorithm": request.algorithm,
                "formulation": scheduling.formulation,
                "optimizer_options": dict(scheduling.optimizer_options),
                "metric_schema_version": evaluation.schema_version,
            },
        )

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

    def _finalize_run(self, run_id: str) -> None:
        """Commit terminal state and traces as soon as execution completes."""

        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                self._refresh_locked(run)

    def _refresh_locked(self, run: RuntimeRun) -> None:
        if run.future is None or not run.future.done() or run.status in {"succeeded", "failed"}:
            return
        try:
            report = run.future.result()
        except Exception as exc:  # surfaced through the run status endpoint
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            self._archive_run_trace(run)
            return
        run.result = report.as_dict()
        run.status = (
            "succeeded"
            if report.workflow.get("state") == "succeeded"
            else "failed"
        )
        self._archive_run_trace(run)

    def _archive_run_trace(self, run: RuntimeRun) -> None:
        session = run.trace_session
        if session is None or run.trace_archived:
            return
        if run.result is not None:
            session.write_response(run.result)
        if run.artifact is not None:
            session.write_json(
                "run_artifact.json",
                run.artifact.as_dict(),
            )
        if run.error:
            session.write_json("error.json", {"error": run.error})
        session.write_json(
            "status.json",
            {
                "run_id": run.run_id,
                "workflow_id": run.workflow_id,
                "status": run.status,
            },
        )
        run.trace_archived = True

    def _runtime_view_locked(self) -> dict[str, object]:
        coordinator = self._coordinator
        assert coordinator is not None
        view = coordinator.describe()
        agents = list(view.get("agents", []))
        kind_counts: dict[str, int] = {}
        for agent in agents:
            kind = str(agent.get("kind", "unknown"))
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        view.update(
            {
                "runtime": "process_local_virtual_time",
                "topology": {
                    "central_schedulers": 1,
                    "orin_agents": kind_counts.get("robot", 0),
                    "edge_agents": kind_counts.get("edge", 0),
                    "cloud_agents": kind_counts.get("cloud", 0),
                    "total_agents": len(agents),
                },
                "run_count": len(self._runs),
            }
        )
        return view


def runtime_for_scene(
    scene,
    *,
    execution_noise: float = 0.04,
    respect_expected_accuracy: bool = False,
) -> InProcessRuntime:
    """Build a process-local runtime adapter for one declared topology."""

    validate_process_local_scene(scene)
    specs = build_node_specs(scene)
    return InProcessRuntime(
        specs,
        build_node_snapshots(scene),
        execution_noise=execution_noise,
        respect_expected_accuracy=respect_expected_accuracy,
    )


def coordinator_for_scene(
    scene,
    *,
    execution_noise: float = 0.04,
    respect_expected_accuracy: bool = False,
    link_snapshots: tuple[LinkSnapshot, ...] | None = None,
    optimizer_registry: OptimizerRegistry | None = None,
    fallback_optimizer: str | None = "heuristic",
) -> CentralCoordinator:
    """Build a CentralCoordinator with its RuntimePort implementation."""

    return CentralCoordinator(
        runtime_for_scene(
            scene,
            execution_noise=execution_noise,
            respect_expected_accuracy=respect_expected_accuracy,
        ),
        link_specs=build_link_specs(scene),
        link_snapshots=(
            build_link_snapshots(scene)
            if link_snapshots is None
            else link_snapshots
        ),
        optimizer_registry=optimizer_registry,
        fallback_optimizer=fallback_optimizer,
    )

def _validate_runtime_topology(request: RuntimeWorkflowRequest) -> None:
    validate_process_local_scene(request.scene)


def validate_process_local_scene(scene) -> None:
    """Reject node kinds not implemented by the process-local adapter."""

    clouds = [node for node in scene.nodes if node.kind == "cloud"]
    if clouds:
        raise ValueError(
            "the process-local runtime supports robot and edge nodes; "
            "cloud nodes are not supported"
        )


runtime_service = LocalRuntimeService()
