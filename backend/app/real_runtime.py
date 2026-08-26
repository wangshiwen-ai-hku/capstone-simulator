"""Background workflow service backed by localhost gRPC agents."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Lock
from uuid import uuid4

from evals.workflow import evaluate_run_artifact
from mars.coordinator import CentralCoordinator, CoordinatorReport
from mars.run_artifact import build_run_artifact
from mars.runtime import GrpcRuntimeAdapter

from .config import get_settings
from .mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_workflow,
)
from .runtime import RuntimeRun
from .scheduling import SchedulingConfiguration, configure_scheduling
from .schemas import RuntimeWorkflowRequest
from .trace_archive import TraceSession


class RealRuntimeService:
    """Run workflows against the statically configured gRPC agents."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mars-real-runtime",
        )
        self._runs: dict[str, RuntimeRun] = {}
        self._agents: tuple[dict[str, object], ...] = ()

    @property
    def endpoints(self) -> dict[str, str]:
        return get_settings().real_agent_endpoint_map()

    def bootstrap(self) -> dict[str, object]:
        agents = asyncio.run(self._inspect_agents())
        with self._lock:
            self._agents = agents
            return self._view_locked()

    def status(self) -> dict[str, object]:
        with self._lock:
            self._refresh_all_locked()
            return self._view_locked()

    def submit(
        self,
        request: RuntimeWorkflowRequest,
        *,
        trace_session: TraceSession | None = None,
    ) -> dict[str, object]:
        endpoints = self.endpoints
        scene_nodes = {node.id for node in request.scene.nodes}
        if scene_nodes != set(endpoints):
            raise ValueError(
                "real runtime scene nodes must match configured agents: "
                f"expected {sorted(endpoints)!r}, got {sorted(scene_nodes)!r}"
            )
        scheduling = configure_scheduling(
            request.algorithm,
            request.optimizer_options,
            formulation=request.formulation,
            legacy_beta=request.model_dump(include={"beta"}).get("beta"),
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

        run_id = f"real_{uuid4().hex[:12]}"
        run = RuntimeRun(
            run_id,
            workflow.workflow_id,
            "accepted",
            trace_session=trace_session,
        )
        with self._lock:
            self._runs[run_id] = run
            future = self._executor.submit(
                self._execute_run,
                run_id,
                workflow,
                request,
                failure_ids,
                scheduling,
                endpoints,
            )
            run.future = future
        future.add_done_callback(lambda _future: self._finalize_run(run_id))
        return {
            "run_id": run_id,
            "workflow_id": workflow.workflow_id,
            "status": "accepted",
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

    def events(
        self,
        run_id: str,
        after_sequence: int = 0,
    ) -> dict[str, object] | None:
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

    async def _inspect_agents(self) -> tuple[dict[str, object], ...]:
        runtime = GrpcRuntimeAdapter(self.endpoints)
        try:
            await runtime.start(0.0)
            return await runtime.describe(0.0)
        finally:
            await runtime.close()

    def _execute_run(
        self,
        run_id: str,
        workflow,
        request: RuntimeWorkflowRequest,
        failure_ids: tuple[str, ...],
        scheduling: SchedulingConfiguration,
        endpoints: dict[str, str],
    ) -> CoordinatorReport:
        runtime = GrpcRuntimeAdapter(endpoints)
        coordinator = CentralCoordinator(
            runtime,
            link_specs=build_link_specs(request.scene),
            link_snapshots=build_link_snapshots(request.scene),
            optimizer_registry=scheduling.registry,
            fallback_optimizer=scheduling.fallback_optimizer,
        )
        with self._lock:
            self._runs[run_id].status = "running"
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
            node_specs=runtime.nodes,
            node_snapshots=runtime.snapshots,
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
            resource_noise=0.0,
        )
        with self._lock:
            self._runs[run_id].artifact = artifact
            self._agents = tuple(report.agents)
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

    def _finalize_run(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                self._refresh_locked(run)

    def _refresh_all_locked(self) -> None:
        for run in self._runs.values():
            self._refresh_locked(run)

    def _refresh_locked(self, run: RuntimeRun) -> None:
        if (
            run.future is None
            or not run.future.done()
            or run.status in {"succeeded", "failed"}
        ):
            return
        try:
            report = run.future.result()
        except Exception as exc:
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            return
        run.result = report.as_dict()
        run.status = (
            "succeeded"
            if report.workflow.get("state") == "succeeded"
            else "failed"
        )

    def _view_locked(self) -> dict[str, object]:
        kind_counts: dict[str, int] = {}
        for agent in self._agents:
            kind = str(agent.get("kind", "unknown"))
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        return {
            "runtime": "grpc_agents",
            "agents": list(self._agents),
            "topology": {
                "central_schedulers": 1,
                "orin_agents": kind_counts.get("robot", 0),
                "edge_agents": kind_counts.get("edge", 0),
                "cloud_agents": kind_counts.get("cloud", 0),
                "total_agents": len(self._agents),
            },
            "run_count": len(self._runs),
        }


real_runtime_service = RealRuntimeService()
