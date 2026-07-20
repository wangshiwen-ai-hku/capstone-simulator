"""FastAPI compatibility facade for the transport-neutral DAG engine."""

from __future__ import annotations

import logging

from edgesched.engine import run_workflow_simulation
from edgesched.models import (
    FailurePolicy,
    NodeKind,
    NodeSnapshot,
    ResourceClass,
    TaskClass,
    TaskInstance,
    TaskSpec,
    WorkflowSpec,
)

from .schemas import SimulateRequest, SimulationResponse

logger = logging.getLogger(__name__)


async def run_simulation(req: SimulateRequest) -> SimulationResponse:
    workflow = _to_workflow(req)
    nodes = _to_nodes(req)
    algorithm = req.algorithm
    if algorithm == "external":
        # The v1 external contract lacks topology and artifact-location fields.
        # DAG-aware placement is used until a v2 external contract is available.
        logger.warning(
            "external v1 task callback is deprecated; using dag_deadline because the endpoint "
            "does not receive workflow topology or artifact locations"
        )
    report = run_workflow_simulation(
        workflow,
        nodes,
        algorithm="dag_deadline" if algorithm == "external" else algorithm,
        seed=req.seed,
        network_jitter=req.network_jitter,
        resource_noise=req.resource_noise,
    )
    if algorithm == "external":
        report.algorithm = "external→dag_deadline"
        report.logs.insert(
            0,
            "DEPRECATED: v1 external scheduler callback replaced by DAG-safe dag_deadline policy; "
            "use the edgesched.v2 workflow adapter contract for external policies.",
        )
        report.transport["external_scheduler_url_ignored"] = req.external_scheduler_url or "missing"
    return SimulationResponse.model_validate(report.as_dict())


def _to_workflow(req: SimulateRequest) -> WorkflowSpec:
    scene = req.scene
    tasks: list[TaskInstance] = []
    for task in scene.tasks:
        task_class = TaskClass(task.task_class.value)
        dominant = (
            ResourceClass.GPU
            if task.gpu_demand > 0.25
            else ResourceClass.IO
            if task.task_type in {"data_compression", "map_fusion"}
            else ResourceClass.CPU
        )
        tasks.append(
            TaskInstance(
                task_id=task.id,
                workflow_id=scene.workflow_id,
                name=task.name,
                source_node_id=task.source_robot_id,
                spec=TaskSpec(
                    task_type=task.task_type,
                    task_class=task_class,
                    compute_demand=task.compute_demand,
                    gpu_demand=task.gpu_demand,
                    latency_budget_ms=task.latency_budget_ms,
                    model_requirement=task.model_requirement,
                    input_size_mb=task.data_size_mb,
                    output_size_mb=task.output_size_mb,
                    bandwidth_requirement_mbps=task.bandwidth_requirement_mbps,
                    energy_budget_j=task.energy_budget_j,
                    dominant_resource=dominant,
                    allow_local_fallback=True,
                ),
                dependency_task_ids=tuple(task.dependencies),
                priority=task.priority,
                stage_index=task.stage_index,
                arrival_time_ms=task.arrival_time_ms,
                deadline_time_ms=task.deadline_ms,
                expected_accuracy=task.expected_accuracy,
                input_ref=f"scene://{scene.id}/{task.id}/input",
            )
        )
    return WorkflowSpec(
        workflow_id=scene.workflow_id,
        tasks=tuple(tasks),
        deadline_time_ms=scene.workflow_deadline_ms,
        failure_policy=FailurePolicy(scene.failure_policy.value),
        metadata={"scene_id": scene.id, "scenario_type": scene.scenario_type},
    )


def _to_nodes(req: SimulateRequest) -> list[NodeSnapshot]:
    resources = {resource.node_id: resource for resource in req.scene.initial_resources}
    out: list[NodeSnapshot] = []
    for node in req.scene.nodes:
        resource = resources[node.id]
        out.append(
            NodeSnapshot(
                node_id=node.id,
                kind=NodeKind(node.kind),
                cpu_capacity=node.cpu_capacity,
                gpu_capacity=node.gpu_capacity,
                memory_gb=node.memory_gb,
                bandwidth_mbps=node.bandwidth_mbps,
                base_latency_ms=node.base_latency_ms + resource.network_latency_ms,
                cpu_util=resource.cpu_util,
                gpu_util=resource.gpu_util,
                memory_util=resource.memory_util,
                temperature_c=resource.temperature_c,
                power_w=resource.power_w,
                safety_capable=node.safety_capable,
            )
        )
    return out
