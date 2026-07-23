"""Translate web-facing scene models into MARS domain objects."""

from __future__ import annotations

from mars.dag import DagIndex, DagValidationError, validate_workflow
from mars.models import (
    DataEdge,
    DataPort,
    FailurePolicy,
    LinkSnapshot as MarsLinkSnapshot,
    LinkSpec as MarsLinkSpec,
    NodeKind,
    NodeSnapshot,
    NodeSpec as MarsNodeSpec,
    PlacementConstraints,
    ResourceClass,
    TaskClass,
    TaskInstance,
    TaskSpec,
    WorkflowSpec,
)
from mars.network import NetworkTopology, synthesize_legacy_full_mesh

from .schemas import BenchmarkScene


class SceneValidationError(ValueError):
    """The web scene cannot be converted into the MARS domain."""


def validate_scene(scene: BenchmarkScene) -> DagIndex:
    """Validate scene references and return the MARS DAG index."""
    _validated_resource_map(scene)
    node_by_id = {node.id: node for node in scene.nodes}
    for task in scene.tasks:
        source = node_by_id.get(task.source_robot_id)
        if source is None:
            raise SceneValidationError(
                f"task {task.id} references unknown source robot: {task.source_robot_id}"
            )
        if source.kind != "robot":
            raise SceneValidationError(
                f"task {task.id} source {task.source_robot_id} must be a robot node"
            )
    try:
        workflow = build_workflow(scene)
        node_specs = build_node_specs(scene)
        node_snapshots = build_node_snapshots(scene)
        link_specs, link_snapshots = _network_inventory(scene)
        NetworkTopology(
            (node.node_id for node in node_specs),
            link_specs,
            link_snapshots,
            node_online={
                snapshot.node_id: snapshot.online
                for snapshot in node_snapshots
            },
        )
        return validate_workflow(workflow)
    except (DagValidationError, ValueError) as exc:
        raise SceneValidationError(str(exc)) from exc


def build_workflow(scene: BenchmarkScene) -> WorkflowSpec:
    """Convert a web scene into the MARS workflow model."""
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
                    allow_local_fallback=task.allow_local_fallback,
                    input_ports=tuple(
                        DataPort(port.name, port.message_type)
                        for port in task.input_ports
                    ),
                    output_ports=tuple(
                        DataPort(port.name, port.message_type)
                        for port in task.output_ports
                    ),
                    placement_constraints=_placement_constraints(task),
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
        data_edges=tuple(
            DataEdge(
                edge.producer_task,
                edge.producer_port,
                edge.consumer_task,
                edge.consumer_port,
                edge.message_type,
            )
            for edge in scene.data_edges
        ),
    )


def build_node_specs(scene: BenchmarkScene) -> list[MarsNodeSpec]:
    """Convert web node declarations into static MARS node specifications."""
    return [
        MarsNodeSpec(
            node_id=node.id,
            kind=NodeKind(node.kind),
            architecture=node.architecture,
            cpu_capacity=node.cpu_capacity,
            gpu_capacity=node.gpu_capacity,
            memory_gb=node.memory_gb,
            bandwidth_mbps=node.bandwidth_mbps,
            base_latency_ms=node.base_latency_ms,
            battery_capacity_wh=node.battery_wh,
            safety_capable=node.safety_capable,
            capabilities=tuple(node.capabilities),
            supported_models=tuple(node.supported_models),
            max_concurrency=node.max_concurrency,
        )
        for node in scene.nodes
    ]


def build_node_snapshots(scene: BenchmarkScene) -> list[NodeSnapshot]:
    """Convert web resource reports into dynamic MARS node snapshots."""
    resources = _validated_resource_map(scene)
    out: list[NodeSnapshot] = []
    for node in scene.nodes:
        resource = resources[node.id]
        out.append(
            NodeSnapshot(
                node_id=node.id,
                cpu_util=resource.cpu_util,
                gpu_util=resource.gpu_util,
                memory_util=resource.memory_util,
                temperature_c=resource.temperature_c,
                power_w=resource.power_w,
                network_latency_ms=resource.network_latency_ms,
                online=resource.online,
            )
        )
    return out


def build_link_specs(scene: BenchmarkScene) -> list[MarsLinkSpec]:
    """Convert explicit links or synthesize the legacy directed full mesh."""

    links, _ = _network_inventory(scene)
    return list(links)


def build_link_snapshots(scene: BenchmarkScene) -> list[MarsLinkSnapshot]:
    """Convert explicit link telemetry or synthesize legacy link state."""

    _, snapshots = _network_inventory(scene)
    return list(snapshots)


def _network_inventory(
    scene: BenchmarkScene,
) -> tuple[tuple[MarsLinkSpec, ...], tuple[MarsLinkSnapshot, ...]]:
    if scene.links is None and scene.link_snapshots is None:
        return synthesize_legacy_full_mesh(
            build_node_specs(scene),
            build_node_snapshots(scene),
        )
    if scene.links is None or scene.link_snapshots is None:
        raise SceneValidationError(
            "links and link_snapshots must both be provided or omitted"
        )
    return (
        tuple(
            MarsLinkSpec(
                link_id=item.id,
                source_node_id=item.source_node_id,
                target_node_id=item.target_node_id,
                bandwidth_mbps=item.bandwidth_mbps,
                base_latency_ms=item.base_latency_ms,
            )
            for item in scene.links
        ),
        tuple(
            MarsLinkSnapshot(
                link_id=item.link_id,
                available_bandwidth_mbps=item.available_bandwidth_mbps,
                latency_ms=item.latency_ms,
                jitter_ms=item.jitter_ms,
                packet_loss_rate=item.packet_loss_rate,
                online=item.online,
            )
            for item in scene.link_snapshots
        ),
    )


def _placement_constraints(task) -> PlacementConstraints:
    explicit = task.placement_constraints
    if explicit is not None:
        pinned_node_id = (
            task.source_robot_id
            if explicit.pin_to_source
            else explicit.pinned_node_id
        )
        return PlacementConstraints(
            pinned_node_id=pinned_node_id,
            allowed_node_kinds=tuple(
                NodeKind(kind) for kind in explicit.allowed_node_kinds
            ),
            preferred_node_kinds=tuple(
                NodeKind(kind) for kind in explicit.preferred_node_kinds
            ),
            required_capabilities=tuple(
                explicit.required_capabilities
            ),
            allow_source_node=explicit.allow_source_node,
            allow_other_robots=explicit.allow_other_robots,
            safety_required=explicit.safety_required,
            allow_fallback=explicit.allow_fallback,
            stateful=explicit.stateful,
            idempotent=explicit.idempotent,
            splittable=explicit.splittable,
            replicable=explicit.replicable,
        )
    task_class = TaskClass(task.task_class.value)
    if task_class is TaskClass.LOCAL_SAFETY:
        return PlacementConstraints(
            pinned_node_id=task.source_robot_id,
            allowed_node_kinds=(NodeKind.ROBOT,),
            preferred_node_kinds=(NodeKind.ROBOT,),
            required_capabilities=("local_safety",),
            allow_source_node=True,
            allow_other_robots=False,
            safety_required=True,
            allow_fallback=False,
            stateful=True,
            idempotent=False,
        )
    if task_class is TaskClass.REALTIME_OFFLOADABLE:
        return PlacementConstraints(
            allowed_node_kinds=(NodeKind.EDGE,),
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=bool(task.allow_local_fallback),
        )
    return PlacementConstraints(
        allowed_node_kinds=(NodeKind.EDGE,),
        preferred_node_kinds=(NodeKind.EDGE,),
        allow_source_node=bool(task.allow_local_fallback),
        allow_other_robots=False,
        allow_fallback=bool(task.allow_local_fallback),
    )


def _validated_resource_map(scene: BenchmarkScene):
    node_ids = [node.id for node in scene.nodes]
    resource_ids = [resource.node_id for resource in scene.initial_resources]
    if any(not node_id.strip() for node_id in node_ids):
        raise SceneValidationError("node ids must be non-empty")
    if len(node_ids) != len(set(node_ids)):
        raise SceneValidationError("node ids must be unique")
    if len(resource_ids) != len(set(resource_ids)):
        raise SceneValidationError("resource snapshot node ids must be unique")
    missing = sorted(set(node_ids) - set(resource_ids))
    unknown = sorted(set(resource_ids) - set(node_ids))
    if missing:
        raise SceneValidationError(f"missing resource snapshots for nodes: {', '.join(missing)}")
    if unknown:
        raise SceneValidationError(f"resource snapshots reference unknown nodes: {', '.join(unknown)}")
    return {resource.node_id: resource for resource in scene.initial_resources}
