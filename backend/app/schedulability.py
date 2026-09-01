"""Generation-time schedulability checks for synthetic Studio scenes."""

from __future__ import annotations

from dataclasses import dataclass

from mars.dag import validate_workflow
from mars.domain import ArtifactRef, TaskInstance, task_resource_demand
from mars.profiling import profile_catalog_from_workloads
from mars.network import NetworkTopology
from mars.scheduler import allowed_nodes, estimate_candidate
from mars.synthetic_workloads import load_default_synthetic_workloads

from .mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from .schemas import BenchmarkScene


@dataclass(frozen=True)
class TaskSchedulability:
    task_id: str
    feasible_node_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return bool(self.feasible_node_ids)


class SceneSchedulabilityError(ValueError):
    """A generated scene contains a task with no executable candidate."""


def audit_scene_schedulability(
    scene: BenchmarkScene,
) -> tuple[TaskSchedulability, ...]:
    """Check task candidates in DAG order, including artifact reachability.

    A child candidate is feasible only when each parent has at least one
    feasible producer location whose artifact can traverse the declared
    directed topology to that child location. This remains a lightweight
    preflight rather than a complete simultaneous scheduling proof.
    """

    workflow = build_workflow(scene)
    node_specs = build_node_specs(scene)
    node_snapshots = build_node_snapshots(scene)
    specs_by_id = {item.node_id: item for item in node_specs}
    snapshots_by_id = {item.node_id: item for item in node_snapshots}
    profiles = profile_catalog_from_workloads(
        load_default_synthetic_workloads()
    )
    index = validate_workflow(workflow)
    task_by_id = {item.task_id: item for item in workflow.tasks}

    topology = NetworkTopology(
        (item.node_id for item in node_specs),
        build_link_specs(scene),
        build_link_snapshots(scene),
        node_online={
            item.node_id: item.online for item in node_snapshots
        },
    )
    reports: list[TaskSchedulability] = []
    feasible_locations: dict[str, tuple[str, ...]] = {}
    for task_id in index.topological_order:
        task = task_by_id[task_id]
        feasible: list[str] = []
        reasons: list[str] = []
        eligible = allowed_nodes(
            task,
            node_specs,
            snapshots_by_id,
        )
        if not eligible:
            reasons.append("placement_constraints_reject_all_nodes")
        for node in eligible:
            parent_artifacts, dependency_reason = (
                _representative_parent_artifacts(
                    task,
                    target_node_id=node.node_id,
                    parent_task_ids=index.parents[task.task_id],
                    task_by_id=task_by_id,
                    feasible_locations=feasible_locations,
                    specs_by_id=specs_by_id,
                    profiles=profiles,
                    topology=topology,
                )
            )
            if dependency_reason:
                reasons.append(f"{node.node_id}:{dependency_reason}")
                continue
            estimate = estimate_candidate(
                task,
                node,
                ready_time_ms=task.arrival_time_ms,
                node_available_ms=task.arrival_time_ms,
                node_specs=specs_by_id,
                node_snapshots=snapshots_by_id,
                parent_artifacts=parent_artifacts,
                profiles=profiles,
                topology=topology,
            )
            if not estimate.feasible:
                reasons.append(f"{node.node_id}:{estimate.reason}")
                continue
            snapshot = snapshots_by_id[node.node_id]
            demand = estimate.resource_demand
            available = (
                node.cpu_capacity * (1.0 - snapshot.cpu_util),
                node.gpu_capacity * (1.0 - snapshot.gpu_util),
                node.memory_gb * (1.0 - snapshot.memory_util),
            )
            required = (
                demand.cpu_units,
                demand.gpu_units,
                demand.memory_gb,
            )
            labels = ("cpu", "gpu", "memory")
            deficits = [
                f"{label}_background_capacity_insufficient"
                for label, need, free in zip(labels, required, available)
                if need > free + 1e-9
            ]
            if deficits:
                reasons.extend(
                    f"{node.node_id}:{reason}" for reason in deficits
                )
                continue
            feasible.append(node.node_id)
        reports.append(
            TaskSchedulability(
                task_id=task.task_id,
                feasible_node_ids=tuple(feasible),
                rejection_reasons=tuple(dict.fromkeys(reasons)),
            )
        )
        feasible_locations[task.task_id] = tuple(feasible)
    return tuple(reports)


def _representative_parent_artifacts(
    task: TaskInstance,
    *,
    target_node_id: str,
    parent_task_ids: tuple[str, ...],
    task_by_id: dict[str, TaskInstance],
    feasible_locations: dict[str, tuple[str, ...]],
    specs_by_id,
    profiles,
    topology: NetworkTopology,
) -> tuple[tuple[ArtifactRef, ...], str]:
    """Choose one reachable feasible artifact location for every parent."""

    artifacts: list[ArtifactRef] = []
    for parent_task_id in parent_task_ids:
        parent_locations = feasible_locations.get(parent_task_id, ())
        if not parent_locations:
            return (), (
                f"dependency_has_no_feasible_candidate:{parent_task_id}"
            )
        parent = task_by_id[parent_task_id]
        selected: ArtifactRef | None = None
        last_reason = "no_online_link_path"
        for parent_node_id in parent_locations:
            parent_node = specs_by_id[parent_node_id]
            profile = profiles.lookup(
                parent.spec.task_type,
                parent_node.kind,
            )
            output_size_mb = (
                profile.output_size_mb
                if profile is not None
                else parent.spec.output_size_mb
            )
            artifact = ArtifactRef(
                artifact_id=(
                    f"preflight:{parent_task_id}@{parent_node_id}"
                ),
                producer_task_id=parent_task_id,
                node_id=parent_node_id,
                size_mb=output_size_mb,
            )
            transfer = topology.estimate(
                transfer_id=artifact.artifact_id,
                source_node_id=parent_node_id,
                target_node_id=target_node_id,
                size_mb=output_size_mb,
                minimum_bandwidth_mbps=(
                    task.spec.bandwidth_requirement_mbps
                ),
            )
            if transfer.feasible:
                selected = artifact
                break
            last_reason = transfer.reason
        if selected is None:
            return (), (
                f"dependency_artifact_unreachable:{parent_task_id}:"
                f"{last_reason}"
            )
        artifacts.append(selected)
    return tuple(artifacts), ""


def ensure_generated_scene_schedulable(
    scene: BenchmarkScene,
) -> tuple[str, ...]:
    """Repair synthetic background load, then reject hard infeasibility."""

    repairs: list[str] = []
    resources = {item.node_id: item for item in scene.initial_resources}
    workflow = build_workflow(scene)
    task_by_id = {item.task_id: item for item in workflow.tasks}
    node_specs = {item.node_id: item for item in build_node_specs(scene)}
    profiles = profile_catalog_from_workloads(
        load_default_synthetic_workloads()
    )
    remaining: list[TaskSchedulability] = []
    while True:
        reports = audit_scene_schedulability(scene)
        remaining = [report for report in reports if not report.feasible]
        repaired_this_pass = False
        snapshots = {
            item.node_id: item for item in build_node_snapshots(scene)
        }
        for report in remaining:
            repairable_node_ids = _background_deficit_node_ids(report)
            if not repairable_node_ids:
                continue
            task = task_by_id[report.task_id]
            # Only generated utilization is repairable. Task demand and
            # declared hardware capacity remain immutable absolute facts.
            for node in allowed_nodes(
                task,
                node_specs.values(),
                snapshots,
            ):
                if node.node_id not in repairable_node_ids:
                    continue
                profile = profiles.lookup(task.spec.task_type, node.kind)
                if profile is not None and not profile.supported:
                    continue
                cpu, gpu, memory = task_resource_demand(task, node)
                if profile is not None:
                    memory = profile.peak_memory_mb / 1024.0
                if (
                    cpu > node.cpu_capacity + 1e-9
                    or gpu > node.gpu_capacity + 1e-9
                    or memory > node.memory_gb + 1e-9
                ):
                    continue
                state = resources[node.node_id]
                before = (
                    state.cpu_util,
                    state.gpu_util,
                    state.memory_util,
                )
                state.cpu_util = min(
                    state.cpu_util,
                    max(
                        0.0,
                        1.0 - cpu / node.cpu_capacity - 0.05,
                    ),
                )
                state.gpu_util = min(
                    state.gpu_util,
                    max(
                        0.0,
                        1.0
                        - gpu / max(node.gpu_capacity, 1e-9)
                        - 0.05,
                    ),
                )
                state.memory_util = min(
                    state.memory_util,
                    max(
                        0.0,
                        1.0 - memory / node.memory_gb - 0.05,
                    ),
                )
                after = (
                    state.cpu_util,
                    state.gpu_util,
                    state.memory_util,
                )
                if after != before:
                    repair = f"{report.task_id}:{node.node_id}"
                    if repair not in repairs:
                        repairs.append(repair)
                    repaired_this_pass = True
                break
        if not repaired_this_pass:
            break

    if remaining:
        detail = "; ".join(
            f"{item.task_id} ({', '.join(item.rejection_reasons)})"
            for item in remaining
        )
        raise SceneSchedulabilityError(
            "generated scene has no executable candidate for: " + detail
        )
    return tuple(repairs)


def _background_deficit_node_ids(
    report: TaskSchedulability,
) -> frozenset[str]:
    suffixes = tuple(
        f":{label}_background_capacity_insufficient"
        for label in ("cpu", "gpu", "memory")
    )
    return frozenset(
        reason[: -len(suffix)]
        for reason in report.rejection_reasons
        for suffix in suffixes
        if reason.endswith(suffix)
    )


__all__ = [
    "SceneSchedulabilityError",
    "TaskSchedulability",
    "audit_scene_schedulability",
    "ensure_generated_scene_schedulable",
]
