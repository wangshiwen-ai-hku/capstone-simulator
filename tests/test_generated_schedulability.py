from __future__ import annotations

import pytest

from backend.app.scene_generator import build_deterministic_scene
from backend.app.schedulability import (
    SceneSchedulabilityError,
    audit_scene_schedulability,
    ensure_generated_scene_schedulable,
)
from backend.app.schemas import (
    BenchmarkScene,
    Difficulty,
    GenerateSceneRequest,
    PlacementConstraintsSpec,
    SimulateRequest,
    TaskCategory,
)
from backend.app.simulation import run_simulation


@pytest.mark.parametrize(
    ("hardware", "expected_tops"),
    [
        ("orin_nano", 67.0),
        ("orin_nx", 157.0),
        ("orin_agx", 275.0),
    ],
)
def test_jetson_capacity_uses_absolute_sparse_int8_tops(
    hardware: str,
    expected_tops: float,
) -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(robot_hardware=hardware)
    )
    robot = next(node for node in scene.nodes if node.kind == "robot")
    assert robot.gpu_capacity == expected_tops


def test_task_accelerator_demand_is_fixed_across_seed_difficulty_and_board() -> None:
    scenes = [
        build_deterministic_scene(
            GenerateSceneRequest(
                robot_count=1,
                edge_count=1,
                task_categories=[TaskCategory.environment_understanding],
                difficulty=difficulty,
                seed=seed,
                robot_hardware=hardware,
            )
        )
        for difficulty, seed, hardware in [
            (Difficulty.easy, 1, "orin_nano"),
            (Difficulty.medium, 7, "orin_nx"),
            (Difficulty.stress, 99, "orin_agx"),
        ]
    ]
    assert {
        task.gpu_demand
        for scene in scenes
        for task in scene.tasks
    } == {48.0}


@pytest.mark.parametrize("hardware", ["orin_nano", "orin_nx", "orin_agx"])
@pytest.mark.parametrize("difficulty", list(Difficulty))
@pytest.mark.parametrize("seed", [1, 7, 31])
def test_generated_scenes_have_a_candidate_for_every_task(
    hardware: str,
    difficulty: Difficulty,
    seed: int,
) -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=2,
            edge_count=1,
            difficulty=difficulty,
            seed=seed,
            robot_hardware=hardware,
        )
    )
    reports = audit_scene_schedulability(scene)
    assert reports
    assert all(item.feasible for item in reports)


def test_preflight_rejects_child_when_parent_artifact_cannot_return() -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            task_categories=[
                TaskCategory.object_detection,
                TaskCategory.local_control,
            ],
            robot_hardware="orin_nx",
            seed=17,
        )
    )
    parent, child = scene.tasks
    edge_node_id = next(
        node.id for node in scene.nodes if node.kind == "edge"
    )
    parent.placement_constraints = PlacementConstraintsSpec(
        pinned_node_id=edge_node_id,
        allowed_node_kinds=["edge"],
        preferred_node_kinds=["edge"],
        required_capabilities=["cuda"],
        allow_source_node=False,
    )
    child.dependencies = [parent.id]
    forward_link_ids = {
        link.id
        for link in scene.links or []
        if link.source_node_id == "robot_1"
        and link.target_node_id == edge_node_id
    }
    scene.links = [
        link for link in scene.links or [] if link.id in forward_link_ids
    ]
    scene.link_snapshots = [
        snapshot
        for snapshot in scene.link_snapshots or []
        if snapshot.link_id in forward_link_ids
    ]
    for resource in scene.initial_resources:
        resource.cpu_util = 0
        resource.gpu_util = 0
        resource.memory_util = 0

    reports = {
        report.task_id: report
        for report in audit_scene_schedulability(scene)
    }

    assert reports[parent.id].feasible_node_ids == (edge_node_id,)
    assert not reports[child.id].feasible
    assert any(
        "dependency_artifact_unreachable" in reason
        and "no_online_link_path" in reason
        for reason in reports[child.id].rejection_reasons
    )
    resources_before = [
        resource.model_dump() for resource in scene.initial_resources
    ]
    with pytest.raises(
        SceneSchedulabilityError,
        match="dependency_artifact_unreachable",
    ):
        ensure_generated_scene_schedulable(scene)
    assert [
        resource.model_dump() for resource in scene.initial_resources
    ] == resources_before


def test_absolute_custom_cpu_demand_cannot_fit_on_smaller_node() -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=1,
            edge_count=0,
            task_categories=[TaskCategory.local_control],
            seed=19,
        )
    )
    payload = scene.model_dump(mode="json")
    payload["nodes"][0]["cpu_capacity"] = 1.0
    payload["tasks"][0]["task_type"] = "custom_four_core_controller"
    payload["tasks"][0]["compute_demand"] = 4.0
    payload["tasks"][0]["gpu_demand"] = 0.0
    custom_scene = BenchmarkScene.model_validate(payload)

    report = audit_scene_schedulability(custom_scene)[0]

    assert not report.feasible
    assert "robot_1:cpu_capacity_insufficient" in report.rejection_reasons


def test_known_workload_declared_cpu_demand_is_authoritative() -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=1,
            edge_count=0,
            task_categories=[TaskCategory.local_control],
            seed=23,
        )
    )
    scene.tasks[0].compute_demand = 20.0
    for resource in scene.initial_resources:
        resource.cpu_util = 0
        resource.gpu_util = 0
        resource.memory_util = 0

    report = audit_scene_schedulability(scene)[0]

    assert not report.feasible
    assert "robot_1:cpu_capacity_insufficient" in report.rejection_reasons


def test_reported_agent_workflow_completes_under_every_stable_policy() -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            scenario_type="custom",
            robot_count=2,
            edge_count=1,
            task_categories=[
                TaskCategory.environment_understanding,
                TaskCategory.localization,
                TaskCategory.map_fusion,
                TaskCategory.obstacle_avoidance,
                TaskCategory.local_planning,
                TaskCategory.local_control,
            ],
            difficulty=Difficulty.medium,
            seed=7,
            robot_hardware="orin_nx",
        )
    )
    for algorithm in [
        "dag_deadline",
        "rule_based",
        "local_first",
        "edge_first",
        "greedy_cost",
    ]:
        result = run_simulation(
            SimulateRequest(
                scene=scene,
                algorithm=algorithm,
                seed=7,
                network_jitter=0,
                resource_noise=0,
            )
        )
        assert result.metrics.success_rate == 1.0, algorithm
        assert result.metrics.skipped_task_count == 0, algorithm
