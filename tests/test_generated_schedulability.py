from __future__ import annotations

import pytest

from backend.app.scene_generator import build_deterministic_scene
from backend.app.schedulability import audit_scene_schedulability
from backend.app.schemas import (
    Difficulty,
    GenerateSceneRequest,
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
