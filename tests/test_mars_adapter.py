from __future__ import annotations

import unittest

from backend.app.mars_adapter import (
    SceneValidationError,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
    validate_scene,
)
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import GenerateSceneRequest


class MarsAdapterValidationTests(unittest.TestCase):
    def setUp(self):
        self.scene = build_deterministic_scene(
            GenerateSceneRequest(robot_count=2, edge_count=1, use_llm=False)
        )

    def test_accepts_scene_with_consistent_node_references(self):
        index = validate_scene(self.scene)
        self.assertEqual(len(index.topological_order), len(self.scene.tasks))

    def test_rejects_missing_resource_snapshot(self):
        self.scene.initial_resources = self.scene.initial_resources[:-1]
        with self.assertRaisesRegex(SceneValidationError, "missing resource snapshots"):
            validate_scene(self.scene)

    def test_rejects_unknown_task_source(self):
        self.scene.tasks[0].source_robot_id = "missing_robot"
        with self.assertRaisesRegex(SceneValidationError, "unknown source robot"):
            validate_scene(self.scene)

    def test_rejects_non_robot_task_source(self):
        edge_id = next(node.id for node in self.scene.nodes if node.kind == "edge")
        self.scene.tasks[0].source_robot_id = edge_id
        with self.assertRaisesRegex(SceneValidationError, "must be a robot node"):
            validate_scene(self.scene)

    def test_maps_local_fallback_into_mars_task_spec(self):
        workload = next(task for task in self.scene.tasks if task.task_class.value == "edge_heavy")
        workload.allow_local_fallback = False
        workflow = build_workflow(self.scene)
        task = next(item for item in workflow.tasks if item.task_id == workload.id)
        self.assertFalse(task.spec.allow_local_fallback)

    def test_keeps_static_node_spec_separate_from_dynamic_snapshot(self):
        specs = {spec.node_id: spec for spec in build_node_specs(self.scene)}
        snapshots = {
            snapshot.node_id: snapshot for snapshot in build_node_snapshots(self.scene)
        }
        robot_id = next(node.id for node in self.scene.nodes if node.kind == "robot")
        self.assertEqual(specs[robot_id].architecture, "jetson-orin")
        self.assertEqual(specs[robot_id].memory_gb, 16)
        self.assertEqual(
            snapshots[robot_id].network_latency_ms,
            next(item.network_latency_ms for item in self.scene.initial_resources if item.node_id == robot_id),
        )

    def test_generated_localization_output_fans_out_to_multiple_consumers(self):
        workflow = build_workflow(self.scene)
        localization_tasks = {
            task.task_id
            for task in workflow.tasks
            if task.spec.task_type == "localization"
        }
        fanout: dict[tuple[str, str], set[str]] = {}
        for edge in workflow.data_edges:
            if edge.producer_task in localization_tasks:
                fanout.setdefault(
                    (edge.producer_task, edge.producer_port), set()
                ).add(edge.consumer_task)
        self.assertTrue(any(len(consumers) >= 2 for consumers in fanout.values()))


if __name__ == "__main__":
    unittest.main()
