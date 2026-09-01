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

    def test_generated_tasks_declare_authoritative_placement(self):
        self.assertTrue(
            all(
                task.placement_constraints is not None
                for task in self.scene.tasks
            )
        )
        localization = next(
            task
            for task in self.scene.tasks
            if task.task_type == "localization"
        )
        perception = next(
            task
            for task in self.scene.tasks
            if task.task_type == "environment_understanding"
        )
        self.assertEqual(
            localization.task_class,
            perception.task_class,
        )
        self.assertEqual(
            localization.placement_constraints.preferred_node_kinds,
            ["robot"],
        )
        self.assertEqual(
            perception.placement_constraints.preferred_node_kinds,
            ["edge"],
        )

    def test_explicit_placement_is_not_derived_from_reporting_class(self):
        workload = next(
            task
            for task in self.scene.tasks
            if task.task_type == "object_detection"
        )
        payload = workload.model_dump(mode="json")
        payload["task_class"] = "local_safety"
        scene_payload = self.scene.model_dump(mode="json")
        scene_payload["tasks"] = [
            payload if task["id"] == workload.id else task
            for task in scene_payload["tasks"]
        ]
        reconstructed = type(self.scene).model_validate(scene_payload)
        workflow = build_workflow(reconstructed)
        mapped = next(
            task
            for task in workflow.tasks
            if task.task_id == workload.id
        )
        self.assertEqual(
            [kind.value for kind in mapped.spec.placement_constraints.allowed_node_kinds],
            ["robot", "edge"],
        )
        self.assertFalse(
            mapped.spec.placement_constraints.safety_required
        )

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

    def test_explicit_placement_normalizes_legacy_fallback_field(self):
        payload = self.scene.model_dump(mode="json")
        workload = next(
            task
            for task in payload["tasks"]
            if task["task_class"] == "edge_heavy"
        )
        self.assertTrue(
            workload["placement_constraints"]["allow_source_node"]
        )
        self.assertTrue(
            workload["placement_constraints"]["allow_fallback"]
        )
        workload["allow_local_fallback"] = False

        reconstructed = type(self.scene).model_validate(payload)
        normalized = next(
            task
            for task in reconstructed.tasks
            if task.id == workload["id"]
        )
        self.assertTrue(normalized.allow_local_fallback)

    def test_keeps_static_node_spec_separate_from_dynamic_snapshot(self):
        specs = {spec.node_id: spec for spec in build_node_specs(self.scene)}
        snapshots = {
            snapshot.node_id: snapshot for snapshot in build_node_snapshots(self.scene)
        }
        robot_id = next(node.id for node in self.scene.nodes if node.kind == "robot")
        self.assertEqual(specs[robot_id].architecture, "jetson-orin-nx")
        self.assertEqual(specs[robot_id].cpu_capacity, 8)
        self.assertEqual(specs[robot_id].memory_gb, 16)
        self.assertEqual(
            snapshots[robot_id].network_latency_ms,
            next(item.network_latency_ms for item in self.scene.initial_resources if item.node_id == robot_id),
        )
        battery_wh = next(
            node.battery_wh for node in self.scene.nodes if node.id == robot_id
        )
        self.assertEqual(
            snapshots[robot_id].remaining_energy_j,
            battery_wh * 3600.0,
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
