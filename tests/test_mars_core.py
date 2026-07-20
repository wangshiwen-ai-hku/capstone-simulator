from __future__ import annotations

import unittest

from mars.dag import DagValidationError, TaskManager, validate_workflow
from mars.engine import run_workflow_simulation
from mars.models import (
    ArtifactRef,
    FailurePolicy,
    NodeKind,
    NodeSnapshot,
    TaskClass,
    TaskInstance,
    TaskSpec,
    TaskState,
    WorkflowSpec,
)
from mars.scheduler import allowed_nodes, estimate_candidate
from mars.profiling import ExecutionProfile, ProfileCatalog, load_default_catalog


def task(
    task_id: str,
    task_class: TaskClass = TaskClass.REALTIME_OFFLOADABLE,
    dependencies: tuple[str, ...] = (),
    *,
    compute: float = 1.0,
    accuracy: float = 1.0,
    allow_local_fallback: bool = True,
    source_node_id: str = "robot_1",
    arrival: float = 0.0,
    deadline: float = 5000.0,
) -> TaskInstance:
    return TaskInstance(
        task_id=task_id,
        workflow_id="wf",
        name=task_id,
        source_node_id=source_node_id,
        spec=TaskSpec(
            task_type=task_id,
            task_class=task_class,
            compute_demand=compute,
            gpu_demand=0.5,
            latency_budget_ms=1000,
            input_size_mb=1.0,
            output_size_mb=0.2,
            allow_local_fallback=allow_local_fallback,
        ),
        dependency_task_ids=dependencies,
        arrival_time_ms=arrival,
        deadline_time_ms=deadline,
        expected_accuracy=accuracy,
    )


def nodes() -> list[NodeSnapshot]:
    return [
        NodeSnapshot("robot_1", NodeKind.ROBOT, 1, 1, 16, 100, 2, power_w=20),
        NodeSnapshot("edge_1", NodeKind.EDGE, 5, 4, 64, 1000, 5, power_w=120, safety_capable=False),
    ]


class DagValidationTests(unittest.TestCase):
    def test_validates_and_orders_multi_parent_dag(self):
        workflow = WorkflowSpec("wf", (task("a"), task("b"), task("c", dependencies=("a", "b"))))
        index = validate_workflow(workflow)
        self.assertEqual(index.parents["c"], ("a", "b"))
        self.assertEqual(index.levels["c"], 1)
        self.assertGreater(index.topological_order.index("c"), index.topological_order.index("a"))

    def test_rejects_cycle_atomically(self):
        workflow = WorkflowSpec("wf", (task("a", dependencies=("b",)), task("b", dependencies=("a",))))
        manager = TaskManager()
        with self.assertRaisesRegex(DagValidationError, "cycle"):
            manager.submit(workflow)
        with self.assertRaisesRegex(RuntimeError, "no workflow"):
            _ = manager.workflow

    def test_rejects_missing_parent(self):
        with self.assertRaisesRegex(DagValidationError, "missing dependencies"):
            validate_workflow(WorkflowSpec("wf", (task("a", dependencies=("missing",)),)))


class TaskManagerTests(unittest.TestCase):
    def test_children_are_blocked_until_all_parents_succeed(self):
        manager = TaskManager()
        manager.submit(WorkflowSpec("wf", (task("a"), task("b"), task("c", dependencies=("a", "b")))))
        self.assertEqual({item.task_id for item in manager.ready()}, {"a", "b"})
        manager.mark_running("a")
        released, _ = manager.complete("a", ok=True, finished_time_ms=10)
        self.assertEqual(released, [])
        self.assertEqual(manager.state_of("c"), TaskState.BLOCKED)
        manager.mark_running("b")
        released, _ = manager.complete("b", ok=True, finished_time_ms=12)
        self.assertEqual(released, ["c"])

    def test_failure_skips_only_descendants_by_default(self):
        manager = TaskManager()
        manager.submit(WorkflowSpec("wf", (task("a"), task("independent"), task("c", dependencies=("a",)))))
        manager.mark_running("a")
        _, skipped = manager.complete("a", ok=False, finished_time_ms=10)
        self.assertEqual(skipped, ["c"])
        self.assertEqual(manager.state_of("independent"), TaskState.READY)

    def test_fail_fast_cancels_all_unresolved_tasks(self):
        manager = TaskManager()
        manager.submit(
            WorkflowSpec(
                "wf",
                (task("a"), task("independent"), task("c", dependencies=("a",))),
                failure_policy=FailurePolicy.FAIL_FAST,
            )
        )
        manager.mark_running("a")
        _, skipped = manager.complete("a", ok=False, finished_time_ms=10)
        self.assertEqual(set(skipped), {"independent", "c"})


class PlacementTests(unittest.TestCase):
    def test_local_safety_can_only_run_on_source_robot(self):
        candidates = allowed_nodes(task("avoid", TaskClass.LOCAL_SAFETY), nodes())
        self.assertEqual([candidate.node_id for candidate in candidates], ["robot_1"])

    def test_local_safety_rejects_edge_even_if_marked_safety_capable(self):
        edge = NodeSnapshot(
            "edge_1",
            NodeKind.EDGE,
            5,
            4,
            64,
            1000,
            5,
            power_w=120,
            safety_capable=True,
        )
        item = task(
            "avoid",
            TaskClass.LOCAL_SAFETY,
            source_node_id="edge_1",
        )
        self.assertEqual(allowed_nodes(item, [edge]), [])
        estimate = estimate_candidate(
            item,
            edge,
            ready_time_ms=0,
            node_available_ms=0,
            nodes={edge.node_id: edge},
            parent_artifacts=(),
        )
        self.assertFalse(estimate.feasible)
        self.assertEqual(estimate.reason, "local_safety_requires_safety_capable_source_robot")

    def test_yolo_class_can_use_robot_or_edge(self):
        candidates = allowed_nodes(task("yolo", TaskClass.REALTIME_OFFLOADABLE), nodes())
        self.assertEqual({candidate.node_id for candidate in candidates}, {"robot_1", "edge_1"})

    def test_edge_heavy_can_disable_local_fallback(self):
        candidates = allowed_nodes(
            task("vla", TaskClass.EDGE_HEAVY, allow_local_fallback=False),
            nodes(),
        )
        self.assertEqual([candidate.node_id for candidate in candidates], ["edge_1"])

    def test_edge_heavy_can_explicitly_allow_local_fallback(self):
        candidates = allowed_nodes(
            task("vla", TaskClass.EDGE_HEAVY, allow_local_fallback=True),
            nodes(),
        )
        self.assertEqual([candidate.node_id for candidate in candidates], ["edge_1", "robot_1"])

    def test_parent_artifact_location_replaces_source_upload_assumption(self):
        task_b = task("b", dependencies=("a",))
        node_by_id = {node.node_id: node for node in nodes()}
        estimate = estimate_candidate(
            task_b,
            node_by_id["edge_1"],
            ready_time_ms=10,
            node_available_ms=0,
            nodes=node_by_id,
            parent_artifacts=(ArtifactRef("x", "a", "edge_1", 20.0),),
        )
        self.assertEqual(estimate.communication_ms, 0.0)
        self.assertEqual(estimate.input_locations, ("edge_1",))

    def test_synthetic_profile_catalog_is_replaceable_and_loaded(self):
        catalog = load_default_catalog()
        self.assertIsNotNone(catalog)
        profile = catalog.lookup("object_detection", NodeKind.EDGE)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.provenance, "synthetic_placeholder")


class EngineTests(unittest.TestCase):
    def test_engine_runs_dag_and_reports_three_classes(self):
        workflow = WorkflowSpec(
            "wf",
            (
                task("avoid", TaskClass.LOCAL_SAFETY),
                task("yolo", TaskClass.REALTIME_OFFLOADABLE),
                task("vla", TaskClass.EDGE_HEAVY, dependencies=("yolo",), compute=4.0),
            ),
            deadline_time_ms=5000,
        )
        report = run_workflow_simulation(workflow, nodes(), algorithm="dag_deadline", seed=2)
        self.assertTrue(report.dag["valid"])
        self.assertEqual(report.metrics["safety_violation_count"], 0)
        self.assertEqual(set(report.task_class_summary), {item.value for item in TaskClass})
        avoid = next(result for result in report.task_results if result.task_id == "avoid")
        self.assertEqual(avoid.target_node_id, "robot_1")

    def test_future_arrival_does_not_reserve_node_ahead_of_released_child(self):
        workflow = WorkflowSpec(
            "wf",
            (
                task("future", arrival=1000),
                task("root"),
                task("child", dependencies=("root",)),
            ),
        )
        report = run_workflow_simulation(workflow, [nodes()[0]], seed=2)
        result = {item.task_id: item for item in report.task_results}
        self.assertGreaterEqual(result["future"].start_time_ms, 1000)
        self.assertLess(result["child"].finish_time_ms, result["future"].start_time_ms)

    def test_early_execution_failure_is_not_a_deadline_miss(self):
        workflow = WorkflowSpec("wf", (task("failure", accuracy=0.0, deadline=5000),))
        report = run_workflow_simulation(
            workflow,
            [nodes()[0]],
            seed=2,
            network_jitter=0,
            resource_noise=0,
        )
        result = report.task_results[0]
        self.assertFalse(result.success)
        self.assertFalse(result.deadline_missed)
        self.assertEqual(report.metrics["deadline_miss_rate"], 0.0)

    def test_report_preserves_measured_profile_provenance(self):
        catalog = ProfileCatalog(
            [
                ExecutionProfile(
                    task_type="measured",
                    task_class=TaskClass.REALTIME_OFFLOADABLE,
                    node_kind=NodeKind.ROBOT,
                    model_variant="fixture",
                    input_shape="1x3x640x640",
                    precision="fp16",
                    batch_size=1,
                    p50_ms=8,
                    p95_ms=10,
                    p99_ms=12,
                    throughput_per_s=100,
                    peak_memory_mb=512,
                    energy_j=1.2,
                    output_size_mb=0.1,
                    provenance="measured_lab",
                )
            ]
        )
        report = run_workflow_simulation(
            WorkflowSpec("wf", (task("measured"),)),
            [nodes()[0]],
            profiles=catalog,
            seed=2,
        )
        self.assertEqual(catalog.provenance, "measured_lab")
        self.assertEqual(report.transport["profile_source"], "measured_lab")
        self.assertEqual(report.transport["profile_catalog_provenance"], "measured_lab")

    def test_report_identifies_formula_fallback_when_catalog_row_is_missing(self):
        catalog = ProfileCatalog(
            [
                ExecutionProfile(
                    task_type="other",
                    task_class=TaskClass.REALTIME_OFFLOADABLE,
                    node_kind=NodeKind.ROBOT,
                    model_variant="fixture",
                    input_shape="1x3x640x640",
                    precision="fp16",
                    batch_size=1,
                    p50_ms=8,
                    p95_ms=10,
                    p99_ms=12,
                    throughput_per_s=100,
                    peak_memory_mb=512,
                    energy_j=1.2,
                    output_size_mb=0.1,
                    provenance="measured_lab",
                )
            ]
        )
        report = run_workflow_simulation(
            WorkflowSpec("wf", (task("unprofiled"),)),
            [nodes()[0]],
            profiles=catalog,
            seed=2,
        )
        self.assertEqual(report.transport["profile_source"], "demand_formula_fallback")
        self.assertEqual(report.transport["profile_sources"], ["demand_formula_fallback"])
        self.assertEqual(report.transport["profile_catalog_provenance"], "measured_lab")


if __name__ == "__main__":
    unittest.main()
