from __future__ import annotations

import unittest

from backend.app.mars_adapter import (
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import GenerateSceneRequest
from mars.coordinator import CentralCoordinator
from mars.domain import (
    DataEdge,
    DataPort,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    TaskClass,
    TaskInstance,
    TaskSpec,
    WorkflowSpec,
)
from mars.runtime import InProcessRuntime


class CoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = build_deterministic_scene(
            GenerateSceneRequest(robot_count=2, edge_count=1, use_llm=False, seed=19)
        )
        self.workflow = build_workflow(self.scene)
        self.node_specs = build_node_specs(self.scene)
        self.snapshots = build_node_snapshots(self.scene)
        self.runtime = self._new_runtime()

    def _new_runtime(self) -> InProcessRuntime:
        return InProcessRuntime(
            self.node_specs,
            self.snapshots,
            max_concurrency={spec.node_id: 2 for spec in self.node_specs},
        )

    def test_three_agents_register_retry_and_release_resources(self):
        failed_task_id = next(
            task.task_id
            for task in self.workflow.tasks
            if task.spec.task_type == "local_llm_7b"
        )
        report = CentralCoordinator(self.runtime).run(
            self.workflow,
            seed=19,
            max_attempts=2,
            fail_first_task_ids=(failed_task_id,),
        )
        payload = report.as_dict()

        self.assertEqual(len(payload["agents"]), 3)
        self.assertEqual(
            [item["kind"] for item in payload["agents"]].count("robot"), 2
        )
        self.assertEqual(
            [item["kind"] for item in payload["agents"]].count("edge"), 1
        )
        self.assertTrue(all(item["registered"] for item in payload["agents"]))
        self.assertTrue(all(item["active_reservations"] == 0 for item in payload["agents"]))
        self.assertEqual(payload["workflow"]["state"], "succeeded")
        self.assertEqual(payload["metrics"]["retry_count"], 1)
        self.assertEqual(payload["metrics"]["retry_success_count"], 1)

        retried = next(
            item for item in payload["task_results"] if item["task_id"] == failed_task_id
        )
        self.assertEqual(retried["attempt_count"], 2)
        self.assertEqual(
            [attempt["state"] for attempt in retried["attempts"]],
            ["failed", "succeeded"],
        )
        self.assertNotEqual(
            retried["attempts"][0]["target_node_id"],
            retried["attempts"][1]["target_node_id"],
        )
        event_types = {event["event_type"] for event in payload["events"]}
        self.assertIn("agent_registered", event_types)
        self.assertIn("retry_scheduled", event_types)
        self.assertIn("artifact_published", event_types)

    def test_localization_fanout_reuses_the_same_artifact_reference(self):
        CentralCoordinator(self.runtime).run(self.workflow, seed=5)
        localization = next(
            task for task in self.workflow.tasks if task.spec.task_type == "localization"
        )
        consumers = {
            edge.consumer_task
            for edge in self.workflow.data_edges
            if edge.producer_task == localization.task_id
        }
        self.assertGreaterEqual(len(consumers), 2)
        invocations = {
            invocation.task_id: invocation
            for invocation in self.runtime.executions
        }
        refs = []
        for consumer in consumers:
            refs.append(
                next(
                    artifact
                    for artifact in invocations[consumer].input_artifacts
                    if artifact.producer_task_id == localization.task_id
                )
            )
        self.assertTrue(all(item == refs[0] for item in refs[1:]))

    def test_retry_uses_an_alternate_agent_across_scheduler_policies(self):
        failed_task_id = next(
            task.task_id
            for task in self.workflow.tasks
            if task.spec.task_type == "local_llm_7b"
        )
        for algorithm in (
            "dag_deadline",
            "rule_based",
            "local_first",
            "edge_first",
            "greedy_cost",
        ):
            with self.subTest(algorithm=algorithm):
                report = CentralCoordinator(self._new_runtime()).run(
                    self.workflow,
                    algorithm=algorithm,
                    fail_first_task_ids=(failed_task_id,),
                )
                retried = next(
                    item
                    for item in report.task_results
                    if item["task_id"] == failed_task_id
                )
                self.assertEqual(retried["state"], "succeeded")
                self.assertNotEqual(
                    retried["attempts"][0]["target_node_id"],
                    retried["attempts"][1]["target_node_id"],
                )

    def test_stateful_non_idempotent_task_is_not_retried(self):
        failed_task_id = next(
            task.task_id
            for task in self.workflow.tasks
            if task.spec.task_type == "localization"
        )

        report = CentralCoordinator(self._new_runtime()).run(
            self.workflow,
            max_attempts=3,
            fail_first_task_ids=(failed_task_id,),
        )
        failed = next(
            item
            for item in report.task_results
            if item["task_id"] == failed_task_id
        )

        self.assertEqual(failed["attempt_count"], 1)
        self.assertNotEqual(failed["state"], "succeeded")
        events = [
            event
            for event in report.events
            if event.task_id == failed_task_id
        ]
        self.assertIn(
            "retry_suppressed",
            {event.event_type for event in events},
        )
        self.assertNotIn(
            "retry_scheduled",
            {event.event_type for event in events},
        )

    def test_only_the_selected_output_port_contributes_transfer_cost(self):
        runtime = _minimal_runtime()
        producer = TaskInstance(
            "producer",
            "multi-output",
            "producer",
            "robot_1",
            TaskSpec(
                "custom_producer",
                TaskClass.LOCAL_SAFETY,
                output_size_mb=10.0,
                output_ports=(
                    DataPort("selected", "selected_type"),
                    DataPort("unused", "unused_type"),
                ),
            ),
        )
        consumer = TaskInstance(
            "consumer",
            "multi-output",
            "consumer",
            "robot_1",
            TaskSpec(
                "custom_edge_consumer",
                TaskClass.EDGE_HEAVY,
                compute_demand=2.0,
                output_size_mb=0.1,
                allow_local_fallback=False,
                input_ports=(DataPort("input", "selected_type"),),
                output_ports=(DataPort("result", "result_type"),),
            ),
            deadline_time_ms=10_000,
        )
        workflow = WorkflowSpec(
            "multi-output",
            (producer, consumer),
            data_edges=(
                DataEdge(
                    "producer",
                    "selected",
                    "consumer",
                    "input",
                    "selected_type",
                ),
            ),
        )
        report = CentralCoordinator(runtime).run(workflow)
        consumer_result = next(
            item for item in report.task_results if item["task_id"] == "consumer"
        )
        self.assertEqual(consumer_result["target_node_id"], "edge_pc")
        self.assertAlmostEqual(report.metrics["transferred_mb"], 5.0, places=5)

    def test_coordinator_uses_only_the_aggregate_runtime_port(self):
        runtime = RecordingRuntime(
            self.node_specs,
            self.snapshots,
            max_concurrency={spec.node_id: 2 for spec in self.node_specs},
        )
        report = CentralCoordinator(runtime).run(self.workflow)

        self.assertEqual(report.workflow["state"], "succeeded")
        self.assertIn("start", runtime.calls)
        self.assertIn("inventory", runtime.calls)
        self.assertIn("dispatch", runtime.calls)
        self.assertIn("receive_completion", runtime.calls)
        self.assertEqual(runtime.calls[-1], "describe")


def _minimal_runtime() -> InProcessRuntime:
    specs = [
        NodeSpec("robot_1", NodeKind.ROBOT, 1, 1, 16, 100, 2),
        NodeSpec("robot_2", NodeKind.ROBOT, 1, 1, 16, 100, 2),
        NodeSpec("edge_pc", NodeKind.EDGE, 8, 4, 64, 1000, 5, safety_capable=False),
    ]
    return InProcessRuntime(
        specs,
        [NodeSnapshot(spec.node_id) for spec in specs],
        max_concurrency={spec.node_id: 2 for spec in specs},
    )


class RecordingRuntime(InProcessRuntime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []

    async def start(self, now_ms):
        self.calls.append("start")
        return await super().start(now_ms)

    async def inventory(self, now_ms):
        self.calls.append("inventory")
        return await super().inventory(now_ms)

    async def dispatch(self, command):
        self.calls.append("dispatch")
        return await super().dispatch(command)

    async def receive_completion(self, dispatch_id):
        self.calls.append("receive_completion")
        return await super().receive_completion(dispatch_id)

    async def cancel(self, attempt_id, reason, now_ms):
        self.calls.append("cancel")
        return await super().cancel(attempt_id, reason, now_ms)

    async def describe(self, makespan_ms):
        self.calls.append("describe")
        return await super().describe(makespan_ms)


if __name__ == "__main__":
    unittest.main()
