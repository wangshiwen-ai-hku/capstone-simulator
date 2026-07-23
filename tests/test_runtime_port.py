from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace

from mars.coordinator import CentralCoordinator
from mars.models import (
    Assignment,
    ExecutionMode,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    TaskClass,
    TaskInstance,
    TaskSpec,
    WorkflowSpec,
    task_resource_demand,
)
from mars.optimizers import (
    PlannedResourceReservation,
    ResourceDemand,
    SchedulingEpoch,
)
from mars.runtime import DispatchCommand, InProcessRuntime, RuntimePort
from mars.scheduler import build_scheduling_problem, plan_scheduling_epoch

from tests.test_mars_core import task


def _node_spec() -> NodeSpec:
    return NodeSpec(
        "robot_1",
        NodeKind.ROBOT,
        1,
        1,
        16,
        100,
        2,
        architecture="jetson-orin",
        capabilities=("gpu",),
    )


def _runtime(
    *,
    online: bool = True,
    runtime_type: type[InProcessRuntime] = InProcessRuntime,
) -> InProcessRuntime:
    spec = _node_spec()
    return runtime_type(
        (spec,),
        (NodeSnapshot("robot_1", online=online),),
        max_concurrency={"robot_1": 1},
    )


class DispatchRaisesAfterAcceptRuntime(InProcessRuntime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cancelled_attempts: list[str] = []

    async def dispatch(self, command):
        ack = await super().dispatch(command)
        if ack.accepted:
            raise ConnectionError("dispatch acknowledgement lost")
        return ack

    async def cancel(self, attempt_id, reason, now_ms):
        self.cancelled_attempts.append(attempt_id)
        return await super().cancel(attempt_id, reason, now_ms)


class MismatchedCompletionRuntime(InProcessRuntime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cancelled_attempts: list[str] = []

    async def receive_completion(self, dispatch_id):
        completion = await super().receive_completion(dispatch_id)
        return replace(completion, attempt_id="stale-attempt")

    async def cancel(self, attempt_id, reason, now_ms):
        self.cancelled_attempts.append(attempt_id)
        return await super().cancel(attempt_id, reason, now_ms)


class BlockingCompletionRuntime(InProcessRuntime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.receive_started = asyncio.Event()

    async def receive_completion(self, dispatch_id):
        self.receive_started.set()
        await asyncio.get_running_loop().create_future()


def _command(
    attempt_id: str = "wf:yolo:attempt:1",
    *,
    attempt_no: int = 1,
) -> DispatchCommand:
    item = task("yolo")
    assignment = Assignment(
        "yolo",
        "robot_1",
        ExecutionMode.LOCAL,
        0,
        5,
        5,
        0,
        1,
        "test",
        optimizer_id="test",
        epoch_id="test-epoch",
    )
    demand = ResourceDemand(*task_resource_demand(item, _node_spec()))
    return DispatchCommand(
        attempt_id=attempt_id,
        attempt_no=attempt_no,
        task=item,
        assignment=assignment,
        resource_reservation=PlannedResourceReservation(
            reservation_id=f"plan:test-epoch:yolo:{attempt_id}",
            epoch_id="test-epoch",
            task_id="yolo",
            node_id="robot_1",
            start_ms=0,
            finish_ms=5,
            demand=demand,
        ),
        transfer_reservations=(),
        input_artifact_bindings=(),
        problem_id="test-problem",
        snapshot_id="test-snapshot",
        policy_id="test-policy",
        policy_version="1",
        seed=7,
    )


class RuntimePortTests(unittest.IsolatedAsyncioTestCase):
    async def test_inprocess_runtime_implements_the_only_runtime_contract(self):
        runtime = _runtime()
        self.assertIsInstance(runtime, RuntimePort)

        inventory = await runtime.start(0.0)
        self.assertEqual([node.node_id for node in inventory.nodes], ["robot_1"])
        self.assertEqual(inventory.heartbeats[0].sequence, 1)
        self.assertEqual(inventory.snapshots["robot_1"].node_id, "robot_1")

        refreshed = await runtime.inventory(5.0)
        self.assertEqual(refreshed.heartbeats[0].sequence, 2)
        self.assertEqual(refreshed.heartbeats[0].sampled_at_ms, 5.0)

    async def test_resource_demand_matches_planner_and_runtime(self):
        node = NodeSpec(
            "small-memory-robot",
            NodeKind.ROBOT,
            4,
            2,
            0.01,
            100,
            2,
        )
        snapshot = NodeSnapshot(node.node_id)
        item = TaskInstance(
            "resource-contract",
            "wf",
            "resource-contract",
            node.node_id,
            TaskSpec(
                "resource-contract",
                TaskClass.REALTIME_OFFLOADABLE,
                compute_demand=4,
                gpu_demand=0.5,
            ),
        )
        expected = task_resource_demand(item, node)
        problem = build_scheduling_problem(
            SchedulingEpoch("resource-contract", 0, (item,)),
            node_specs={node.node_id: node},
            node_snapshots={node.node_id: snapshot},
            parent_artifacts={},
            ready_time_ms={item.task_id: 0},
            link_specs=(),
            link_snapshots=(),
        )
        candidate = problem.candidates[item.task_id][0]
        self.assertEqual(
            (
                candidate.resource_demand.cpu_units,
                candidate.resource_demand.gpu_units,
                candidate.resource_demand.memory_gb,
            ),
            expected,
        )

        plan = plan_scheduling_epoch(
            problem.epoch,
            optimizer="greedy_cost",
            node_specs={node.node_id: node},
            node_snapshots={node.node_id: snapshot},
            parent_artifacts={},
            ready_time_ms={item.task_id: 0},
            link_specs=(),
            link_snapshots=(),
        )
        runtime = InProcessRuntime((node,), (snapshot,))
        await runtime.start(0)
        ack = await runtime.dispatch(
            DispatchCommand(
                attempt_id="wf:resource-contract:attempt:1",
                attempt_no=1,
                task=item,
                assignment=plan.assignments[0],
                resource_reservation=plan.node_reservations[0],
                transfer_reservations=plan.transfer_reservations,
                input_artifact_bindings=(
                    problem.input_artifact_bindings[item.task_id]
                ),
                problem_id=plan.problem_id,
                snapshot_id=plan.snapshot_id,
                policy_id=plan.policy_id,
                policy_version=plan.policy_version,
                seed=7,
            )
        )
        self.assertTrue(ack.accepted)
        description = (await runtime.describe(1))[0]
        resources = description["resources"]
        self.assertAlmostEqual(
            resources["available_cpu"],
            node.cpu_capacity - expected[0],
        )
        self.assertAlmostEqual(
            resources["available_gpu"],
            node.gpu_capacity - expected[1],
        )
        self.assertAlmostEqual(
            resources["available_memory_gb"],
            node.memory_gb - expected[2],
        )
        await runtime.receive_completion(ack.dispatch_id)

    async def test_dispatch_command_rejects_a_modified_plan_fragment(self):
        command = _command()

        with self.assertRaisesRegex(
            ValueError,
            "planned compute time",
        ):
            replace(
                command,
                assignment=replace(
                    command.assignment,
                    compute_ms=command.assignment.compute_ms + 1,
                ),
            )

    async def test_runtime_rejects_a_forged_resource_demand(self):
        runtime = _runtime()
        await runtime.start(0.0)
        command = _command()
        forged = replace(
            command,
            resource_reservation=replace(
                command.resource_reservation,
                demand=replace(
                    command.resource_reservation.demand,
                    cpu_units=0.9,
                ),
            ),
        )

        ack = await runtime.dispatch(forged)

        self.assertFalse(ack.accepted)
        self.assertEqual(
            ack.error_code,
            "resource_reservation_mismatch",
        )
        self.assertEqual(
            (await runtime.describe(0.0))[0]["active_reservations"],
            0,
        )

    async def test_concurrency_override_is_advertised_consistently(self):
        node = NodeSpec(
            "robot_1",
            NodeKind.ROBOT,
            4,
            2,
            16,
            100,
            2,
            max_concurrency=1,
        )
        runtime = InProcessRuntime(
            (node,),
            (NodeSnapshot(node.node_id),),
            max_concurrency={node.node_id: 3},
        )

        inventory = await runtime.start(0)
        description = (await runtime.describe(0))[0]

        self.assertEqual(inventory.nodes[0].max_concurrency, 3)
        self.assertEqual(description["max_concurrency"], 3)

    async def test_coordinator_runs_directly_on_the_async_runtime_port(self):
        runtime = _runtime()
        coordinator = CentralCoordinator(runtime)
        workflow = WorkflowSpec("wf", (task("yolo"),))
        report = await coordinator.run_async(workflow)

        self.assertEqual(report.workflow["state"], "succeeded")
        self.assertEqual(report.metrics["task_count"], 1)
        with self.assertRaisesRegex(RuntimeError, "one-shot"):
            await coordinator.run_async(workflow)

    async def test_coordinator_describe_reads_an_immutable_cached_view(self):
        runtime = _runtime()
        coordinator = CentralCoordinator(runtime)
        initialized = await coordinator.initialize_async()
        runtime_before = await runtime.describe(0.0)

        first = coordinator.describe()
        first["agents"].clear()
        second = coordinator.describe()
        runtime_after = await runtime.describe(0.0)

        self.assertEqual(initialized, second)
        self.assertEqual(runtime_before, runtime_after)

    async def test_dispatch_and_completion_are_correlated_to_one_attempt(self):
        runtime = _runtime()
        await runtime.start(0.0)

        ack = await runtime.dispatch(_command())
        self.assertTrue(ack.accepted)
        completion = await runtime.receive_completion(ack.dispatch_id)
        self.assertEqual(completion.dispatch_id, ack.dispatch_id)
        self.assertEqual(completion.attempt_id, ack.attempt_id)
        self.assertEqual(completion.task_id, ack.task_id)
        self.assertEqual(completion.agent_id, ack.agent_id)
        self.assertTrue(completion.ok)
        self.assertEqual((await runtime.describe(10.0))[0]["active_reservations"], 0)

        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            await runtime.receive_completion(ack.dispatch_id)

    async def test_dispatch_exception_triggers_best_effort_cancel(self):
        runtime = _runtime(runtime_type=DispatchRaisesAfterAcceptRuntime)
        coordinator = CentralCoordinator(runtime)

        with self.assertRaisesRegex(ConnectionError, "acknowledgement lost"):
            await coordinator.run_async(WorkflowSpec("wf", (task("yolo"),)))

        self.assertEqual(runtime.cancelled_attempts, ["wf:yolo:attempt:1"])
        self.assertEqual((await runtime.describe(1.0))[0]["active_reservations"], 0)

    async def test_mismatched_completion_triggers_best_effort_cancel(self):
        runtime = _runtime(runtime_type=MismatchedCompletionRuntime)
        coordinator = CentralCoordinator(runtime)

        with self.assertRaisesRegex(RuntimeError, "mismatched completion"):
            await coordinator.run_async(WorkflowSpec("wf", (task("yolo"),)))

        self.assertEqual(runtime.cancelled_attempts, ["wf:yolo:attempt:1"])
        self.assertEqual((await runtime.describe(1.0))[0]["active_reservations"], 0)

    async def test_coordinator_cancellation_releases_the_active_attempt(self):
        runtime = _runtime(runtime_type=BlockingCompletionRuntime)
        coordinator = CentralCoordinator(runtime)
        run = asyncio.create_task(
            coordinator.run_async(WorkflowSpec("wf", (task("yolo"),)))
        )
        await runtime.receive_started.wait()

        run.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run

        self.assertEqual((await runtime.describe(1.0))[0]["active_reservations"], 0)

    async def test_cancel_discards_completion_and_releases_resources(self):
        runtime = _runtime()
        await runtime.start(0.0)

        ack = await runtime.dispatch(_command())
        self.assertTrue(ack.accepted)
        self.assertEqual((await runtime.describe(1.0))[0]["active_reservations"], 1)
        self.assertTrue(await runtime.cancel(ack.attempt_id, "test_cancel", 2.0))
        self.assertEqual((await runtime.describe(2.0))[0]["active_reservations"], 0)

        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            await runtime.receive_completion(ack.dispatch_id)

    async def test_duplicate_dispatch_cannot_replace_the_original_attempt(self):
        runtime = _runtime()
        await runtime.start(0.0)
        command = _command()

        original = await runtime.dispatch(command)
        duplicate = await runtime.dispatch(command)
        self.assertTrue(original.accepted)
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.error_code, "duplicate_attempt")

        completion = await runtime.receive_completion(original.dispatch_id)
        self.assertEqual(completion.attempt_id, command.attempt_id)
        self.assertEqual((await runtime.describe(10.0))[0]["active_reservations"], 0)

        changed_number = await runtime.dispatch(
            _command(command.attempt_id, attempt_no=2)
        )
        self.assertFalse(changed_number.accepted)
        self.assertEqual(changed_number.error_code, "duplicate_attempt")

    async def test_offline_agent_rejects_dispatch_without_resource_leak(self):
        runtime = _runtime(online=False)
        await runtime.start(0.0)

        ack = await runtime.dispatch(_command())
        self.assertFalse(ack.accepted)
        self.assertEqual(ack.error_code, "agent_offline")
        self.assertEqual((await runtime.describe(1.0))[0]["active_reservations"], 0)


class CoordinatorLoopLifecycleTests(unittest.TestCase):
    def test_runtime_cannot_be_reused_across_event_loops(self):
        coordinator = CentralCoordinator(_runtime())
        asyncio.run(coordinator.initialize_async())

        with self.assertRaisesRegex(RuntimeError, "one event loop"):
            coordinator.run(WorkflowSpec("wf", (task("yolo"),)))


if __name__ == "__main__":
    unittest.main()
