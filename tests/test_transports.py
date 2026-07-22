from __future__ import annotations

import unittest

from mars.models import Assignment, ExecutionMode, NodeSnapshot, NodeSpec, WorkflowSpec
from mars.transports.inmemory import InMemoryTransport

from tests.test_mars_core import task
from mars.models import NodeKind


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_inmemory_registration_is_idempotent(self):
        transport = InMemoryTransport()
        spec = NodeSpec(
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
        self.assertTrue(await transport.register(spec))
        self.assertFalse(await transport.register(spec))

    async def test_static_registration_and_dynamic_snapshot_are_separate(self):
        transport = InMemoryTransport()
        spec = NodeSpec("robot_1", NodeKind.ROBOT, 1, 1, 16, 100, 2)
        snapshot = NodeSnapshot("robot_1", gpu_util=0.7, temperature_c=62)
        await transport.register(spec)
        await transport.publish_node_state(snapshot)
        self.assertEqual(transport.registrations["robot_1"], spec)
        self.assertEqual(transport.node_states["robot_1"], snapshot)

    async def test_inmemory_workflow_submission_is_idempotent(self):
        transport = InMemoryTransport()
        workflow = WorkflowSpec("wf", (task("capture"),))
        self.assertEqual(await transport.submit_workflow(workflow), "wf")
        self.assertEqual(await transport.submit_workflow(workflow), "wf")
        self.assertEqual(len(transport.workflows), 1)

    async def test_inmemory_dispatch_preserves_domain_objects(self):
        transport = InMemoryTransport()
        item = task("yolo")
        assignment = Assignment("yolo", "edge_1", ExecutionMode.EDGE, 0, 10, 5, 5, 1, "test")
        dispatch_id = await transport.dispatch(item, assignment)
        self.assertEqual(dispatch_id, "inmemory:yolo:1")
        self.assertEqual(transport.dispatches, [(item, assignment)])


if __name__ == "__main__":
    unittest.main()
