from __future__ import annotations

import unittest

from edgesched.models import Assignment, ExecutionMode, WorkflowSpec
from edgesched.transports.base import NodeRegistration
from edgesched.transports.inmemory import InMemoryTransport

from tests.test_edgesched_core import task
from edgesched.models import NodeKind


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_inmemory_registration_is_idempotent(self):
        transport = InMemoryTransport()
        registration = NodeRegistration("robot_1", NodeKind.ROBOT, "jetson-orin", ("gpu",))
        self.assertTrue(await transport.register(registration))
        self.assertFalse(await transport.register(registration))

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
