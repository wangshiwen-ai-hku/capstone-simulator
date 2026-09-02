"""Architectural boundaries for simulated and deployable runtimes."""

from __future__ import annotations

import unittest
from dataclasses import replace

import mars.runtime as runtime_api
from mars.domain import ExecutionMode, NodeKind, NodeSnapshot
from mars.runtime import (
    DispatchCommand,
    ExecutionAgent,
    InProcessRuntime,
    InProcessRuntimeAdapter,
    RuntimePort,
    SimulatedExecutionAgent,
    SimulationEnvironment,
)

from tests.runtime_contract import (
    ExecutionAgentContractHarness,
    RuntimePortContractHarness,
)
from tests.test_runtime_port import _command, _node_spec, _runtime


def _adapter(*, online: bool = True) -> InProcessRuntimeAdapter:
    return _runtime(
        online=online,
        runtime_type=InProcessRuntimeAdapter,
    )


def _contract_command(attempt_id: str, attempt_no: int) -> DispatchCommand:
    return _command(attempt_id, attempt_no=attempt_no)


def _standalone_agent() -> ExecutionAgent:
    environment = SimulationEnvironment(
        (_node_spec(),),
        (NodeSnapshot("robot_1"),),
    )
    agent = environment.get_agent("robot_1")
    assert agent is not None
    return agent


class RuntimeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.contract = RuntimePortContractHarness(
            runtime_factory=lambda online: _adapter(online=online),
            command_factory=_contract_command,
            agent_id="robot_1",
        )
        self.agent_contract = ExecutionAgentContractHarness(
            agent_factory=_standalone_agent,
            command_factory=_contract_command,
            agent_id="robot_1",
        )

    def test_runtime_boundary_types_are_public_and_structural(self) -> None:
        expected_exports = {
            "ExecutionAgent",
            "InProcessRuntimeAdapter",
            "SimulatedExecutionAgent",
            "SimulationEnvironment",
        }
        self.assertTrue(expected_exports.issubset(runtime_api.__all__))

        adapter = _adapter()
        environment = SimulationEnvironment(
            (_node_spec(),),
            (NodeSnapshot("robot_1"),),
        )
        agent = environment.get_agent("robot_1")
        self.assertIsInstance(adapter, RuntimePort)
        self.assertIsInstance(agent, ExecutionAgent)
        self.assertIsInstance(agent, SimulatedExecutionAgent)
        self.assertIsInstance(environment, SimulationEnvironment)
        self.assertFalse(hasattr(adapter, "environment"))

    def test_legacy_runtime_is_a_compatibility_subclass(self) -> None:
        self.assertIsNot(InProcessRuntime, InProcessRuntimeAdapter)
        self.assertTrue(issubclass(InProcessRuntime, InProcessRuntimeAdapter))

    async def test_adapter_records_private_simulation_execution(self) -> None:
        adapter = _adapter()
        other_adapter = _adapter()
        await adapter.start(0.0)
        command = _command("wf:yolo:boundary:delegation")
        acknowledgement = await adapter.dispatch(command)

        self.assertTrue(acknowledgement.accepted)
        self.assertEqual(
            [item.attempt_id for item in adapter.executions],
            [command.attempt_id],
        )
        self.assertEqual(other_adapter.executions, ())

    async def test_execution_agent_registration_and_heartbeat_contract(
        self,
    ) -> None:
        await self.agent_contract.check_registration_and_heartbeat()

    async def test_execution_agent_dispatch_and_completion_contract(
        self,
    ) -> None:
        await self.agent_contract.check_dispatch_and_completion_correlation()

    async def test_execution_agent_cancellation_contract(self) -> None:
        await self.agent_contract.check_cancellation()

    async def test_contract_start_and_inventory(self) -> None:
        await self.contract.check_start_and_inventory()

    async def test_contract_capabilities_and_reporting(self) -> None:
        await self.contract.check_capabilities_and_reporting()

    async def test_contract_ack_and_completion_correlation(self) -> None:
        await self.contract.check_ack_and_completion_correlation()

    async def test_contract_duplicate_dispatch(self) -> None:
        await self.contract.check_duplicate_dispatch()

    async def test_contract_cancel(self) -> None:
        await self.contract.check_cancel()

    async def test_contract_offline_rejection(self) -> None:
        await self.contract.check_offline_rejection()

    async def test_contract_unknown_agent_rejection(self) -> None:
        await self.contract.check_unknown_agent_rejection()

    async def test_contract_overcapacity_rejection(self) -> None:
        await self.contract.check_overcapacity_rejection()

    async def test_adapter_routes_to_the_selected_agent_in_multi_node_inventory(
        self,
    ) -> None:
        first = _node_spec()
        second = replace(first, node_id="edge_1", kind=NodeKind.EDGE)
        adapter = InProcessRuntimeAdapter(
            (first, second),
            (NodeSnapshot("robot_1"), NodeSnapshot("edge_1")),
        )
        await adapter.start(0.0)
        command = _command("wf:yolo:boundary:multi-node")
        command = replace(
            command,
            assignment=replace(
                command.assignment,
                target_node_id="edge_1",
                execution_mode=ExecutionMode.EDGE,
            ),
            resource_reservation=replace(
                command.resource_reservation,
                node_id="edge_1",
            ),
        )

        acknowledgement = await adapter.dispatch(command)
        self.assertTrue(acknowledgement.accepted)
        self.assertEqual(acknowledgement.agent_id, "edge_1")
        completion = await adapter.receive_completion(
            acknowledgement.dispatch_id
        )
        self.assertEqual(completion.agent_id, "edge_1")


if __name__ == "__main__":
    unittest.main()
