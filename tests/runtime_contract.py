"""Reusable behavioral checks for runtime adapters and execution agents.

In-process, gRPC, and DDS implementations can instantiate these harnesses with
their own factories. Keeping the checks transport-neutral prevents the
simulator from becoming the accidental definition of either contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from mars.runtime import DispatchCommand, ExecutionAgent, RuntimePort


RuntimeFactory = Callable[[bool], RuntimePort]
CommandFactory = Callable[[str, int], DispatchCommand]
AgentFactory = Callable[[], ExecutionAgent]


@dataclass(frozen=True)
class ExecutionAgentContractHarness:
    """Single-node lifecycle checks shared by local and proxy agents."""

    agent_factory: AgentFactory
    command_factory: CommandFactory
    agent_id: str

    async def check_registration_and_heartbeat(self) -> None:
        agent = self.agent_factory()
        assert isinstance(agent, ExecutionAgent)
        assert not agent.registered
        assert await agent.register(0.0)
        assert not await agent.register(0.0)

        heartbeat = await agent.heartbeat(5.0)
        assert heartbeat.agent_id == self.agent_id
        assert heartbeat.snapshot.node_id == self.agent_id
        assert heartbeat.sampled_at_ms == 5.0

    async def check_dispatch_and_completion_correlation(self) -> None:
        agent = self.agent_factory()
        await agent.register(0.0)
        command = self.command_factory("wf:yolo:agent:correlation", 1)

        acknowledgement = await agent.dispatch(command)
        assert acknowledgement.accepted
        assert acknowledgement.attempt_id == command.attempt_id
        assert acknowledgement.agent_id == self.agent_id

        duplicate = await agent.dispatch(command)
        assert not duplicate.accepted
        assert duplicate.error_code == "duplicate_attempt"

        completion = await agent.receive_completion(
            acknowledgement.dispatch_id
        )
        assert completion.dispatch_id == acknowledgement.dispatch_id
        assert completion.attempt_id == command.attempt_id
        assert completion.agent_id == self.agent_id

    async def check_cancellation(self) -> None:
        agent = self.agent_factory()
        await agent.register(0.0)
        command = self.command_factory("wf:yolo:agent:cancel", 1)
        acknowledgement = await agent.dispatch(command)
        assert acknowledgement.accepted

        assert await agent.cancel(command.attempt_id, "contract_test", 1.0)
        assert not await agent.cancel(
            command.attempt_id,
            "contract_test_duplicate",
            1.0,
        )
        try:
            await agent.receive_completion(acknowledgement.dispatch_id)
        except RuntimeError as error:
            assert "already consumed" in str(error)
        else:
            raise AssertionError("cancelled dispatch exposed a completion")


@dataclass(frozen=True)
class RuntimePortContractHarness:
    """Small conformance suite shared by present and future adapters."""

    runtime_factory: RuntimeFactory
    command_factory: CommandFactory
    agent_id: str

    async def check_start_and_inventory(self) -> None:
        runtime = self.runtime_factory(True)
        assert isinstance(runtime, RuntimePort)

        started = await runtime.start(0.0)
        assert tuple(node.node_id for node in started.nodes) == (
            self.agent_id,
        )
        first_heartbeat = started.heartbeats[0]
        assert first_heartbeat.agent_id == self.agent_id
        assert first_heartbeat.snapshot.node_id == self.agent_id

        refreshed = await runtime.inventory(5.0)
        second_heartbeat = refreshed.heartbeats[0]
        assert second_heartbeat.sequence > first_heartbeat.sequence
        assert second_heartbeat.sampled_at_ms == 5.0

    async def check_capabilities_and_reporting(self) -> None:
        runtime = self.runtime_factory(True)
        inventory = await runtime.start(0.0)
        capabilities = runtime.capabilities
        assert all(
            isinstance(value, bool)
            for value in (
                capabilities.discovery,
                capabilities.reliable_control,
                capabilities.feedback,
                capabilities.cancellation,
                capabilities.liveliness,
                capabilities.virtual_time,
            )
        )

        descriptions = await runtime.describe(10.0)
        assert len(descriptions) == len(inventory.nodes)
        assert {item["agent_id"] for item in descriptions} == {
            node.node_id for node in inventory.nodes
        }

    async def check_ack_and_completion_correlation(self) -> None:
        runtime = self.runtime_factory(True)
        await runtime.start(0.0)
        command = self.command_factory("wf:yolo:contract:correlation", 1)

        acknowledgement = await runtime.dispatch(command)
        assert acknowledgement.accepted
        assert acknowledgement.attempt_id == command.attempt_id
        assert acknowledgement.task_id == command.task.task_id
        assert acknowledgement.agent_id == self.agent_id

        completion = await runtime.receive_completion(
            acknowledgement.dispatch_id
        )
        assert completion.dispatch_id == acknowledgement.dispatch_id
        assert completion.attempt_id == acknowledgement.attempt_id
        assert completion.task_id == acknowledgement.task_id
        assert completion.agent_id == acknowledgement.agent_id

    async def check_duplicate_dispatch(self) -> None:
        runtime = self.runtime_factory(True)
        await runtime.start(0.0)
        command = self.command_factory("wf:yolo:contract:duplicate", 1)

        first = await runtime.dispatch(command)
        duplicate = await runtime.dispatch(command)

        assert first.accepted
        assert not duplicate.accepted
        assert duplicate.error_code == "duplicate_attempt"
        completion = await runtime.receive_completion(first.dispatch_id)
        assert completion.attempt_id == command.attempt_id

    async def check_cancel(self) -> None:
        runtime = self.runtime_factory(True)
        await runtime.start(0.0)
        command = self.command_factory("wf:yolo:contract:cancel", 1)
        acknowledgement = await runtime.dispatch(command)
        assert acknowledgement.accepted

        assert await runtime.cancel(command.attempt_id, "contract_test", 1.0)
        assert not await runtime.cancel(
            command.attempt_id,
            "contract_test_duplicate",
            1.0,
        )
        try:
            await runtime.receive_completion(acknowledgement.dispatch_id)
        except RuntimeError as error:
            assert "already consumed" in str(error)
        else:
            raise AssertionError("cancelled dispatch exposed a completion")

    async def check_offline_rejection(self) -> None:
        runtime = self.runtime_factory(False)
        await runtime.start(0.0)
        acknowledgement = await runtime.dispatch(
            self.command_factory("wf:yolo:contract:offline", 1)
        )

        assert not acknowledgement.accepted
        assert acknowledgement.error_code == "agent_offline"

    async def check_unknown_agent_rejection(self) -> None:
        runtime = self.runtime_factory(True)
        await runtime.start(0.0)
        command = self.command_factory("wf:yolo:contract:unknown", 1)
        target_node_id = "missing_runtime_agent"
        command = replace(
            command,
            assignment=replace(
                command.assignment,
                target_node_id=target_node_id,
            ),
            resource_reservation=replace(
                command.resource_reservation,
                node_id=target_node_id,
            ),
        )

        acknowledgement = await runtime.dispatch(command)
        assert not acknowledgement.accepted
        assert acknowledgement.agent_id == target_node_id
        assert acknowledgement.error_code == "unknown_agent"

    async def check_overcapacity_rejection(self) -> None:
        runtime = self.runtime_factory(True)
        inventory = await runtime.start(0.0)
        node = next(
            item for item in inventory.nodes if item.node_id == self.agent_id
        )
        command = self.command_factory("wf:yolo:contract:overcapacity", 1)
        reservation = command.resource_reservation
        overcapacity = replace(
            command,
            resource_reservation=replace(
                reservation,
                demand=replace(
                    reservation.demand,
                    cpu_units=node.cpu_capacity + 1.0,
                ),
            ),
        )

        acknowledgement = await runtime.dispatch(overcapacity)
        assert not acknowledgement.accepted
        assert acknowledgement.error_code == "resources_unavailable"
