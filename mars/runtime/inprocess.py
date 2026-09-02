"""Coordinator-facing adapter for the deterministic local simulator."""

from __future__ import annotations

from typing import Iterable, Mapping

from ..domain.topology import NodeSnapshot, NodeSpec
from .agent import ExecutionAgent
from .base import (
    AttemptCompletion,
    DispatchAck,
    DispatchCommand,
    RuntimeCapabilities,
    RuntimeInventory,
)
from .simulation import ExecutionInvocation, SimulationEnvironment


class InProcessRuntimeAdapter:
    """Route ``RuntimePort`` operations to local execution agents.

    Simulation mechanics live in :class:`SimulationEnvironment` and
    :class:`~mars.runtime.simulation.SimulatedExecutionAgent`.  This adapter
    owns only coordinator-facing validation, routing, inventory aggregation,
    and global dispatch/attempt correlation.
    """

    capabilities = RuntimeCapabilities(
        discovery=False,
        reliable_control=True,
        feedback=True,
        cancellation=True,
        liveliness=True,
        virtual_time=True,
    )

    def __init__(
        self,
        node_specs: Iterable[NodeSpec],
        snapshots: Iterable[NodeSnapshot] = (),
        *,
        max_concurrency: Mapping[str, int] | None = None,
        supported_task_types: Mapping[str, tuple[str, ...]] | None = None,
        fail_first_task_ids: Iterable[str] = (),
        execution_noise: float = 0.04,
        respect_expected_accuracy: bool = False,
        sample_execution_failures: bool | None = None,
    ) -> None:
        self._environment = SimulationEnvironment(
            node_specs,
            snapshots,
            max_concurrency=max_concurrency,
            supported_task_types=supported_task_types,
            fail_first_task_ids=fail_first_task_ids,
            execution_noise=execution_noise,
            respect_expected_accuracy=respect_expected_accuracy,
            sample_execution_failures=sample_execution_failures,
        )
        self._agents: Mapping[str, ExecutionAgent] = self._environment.agents
        self._node_order = self._environment.node_order
        self._agent_by_dispatch: dict[str, ExecutionAgent] = {}
        self._dispatch_by_attempt: dict[str, str] = {}
        self._consumed_dispatches: set[str] = set()
        self._terminal_attempts: set[str] = set()
        self._cancelled_attempts: dict[str, str] = {}

    @property
    def executions(self) -> tuple[ExecutionInvocation, ...]:
        """Compatibility view used by deterministic simulation tests."""

        return self._environment.executions

    async def start(self, now_ms: float) -> RuntimeInventory:
        for node_id in self._node_order:
            await self._agents[node_id].register(now_ms)
        return await self.inventory(now_ms)

    async def inventory(self, now_ms: float) -> RuntimeInventory:
        heartbeats = []
        for node_id in self._node_order:
            agent = self._agents[node_id]
            if not agent.registered:
                await agent.register(now_ms)
            heartbeats.append(await agent.heartbeat(now_ms))
        return RuntimeInventory(
            nodes=tuple(
                self._agents[node_id].node_spec
                for node_id in self._node_order
            ),
            heartbeats=tuple(heartbeats),
        )

    async def dispatch(self, command: DispatchCommand) -> DispatchAck:
        node_id = command.assignment.target_node_id
        rejection = self._validate_command(command)
        agent = self._environment.get_agent(node_id)
        if rejection or agent is None:
            return self._rejected_ack(
                command,
                node_id,
                rejection or "unknown_agent",
            )

        expected_dispatch_id = (
            f"inprocess:{node_id}:{command.attempt_id}:{command.attempt_no}"
        )
        if (
            expected_dispatch_id in self._agent_by_dispatch
            or expected_dispatch_id in self._consumed_dispatches
            or command.attempt_id in self._dispatch_by_attempt
            or command.attempt_id in self._terminal_attempts
        ):
            return self._rejected_ack(
                command,
                node_id,
                "duplicate_attempt",
            )

        acknowledgement = await agent.dispatch(command)
        if not acknowledgement.accepted:
            return acknowledgement
        if not acknowledgement.dispatch_id:
            raise RuntimeError("accepted dispatch must provide a dispatch id")
        if acknowledgement.dispatch_id in self._agent_by_dispatch:
            await agent.cancel(
                command.attempt_id,
                "duplicate_dispatch_id",
                acknowledgement.scheduled_start_ms or 0.0,
            )
            return self._rejected_ack(
                command,
                node_id,
                "duplicate_attempt",
            )

        self._agent_by_dispatch[acknowledgement.dispatch_id] = agent
        self._dispatch_by_attempt[command.attempt_id] = (
            acknowledgement.dispatch_id
        )
        return acknowledgement

    async def receive_completion(self, dispatch_id: str) -> AttemptCompletion:
        if dispatch_id in self._consumed_dispatches:
            raise RuntimeError(f"completion already consumed: {dispatch_id}")
        agent = self._agent_by_dispatch.get(dispatch_id)
        if agent is None:
            raise KeyError(f"unknown dispatch id: {dispatch_id}")

        completion = await agent.receive_completion(dispatch_id)
        self._agent_by_dispatch.pop(dispatch_id, None)
        self._dispatch_by_attempt.pop(completion.attempt_id, None)
        self._consumed_dispatches.add(dispatch_id)
        self._terminal_attempts.add(completion.attempt_id)
        return completion

    async def cancel(
        self,
        attempt_id: str,
        reason: str,
        now_ms: float,
    ) -> bool:
        dispatch_id = self._dispatch_by_attempt.pop(attempt_id, None)
        if dispatch_id is None:
            return False
        agent = self._agent_by_dispatch.pop(dispatch_id, None)
        if agent is None:
            return False

        cancelled = await agent.cancel(attempt_id, reason, now_ms)
        self._consumed_dispatches.add(dispatch_id)
        self._terminal_attempts.add(attempt_id)
        self._cancelled_attempts[attempt_id] = reason
        return cancelled

    async def describe(self, makespan_ms: float) -> tuple[dict[str, object], ...]:
        return await self._environment.describe(makespan_ms)

    @staticmethod
    def _validate_command(command: DispatchCommand) -> str:
        if command.attempt_no < 1:
            return "invalid_attempt_number"
        if command.task.task_id != command.assignment.task_id:
            return "task_assignment_mismatch"
        if not command.assignment.target_node_id:
            return "missing_target_agent"
        if not command.attempt_id:
            return "missing_attempt_id"
        return ""

    @staticmethod
    def _rejected_ack(
        command: DispatchCommand,
        agent_id: str,
        error_code: str,
    ) -> DispatchAck:
        return DispatchAck(
            dispatch_id="",
            attempt_id=command.attempt_id,
            task_id=command.task.task_id,
            agent_id=agent_id,
            accepted=False,
            error_code=error_code,
        )


class InProcessRuntime(InProcessRuntimeAdapter):
    """Backward-compatible name for :class:`InProcessRuntimeAdapter`."""
