"""gRPC implementation of the coordinator-facing runtime contract."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import math
from uuid import uuid4

import grpc

from interfaces.proto.mars.v1 import (
    common_pb2,
    optimization_pb2,
    runtime_pb2,
    runtime_service_pb2,
    runtime_service_pb2_grpc,
    topology_pb2,
    workflow_pb2,
)

from ..domain.artifact import ArtifactRef, InputArtifactBinding
from ..domain.execution import ExecutionMode
from ..domain.task import ResourceClass, TaskClass
from ..domain.topology import NodeKind, NodeSnapshot, NodeSpec
from .base import (
    AgentHeartbeat,
    AttemptCompletion,
    DispatchAck,
    DispatchCommand,
    RuntimeCapabilities,
    RuntimeInventory,
)


_NODE_KIND_TO_PROTO = {
    NodeKind.ROBOT: common_pb2.NODE_KIND_ROBOT,
    NodeKind.EDGE: common_pb2.NODE_KIND_EDGE,
    NodeKind.CLOUD: common_pb2.NODE_KIND_CLOUD,
}
_NODE_KIND_FROM_PROTO = {value: key for key, value in _NODE_KIND_TO_PROTO.items()}
_TASK_CLASS_TO_PROTO = {
    TaskClass.LOCAL_SAFETY: common_pb2.TASK_CLASS_LOCAL_SAFETY,
    TaskClass.REALTIME_OFFLOADABLE: common_pb2.TASK_CLASS_REALTIME_OFFLOADABLE,
    TaskClass.EDGE_HEAVY: common_pb2.TASK_CLASS_EDGE_HEAVY,
}
_RESOURCE_CLASS_TO_PROTO = {
    ResourceClass.CPU: common_pb2.RESOURCE_CLASS_CPU,
    ResourceClass.GPU: common_pb2.RESOURCE_CLASS_GPU,
    ResourceClass.IO: common_pb2.RESOURCE_CLASS_IO,
}
_EXECUTION_MODE_TO_PROTO = {
    ExecutionMode.LOCAL: common_pb2.EXECUTION_MODE_LOCAL,
    ExecutionMode.PEER: common_pb2.EXECUTION_MODE_PEER,
    ExecutionMode.EDGE: common_pb2.EXECUTION_MODE_EDGE,
    ExecutionMode.CLOUD: common_pb2.EXECUTION_MODE_CLOUD,
    ExecutionMode.FALLBACK_LOCAL: common_pb2.EXECUTION_MODE_FALLBACK_LOCAL,
    ExecutionMode.DROP: common_pb2.EXECUTION_MODE_DROP,
}


class GrpcRuntimeAdapter:
    """Translate RuntimePort operations to one gRPC service per agent."""

    capabilities = RuntimeCapabilities(
        discovery=True,
        reliable_control=True,
        feedback=True,
        cancellation=True,
        liveliness=True,
        virtual_time=False,
    )

    def __init__(
        self,
        endpoints: Mapping[str, str],
        *,
        timeout_seconds: float = 5.0,
        completion_timeout_seconds: float = 120.0,
    ) -> None:
        if not endpoints:
            raise ValueError("at least one real agent endpoint is required")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (timeout_seconds, completion_timeout_seconds)
        ):
            raise ValueError("runtime timeouts must be finite and positive")
        self._endpoints = dict(endpoints)
        self._timeout_seconds = timeout_seconds
        self._completion_timeout_seconds = completion_timeout_seconds
        self._channels: dict[str, grpc.aio.Channel] = {}
        self._stubs: dict[str, runtime_service_pb2_grpc.AgentRuntimeStub] = {}
        self._nodes: dict[str, NodeSpec] = {}
        self._stream_tasks: list[asyncio.Task[None]] = []
        self._pending: dict[str, asyncio.Future[AttemptCompletion]] = {}
        self._buffered: dict[str, AttemptCompletion] = {}
        self._early_completions: dict[str, AttemptCompletion] = {}
        self._commands: dict[str, DispatchCommand] = {}
        self._dispatch_attempts: dict[str, str] = {}
        self._accepted_dispatches: dict[str, str] = {}
        self._dispatch_deadlines: dict[str, float] = {}
        self._attempt_agents: dict[str, str] = {}
        self._seen_attempts: set[str] = set()
        self._stream_errors: dict[str, RuntimeError] = {}
        self._completed_by_agent: dict[str, int] = {}
        self._failed_by_agent: dict[str, int] = {}
        self._busy_time_by_agent: dict[str, float] = {}
        self._recorded_dispatches: set[str] = set()
        self._last_inventory: RuntimeInventory | None = None
        self._started = False
        self._has_dispatched = False
        self._closing = False

    async def start(self, now_ms: float) -> RuntimeInventory:
        if self._started:
            return await self.inventory(now_ms)
        instance_id = uuid4().hex
        try:
            for agent_id, endpoint in self._endpoints.items():
                channel = grpc.aio.insecure_channel(endpoint)
                # Track the channel before registration so any failure closes it.
                self._channels[agent_id] = channel
                stub = runtime_service_pb2_grpc.AgentRuntimeStub(channel)
                response = await stub.RegisterAgent(
                    runtime_pb2.RegisterAgentRequest(
                        schema_version="mars.runtime.v1",
                        agent_id=agent_id,
                        agent_instance_id=instance_id,
                        capabilities=_capabilities_to_proto(self.capabilities),
                        sent_at_ms=now_ms,
                    ),
                    timeout=self._timeout_seconds,
                )
                if not response.accepted:
                    raise RuntimeError(
                        f"agent {agent_id} rejected registration: "
                        f"{response.error_code or response.error_message}"
                    )
                node = _node_from_proto(response.node)
                if node.node_id != agent_id:
                    raise RuntimeError(
                        f"agent endpoint {agent_id} returned node {node.node_id}"
                    )
                self._stubs[agent_id] = stub
                self._nodes[agent_id] = node
                self._completed_by_agent[agent_id] = 0
                self._failed_by_agent[agent_id] = 0
                self._busy_time_by_agent[agent_id] = 0.0
                self._stream_tasks.append(
                    asyncio.create_task(self._listen(agent_id, stub))
                )
            self._started = True
            return await self.inventory(now_ms)
        except BaseException:
            await self.close()
            raise

    async def inventory(self, now_ms: float) -> RuntimeInventory:
        if not self._started:
            return await self.start(now_ms)
        if self._stream_errors:
            raise next(iter(self._stream_errors.values()))
        responses = await asyncio.gather(
            *(
                stub.GetState(
                    runtime_service_pb2.GetStateRequest(agent_id=agent_id),
                    timeout=self._timeout_seconds,
                )
                for agent_id, stub in self._stubs.items()
            )
        )
        heartbeats: list[AgentHeartbeat] = []
        for agent_id, response in zip(self._stubs, responses, strict=True):
            if response.error_code:
                raise RuntimeError(
                    f"agent {agent_id} state failed: {response.error_code}"
                )
            if (
                response.heartbeat.agent_id != agent_id
                or response.heartbeat.node_snapshot.node_id != agent_id
            ):
                raise RuntimeError(f"agent {agent_id} returned mismatched state")
            heartbeats.append(_heartbeat_from_proto(response.heartbeat))
        if self._stream_errors:
            raise next(iter(self._stream_errors.values()))
        inventory = RuntimeInventory(
            nodes=tuple(self._nodes.values()),
            heartbeats=tuple(heartbeats),
        )
        self._last_inventory = inventory
        return inventory

    @property
    def nodes(self) -> tuple[NodeSpec, ...]:
        return tuple(self._nodes.values())

    @property
    def snapshots(self) -> tuple[NodeSnapshot, ...]:
        if self._last_inventory is None:
            return ()
        return tuple(self._last_inventory.snapshots.values())

    async def dispatch(self, command: DispatchCommand) -> DispatchAck:
        agent_id = command.assignment.target_node_id
        stub = self._stubs.get(agent_id)
        if stub is None or self._closing:
            return DispatchAck(
                dispatch_id="",
                attempt_id=command.attempt_id,
                task_id=command.task.task_id,
                agent_id=agent_id,
                accepted=False,
                error_code="unknown_agent",
            )
        if agent_id in self._stream_errors:
            raise self._stream_errors[agent_id]
        if command.attempt_id in self._seen_attempts:
            return DispatchAck(
                dispatch_id="",
                attempt_id=command.attempt_id,
                task_id=command.task.task_id,
                agent_id=agent_id,
                accepted=False,
                error_code="duplicate_attempt",
            )
        self._seen_attempts.add(command.attempt_id)
        self._commands[command.attempt_id] = command
        # Track before the RPC: a fast completion can precede its acknowledgement,
        # and a lost acknowledgement must still allow best-effort cancellation.
        self._attempt_agents[command.attempt_id] = agent_id
        deadline = asyncio.get_running_loop().time() + self._completion_timeout_seconds
        try:
            response = await stub.DispatchTask(
                _command_to_proto(command),
                timeout=self._timeout_seconds,
            )
        except BaseException:
            self._commands.pop(command.attempt_id, None)
            self._early_completions.pop(command.attempt_id, None)
            raise
        if (
            response.attempt_id != command.attempt_id
            or response.task_id != command.task.task_id
            or response.agent_id != agent_id
            or (response.accepted and not response.dispatch_id)
        ):
            raise RuntimeError(f"agent {agent_id} returned mismatched dispatch ack")
        if response.accepted:
            if (
                response.dispatch_id in self._dispatch_attempts
                or response.dispatch_id in self._recorded_dispatches
            ):
                raise RuntimeError("agent reused a dispatch id")
            self._dispatch_attempts[response.dispatch_id] = command.attempt_id
            self._accepted_dispatches[command.attempt_id] = response.dispatch_id
            self._dispatch_deadlines[response.dispatch_id] = deadline
            self._has_dispatched = True
            early = self._early_completions.pop(command.attempt_id, None)
            if early is not None and early.dispatch_id == response.dispatch_id:
                self._record_completion(early)
        else:
            self._commands.pop(command.attempt_id, None)
            self._attempt_agents.pop(command.attempt_id, None)
            self._early_completions.pop(command.attempt_id, None)
        return DispatchAck(
            dispatch_id=response.dispatch_id,
            attempt_id=response.attempt_id,
            task_id=response.task_id,
            agent_id=response.agent_id,
            accepted=response.accepted,
            error_code=response.error_code or response.error_message,
        )

    async def receive_completion(self, dispatch_id: str) -> AttemptCompletion:
        attempt_id = self._dispatch_attempts.get(dispatch_id)
        if attempt_id is None:
            raise ValueError(f"unknown or consumed dispatch: {dispatch_id}")
        if dispatch_id in self._pending:
            raise ValueError(f"completion already being received: {dispatch_id}")
        command = self._commands[attempt_id]
        agent_id = command.assignment.target_node_id
        future = asyncio.get_running_loop().create_future()
        self._pending[dispatch_id] = future
        try:
            buffered = self._buffered.pop(dispatch_id, None)
            if buffered is not None:
                return buffered
            if agent_id in self._stream_errors:
                raise self._stream_errors[agent_id]
            remaining = (
                self._dispatch_deadlines[dispatch_id]
                - asyncio.get_running_loop().time()
            )
            try:
                return await asyncio.wait_for(future, max(0.0, remaining))
            except asyncio.TimeoutError as exc:
                # Before Python 3.11, asyncio.TimeoutError is distinct from
                # built-in TimeoutError. Normalize the public adapter failure.
                # A timeout is an unknown outcome, not a safe automatic retry.
                # The coordinator aborts the run and releases other attempts.
                try:
                    await self.cancel(
                        attempt_id,
                        "completion_timeout",
                        command.assignment.estimated_start_ms,
                    )
                except grpc.RpcError:
                    pass
                raise TimeoutError(
                    f"completion timeout for {dispatch_id} on {agent_id}; "
                    "execution outcome may be unknown"
                ) from exc
        finally:
            self._pending.pop(dispatch_id, None)
            self._dispatch_attempts.pop(dispatch_id, None)
            self._accepted_dispatches.pop(attempt_id, None)
            self._dispatch_deadlines.pop(dispatch_id, None)
            self._commands.pop(attempt_id, None)

    async def cancel(
        self,
        attempt_id: str,
        reason: str,
        now_ms: float,
    ) -> bool:
        agent_id = self._attempt_agents.get(attempt_id)
        if agent_id is None:
            return False
        response = await self._stubs[agent_id].CancelAttempt(
            runtime_pb2.CancelAttemptRequest(
                attempt_id=attempt_id,
                reason=reason,
                requested_at_ms=now_ms,
            ),
            timeout=self._timeout_seconds,
        )
        if response.cancelled:
            self._attempt_agents.pop(attempt_id, None)
        return response.cancelled

    async def describe(
        self,
        makespan_ms: float,
    ) -> tuple[dict[str, object], ...]:
        inventory = await self.inventory(makespan_ms)
        descriptions = tuple(
            {
                "agent_id": node.node_id,
                "kind": node.kind.value,
                "architecture": node.architecture,
                "registered": True,
                "online": inventory.snapshots[node.node_id].online,
                "heartbeat_sequence": next(
                    item.sequence
                    for item in inventory.heartbeats
                    if item.agent_id == node.node_id
                ),
                "last_heartbeat_ms": round(
                    next(
                        item.sampled_at_ms
                        for item in inventory.heartbeats
                        if item.agent_id == node.node_id
                    ),
                    4,
                ),
                "active_reservations": next(
                    item.active_reservations
                    for item in inventory.heartbeats
                    if item.agent_id == node.node_id
                ),
                "max_concurrency": node.max_concurrency,
                "completed_attempts": self._completed_by_agent[node.node_id],
                "failed_attempts": self._failed_by_agent[node.node_id],
                "busy_time_ms": self._busy_time_by_agent[node.node_id],
                "utilization": inventory.snapshots[node.node_id].gpu_util,
                "capabilities": list(node.capabilities),
                "supported_models": list(node.supported_models),
                "endpoint": self._endpoints[node.node_id],
                "resources": {
                    "cpu_capacity": node.cpu_capacity,
                    "gpu_capacity": node.gpu_capacity,
                    "memory_gb": node.memory_gb,
                },
            }
            for node in inventory.nodes
        )
        if self._has_dispatched and not self._dispatch_attempts:
            await self.close()
        return descriptions

    async def close(self) -> None:
        self._closing = True
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("runtime closed before completion"))
        await asyncio.gather(
            *(
                self.cancel(attempt_id, "runtime_closed", 0.0)
                for attempt_id in tuple(self._attempt_agents)
            ),
            return_exceptions=True,
        )
        for task in self._stream_tasks:
            task.cancel()
        if self._stream_tasks:
            await asyncio.gather(*self._stream_tasks, return_exceptions=True)
        self._stream_tasks.clear()
        await asyncio.gather(*(channel.close() for channel in self._channels.values()))
        self._channels.clear()
        self._stubs.clear()
        self._commands.clear()
        self._dispatch_attempts.clear()
        self._accepted_dispatches.clear()
        self._dispatch_deadlines.clear()
        self._attempt_agents.clear()
        self._early_completions.clear()
        self._buffered.clear()
        self._seen_attempts.clear()
        self._stream_errors.clear()
        self._recorded_dispatches.clear()
        self._started = False
        self._has_dispatched = False
        self._closing = False

    async def _listen(
        self,
        agent_id: str,
        stub: runtime_service_pb2_grpc.AgentRuntimeStub,
    ) -> None:
        try:
            stream = stub.StreamCompletions(
                runtime_service_pb2.CompletionSubscription(agent_id=agent_id)
            )
            async for message in stream:
                command = self._commands.get(message.attempt_id)
                if command is None:
                    # Agents may replay results from other runtime sessions.
                    continue
                if (
                    message.agent_id != agent_id
                    or command.assignment.target_node_id != agent_id
                    or message.task_id != command.task.task_id
                    or not message.dispatch_id
                ):
                    raise RuntimeError("completion does not match its command")
                completion = _completion_from_proto(message)
                dispatch_id = self._accepted_dispatches.get(message.attempt_id)
                if dispatch_id is None:
                    self._early_completions.setdefault(message.attempt_id, completion)
                elif message.dispatch_id == dispatch_id:
                    self._record_completion(completion)
            if not self._closing:
                raise RuntimeError("completion stream ended")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = RuntimeError(
                f"agent {agent_id} completion stream failed: {exc}; "
                "execution outcome may be unknown"
            )
            self._stream_errors[agent_id] = error
            for dispatch_id, future in tuple(self._pending.items()):
                attempt_id = self._dispatch_attempts.get(dispatch_id)
                command = self._commands.get(attempt_id)
                if (
                    command is not None
                    and command.assignment.target_node_id == agent_id
                    and not future.done()
                ):
                    future.set_exception(error)

    def _record_completion(self, completion: AttemptCompletion) -> None:
        if completion.dispatch_id in self._recorded_dispatches:
            return
        if (
            asyncio.get_running_loop().time()
            > self._dispatch_deadlines[completion.dispatch_id]
        ):
            # The deadline applies to the dispatch, not to when a caller starts
            # receiving. Never turn a late result into a successful validation.
            return
        self._recorded_dispatches.add(completion.dispatch_id)
        counter = self._completed_by_agent if completion.ok else self._failed_by_agent
        counter[completion.agent_id] += 1
        self._busy_time_by_agent[completion.agent_id] += completion.compute_time_ms
        self._attempt_agents.pop(completion.attempt_id, None)
        future = self._pending.get(completion.dispatch_id)
        if future is not None:
            if not future.done():
                future.set_result(completion)
        else:
            self._buffered[completion.dispatch_id] = completion


def _capabilities_to_proto(
    value: RuntimeCapabilities,
) -> runtime_pb2.RuntimeCapabilities:
    return runtime_pb2.RuntimeCapabilities(
        discovery=value.discovery,
        reliable_control=value.reliable_control,
        feedback=value.feedback,
        cancellation=value.cancellation,
        liveliness=value.liveliness,
        virtual_time=value.virtual_time,
    )


def _node_from_proto(value: topology_pb2.NodeSpec) -> NodeSpec:
    return NodeSpec(
        node_id=value.node_id,
        kind=_NODE_KIND_FROM_PROTO[value.kind],
        cpu_capacity=value.cpu_capacity_units,
        gpu_capacity=value.gpu_capacity_units,
        memory_gb=value.memory_capacity_gb,
        bandwidth_mbps=value.network_bandwidth_mbps,
        base_latency_ms=value.base_latency_ms,
        architecture=value.architecture
        if value.HasField("architecture")
        else "generic",
        battery_capacity_wh=(
            value.battery_capacity_wh if value.HasField("battery_capacity_wh") else None
        ),
        safety_capable=(
            value.safety_capable if value.HasField("safety_capable") else True
        ),
        capabilities=tuple(value.capabilities),
        supported_models=tuple(value.supported_models),
        max_concurrency=(
            value.max_concurrency if value.HasField("max_concurrency") else 1
        ),
    )


def _heartbeat_from_proto(value: runtime_pb2.AgentHeartbeat) -> AgentHeartbeat:
    snapshot = value.node_snapshot
    return AgentHeartbeat(
        agent_id=value.agent_id,
        sequence=value.sequence,
        sampled_at_ms=value.sampled_at_ms,
        snapshot=NodeSnapshot(
            node_id=snapshot.node_id,
            cpu_util=snapshot.cpu_utilization_ratio,
            gpu_util=snapshot.gpu_utilization_ratio,
            memory_util=snapshot.memory_utilization_ratio,
            temperature_c=snapshot.temperature_celsius,
            power_w=snapshot.power_watts,
            network_latency_ms=snapshot.network_latency_ms,
            online=snapshot.online if snapshot.HasField("online") else True,
        ),
        active_reservations=value.active_reservation_count,
    )


def _command_to_proto(command: DispatchCommand) -> runtime_pb2.DispatchCommand:
    task = command.task
    spec = task.spec
    task_message = workflow_pb2.TaskInstance(
        task_id=task.task_id,
        workflow_id=task.workflow_id,
        name=task.name,
        source_node_id=task.source_node_id,
        dependency_task_ids=task.dependency_task_ids,
        priority=task.priority,
        stage_index=task.stage_index,
        arrival_time_ms=task.arrival_time_ms,
        deadline_time_ms=task.deadline_time_ms,
        expected_accuracy=task.expected_accuracy,
        input_ref=task.input_ref,
    )
    task_message.spec.CopyFrom(
        workflow_pb2.TaskSpec(
            task_type=spec.task_type,
            task_class=_TASK_CLASS_TO_PROTO[spec.task_class],
            compute_demand=spec.compute_demand,
            gpu_demand=spec.gpu_demand,
            latency_budget_ms=spec.latency_budget_ms,
            model_requirement=spec.model_requirement,
            input_size_mb=spec.input_size_mb,
            output_size_mb=spec.output_size_mb,
            bandwidth_requirement_mbps=spec.bandwidth_requirement_mbps,
            energy_budget_j=spec.energy_budget_j,
            dominant_resource=_RESOURCE_CLASS_TO_PROTO[spec.dominant_resource],
            allow_local_fallback=spec.allow_local_fallback,
            input_ports=[
                workflow_pb2.DataPort(name=item.name, message_type=item.message_type)
                for item in spec.input_ports
            ],
            output_ports=[
                workflow_pb2.DataPort(name=item.name, message_type=item.message_type)
                for item in spec.output_ports
            ],
        )
    )
    if spec.placement_constraints is not None:
        placement = spec.placement_constraints
        task_message.spec.placement_constraints.CopyFrom(
            workflow_pb2.PlacementConstraints(
                pinned_node_id=placement.pinned_node_id,
                allowed_node_kinds=[
                    _NODE_KIND_TO_PROTO[item] for item in placement.allowed_node_kinds
                ],
                preferred_node_kinds=[
                    _NODE_KIND_TO_PROTO[item] for item in placement.preferred_node_kinds
                ],
                required_capabilities=placement.required_capabilities,
                allow_source_node=placement.allow_source_node,
                allow_other_robots=placement.allow_other_robots,
                safety_required=placement.safety_required,
                allow_fallback=placement.allow_fallback,
                stateful=placement.stateful,
                idempotent=placement.idempotent,
                splittable=placement.splittable,
                replicable=placement.replicable,
            )
        )
    assignment = command.assignment
    assignment_message = optimization_pb2.Assignment(
        task_id=assignment.task_id,
        target_node_id=assignment.target_node_id,
        execution_mode=_EXECUTION_MODE_TO_PROTO[assignment.execution_mode],
        estimated_start_time_ms=assignment.estimated_start_ms,
        estimated_finish_time_ms=assignment.estimated_finish_ms,
        compute_time_ms=assignment.compute_ms,
        communication_time_ms=assignment.communication_ms,
        energy_j=assignment.energy_j,
        reason=assignment.reason,
        input_node_ids=assignment.input_locations,
        transfer_link_ids=assignment.transfer_link_ids,
        optimizer_id=assignment.optimizer_id,
        epoch_id=assignment.epoch_id,
        output_size_mb=assignment.output_size_mb,
        success_probability_ratio=assignment.success_probability,
    )
    resource = command.resource_reservation
    return runtime_pb2.DispatchCommand(
        schema_version="mars.runtime.v1",
        attempt_id=command.attempt_id,
        attempt_number=command.attempt_no,
        task=task_message,
        assignment=assignment_message,
        resource_reservation=topology_pb2.PlannedResourceReservation(
            reservation_id=resource.reservation_id,
            epoch_id=resource.epoch_id,
            task_id=resource.task_id,
            node_id=resource.node_id,
            start_time_ms=resource.start_ms,
            finish_time_ms=resource.finish_ms,
            demand=topology_pb2.ResourceDemand(
                cpu_units=resource.demand.cpu_units,
                gpu_units=resource.demand.gpu_units,
                memory_gb=resource.demand.memory_gb,
            ),
        ),
        transfer_reservations=[
            topology_pb2.TransferReservation(
                reservation_id=item.reservation_id,
                epoch_id=item.epoch_id,
                task_id=item.task_id,
                transfer_id=item.transfer_id,
                path_link_ids=item.path_link_ids,
                start_time_ms=item.start_ms,
                finish_time_ms=item.finish_ms,
                size_mb=item.size_mb,
            )
            for item in command.transfer_reservations
        ],
        input_artifact_bindings=[
            _binding_to_proto(item) for item in command.input_artifact_bindings
        ],
        random_seed=command.seed,
        inject_failure=command.inject_failure,
        problem_id=command.problem_id,
        snapshot_id=command.snapshot_id,
        policy_id=command.policy_id,
        policy_version=command.policy_version,
        solve_request_id=command.solve_request_id,
    )


def _binding_to_proto(
    value: InputArtifactBinding,
) -> workflow_pb2.InputArtifactBinding:
    return workflow_pb2.InputArtifactBinding(
        consumer_task_id=value.consumer_task_id,
        consumer_port=value.consumer_port,
        artifact=_artifact_to_proto(value.artifact),
    )


def _artifact_to_proto(value: ArtifactRef) -> workflow_pb2.ArtifactRef:
    return workflow_pb2.ArtifactRef(
        artifact_id=value.artifact_id,
        producer_task_id=value.producer_task_id,
        node_id=value.node_id,
        size_mb=value.size_mb,
        uri=value.uri,
        checksum=value.checksum,
        producer_port=value.producer_port,
        message_type=value.message_type,
    )


def _completion_from_proto(
    value: runtime_pb2.AttemptCompletion,
) -> AttemptCompletion:
    return AttemptCompletion(
        dispatch_id=value.dispatch_id,
        attempt_id=value.attempt_id,
        task_id=value.task_id,
        agent_id=value.agent_id,
        ok=value.ok,
        started_time_ms=value.started_time_ms,
        finished_time_ms=value.finished_time_ms,
        compute_time_ms=value.compute_time_ms,
        energy_j=value.energy_j,
        outputs=tuple(
            ArtifactRef(
                artifact_id=item.artifact_id,
                producer_task_id=item.producer_task_id,
                node_id=item.node_id,
                size_mb=item.size_mb,
                uri=item.uri,
                checksum=item.checksum,
                producer_port=(
                    item.producer_port if item.HasField("producer_port") else "result"
                ),
                message_type=item.message_type,
            )
            for item in value.outputs
        ),
        error_code=value.error_code or value.error_message,
    )


__all__ = ["GrpcRuntimeAdapter"]
