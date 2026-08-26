"""Small gRPC agent that executes validated dispatches on localhost."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from time import monotonic
from typing import Any

import grpc

from interfaces.proto.mars.v1 import (
    common_pb2,
    runtime_pb2,
    runtime_service_pb2,
    runtime_service_pb2_grpc,
    topology_pb2,
    workflow_pb2,
)


_NODE_KINDS = {
    "robot": common_pb2.NODE_KIND_ROBOT,
    "edge": common_pb2.NODE_KIND_EDGE,
    "cloud": common_pb2.NODE_KIND_CLOUD,
}


@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    listen: str
    node: dict[str, Any]
    snapshot: dict[str, Any]
    execution_delay_ms: float = 1.0


def load_agent_configs(path: str | Path) -> tuple[AgentConfig, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        AgentConfig(
            agent_id=item["agent_id"],
            listen=item["listen"],
            node=dict(item["node"]),
            snapshot=dict(item["snapshot"]),
            execution_delay_ms=float(item.get("execution_delay_ms", 1.0)),
        )
        for item in payload["agents"]
    )


class MockAgentService(runtime_service_pb2_grpc.AgentRuntimeServicer):
    """Serve one configured execution node and simulate accepted attempts."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.node = _node_message(config)
        self._started = monotonic()
        self._sequence = 0
        self._active: dict[str, asyncio.Task[None]] = {}
        self._subscribers: set[asyncio.Queue[runtime_pb2.AttemptCompletion]] = set()
        self._history: list[runtime_pb2.AttemptCompletion] = []

    async def RegisterAgent(self, request, context):
        if request.agent_id != self.config.agent_id:
            return runtime_pb2.RegisterAgentResponse(
                accepted=False,
                error_code="agent_id_mismatch",
                error_message=(
                    f"configured agent is {self.config.agent_id}, "
                    f"not {request.agent_id}"
                ),
            )
        return runtime_pb2.RegisterAgentResponse(
            accepted=True,
            registration_id=f"mock:{self.config.agent_id}",
            control_plane_time_ms=self._now_ms(),
            heartbeat_interval_ms=1000.0,
            node=self.node,
        )

    async def GetState(self, request, context):
        if request.agent_id != self.config.agent_id:
            return runtime_service_pb2.GetStateResponse(
                error_code="agent_id_mismatch",
                error_message=f"unknown agent {request.agent_id}",
            )
        self._sequence += 1
        now_ms = self._now_ms()
        snapshot = self.config.snapshot
        return runtime_service_pb2.GetStateResponse(
            heartbeat=runtime_pb2.AgentHeartbeat(
                agent_id=self.config.agent_id,
                sequence=self._sequence,
                sampled_at_ms=now_ms,
                node_snapshot=topology_pb2.NodeSnapshot(
                    node_id=self.config.agent_id,
                    cpu_utilization_ratio=snapshot["cpu_util"],
                    gpu_utilization_ratio=snapshot["gpu_util"],
                    memory_utilization_ratio=snapshot["memory_util"],
                    temperature_celsius=snapshot["temperature_c"],
                    power_watts=snapshot["power_w"],
                    network_latency_ms=snapshot["network_latency_ms"],
                    online=True,
                    sampled_at_ms=now_ms,
                    snapshot_sequence=self._sequence,
                    active_task_count=len(self._active),
                    queue_depth=0,
                ),
                active_reservation_count=len(self._active),
            )
        )

    async def DispatchTask(self, request, context):
        if request.assignment.target_node_id != self.config.agent_id:
            return _rejected_ack(request, self.config.agent_id, "wrong_target")
        if request.task.task_id != request.assignment.task_id:
            return _rejected_ack(request, self.config.agent_id, "task_mismatch")
        if request.attempt_id in self._active:
            return _rejected_ack(request, self.config.agent_id, "duplicate_attempt")
        dispatch_id = (
            f"grpc:{self.config.agent_id}:{request.attempt_id}:"
            f"{request.attempt_number}"
        )
        self._active[request.attempt_id] = asyncio.create_task(
            self._complete(dispatch_id, request)
        )
        return runtime_pb2.DispatchAck(
            dispatch_id=dispatch_id,
            attempt_id=request.attempt_id,
            task_id=request.task.task_id,
            agent_id=self.config.agent_id,
            accepted=True,
        )

    async def StreamCompletions(self, request, context):
        if request.agent_id != self.config.agent_id:
            return
        queue: asyncio.Queue[runtime_pb2.AttemptCompletion] = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            for completion in self._history:
                yield completion
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    async def CancelAttempt(self, request, context):
        task = self._active.pop(request.attempt_id, None)
        if task is None:
            return runtime_pb2.CancelAttemptResponse(
                attempt_id=request.attempt_id,
                cancelled=False,
                error_code="unknown_attempt",
            )
        task.cancel()
        return runtime_pb2.CancelAttemptResponse(
            attempt_id=request.attempt_id,
            cancelled=True,
        )

    async def _complete(self, dispatch_id, request) -> None:
        try:
            await asyncio.sleep(self.config.execution_delay_ms / 1000.0)
            assignment = request.assignment
            ok = not request.inject_failure
            completion = runtime_pb2.AttemptCompletion(
                dispatch_id=dispatch_id,
                attempt_id=request.attempt_id,
                task_id=request.task.task_id,
                agent_id=self.config.agent_id,
                ok=ok,
                started_time_ms=assignment.estimated_start_time_ms,
                finished_time_ms=assignment.estimated_finish_time_ms,
                compute_time_ms=assignment.compute_time_ms,
                energy_j=assignment.energy_j,
                outputs=(
                    _outputs(request, self.config.agent_id)
                    if ok
                    else []
                ),
                error_code=(
                    "injected_first_attempt_failure"
                    if request.inject_failure
                    else ""
                ),
            )
            self._history.append(completion)
            for queue in tuple(self._subscribers):
                queue.put_nowait(completion)
        finally:
            self._active.pop(request.attempt_id, None)

    def _now_ms(self) -> float:
        return (monotonic() - self._started) * 1000.0


def _node_message(config: AgentConfig) -> topology_pb2.NodeSpec:
    node = config.node
    message = topology_pb2.NodeSpec(
        node_id=config.agent_id,
        kind=_NODE_KINDS[node["kind"]],
        cpu_capacity_units=node["cpu_capacity"],
        gpu_capacity_units=node["gpu_capacity"],
        memory_capacity_gb=node["memory_gb"],
        network_bandwidth_mbps=node["bandwidth_mbps"],
        base_latency_ms=node["base_latency_ms"],
        architecture=node.get("architecture", "generic"),
        safety_capable=node.get("safety_capable", True),
        capabilities=node.get("capabilities", []),
        supported_models=node.get("supported_models", []),
        max_concurrency=node.get("max_concurrency", 1),
    )
    if node.get("battery_capacity_wh") is not None:
        message.battery_capacity_wh = node["battery_capacity_wh"]
    return message


def _outputs(request, agent_id: str) -> list[workflow_pb2.ArtifactRef]:
    ports = list(request.task.spec.output_ports) or [
        workflow_pb2.DataPort(name="result", message_type="")
    ]
    size = request.assignment.output_size_mb / len(ports)
    return [
        workflow_pb2.ArtifactRef(
            artifact_id=(
                f"artifact:{request.task.workflow_id}:"
                f"{request.task.task_id}:{port.name}"
            ),
            producer_task_id=request.task.task_id,
            node_id=agent_id,
            size_mb=size,
            uri=(
                f"agent://{agent_id}/{request.task.workflow_id}/"
                f"{request.task.task_id}/{port.name}"
            ),
            producer_port=port.name,
            message_type=port.message_type,
        )
        for port in ports
    ]


def _rejected_ack(request, agent_id: str, code: str) -> runtime_pb2.DispatchAck:
    return runtime_pb2.DispatchAck(
        attempt_id=request.attempt_id,
        task_id=request.task.task_id,
        agent_id=agent_id,
        accepted=False,
        error_code=code,
    )


async def start_agent_server(
    config: AgentConfig,
) -> tuple[grpc.aio.Server, int]:
    server = grpc.aio.server()
    runtime_service_pb2_grpc.add_AgentRuntimeServicer_to_server(
        MockAgentService(config),
        server,
    )
    port = server.add_insecure_port(config.listen)
    await server.start()
    return server, port


__all__ = [
    "AgentConfig",
    "MockAgentService",
    "load_agent_configs",
    "start_agent_server",
]
