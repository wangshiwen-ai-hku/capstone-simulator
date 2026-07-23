"""In-process implementation of the MARS runtime port."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from ..models import (
    ArtifactRef,
    Assignment,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    TaskInstance,
    resolved_placement_constraints,
    task_resource_demand,
)
from .base import (
    AgentHeartbeat,
    AttemptCompletion,
    DispatchAck,
    DispatchCommand,
    RuntimeCapabilities,
    RuntimeInventory,
)


@dataclass(frozen=True)
class _ResourceReservation:
    reservation_id: str
    attempt_id: str
    task_id: str
    node_id: str
    scheduled_start_ms: float
    cpu_units: float
    gpu_units: float
    memory_gb: float


@dataclass(frozen=True)
class ExecutionInvocation:
    attempt_id: str
    task_id: str
    attempt_no: int
    input_artifacts: tuple[ArtifactRef, ...]
    injected_failure: bool


@dataclass(frozen=True)
class _ExecutionResult:
    attempt_id: str
    task_id: str
    agent_id: str
    ok: bool
    compute_time_ms: float
    energy_j: float
    outputs: tuple[ArtifactRef, ...]
    error_code: str = ""


class _SimulatedAgent:
    """One fake Orin or edge executor with explicit capacity accounting.

    Execution is virtual-time based: ``execute`` returns a repeatable sampled
    duration immediately, while the coordinator decides when that completion
    becomes visible.  No wall-clock sleep or external service is required.
    """

    def __init__(
        self,
        node_spec: NodeSpec,
        snapshot: NodeSnapshot | None = None,
        *,
        max_concurrency: int = 1,
        supported_task_types: tuple[str, ...] = (),
        fail_first_task_ids: tuple[str, ...] = (),
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        if snapshot is not None and snapshot.node_id != node_spec.node_id:
            raise ValueError("agent snapshot must match node_spec.node_id")
        self._node_spec = node_spec
        self._base_snapshot = snapshot or NodeSnapshot(node_spec.node_id)
        self.max_concurrency = max_concurrency
        self.supported_task_types = frozenset(supported_task_types)
        self.fail_first_task_ids = frozenset(fail_first_task_ids)
        self.executions: list[ExecutionInvocation] = []
        self._registered = False
        self._heartbeat_sequence = 0
        self._last_heartbeat_ms = 0.0
        self._reservations: dict[str, _ResourceReservation] = {}
        self._busy_ms = 0.0
        self._completed_attempts = 0
        self._failed_attempts = 0

    @property
    def node_spec(self) -> NodeSpec:
        return self._node_spec

    @property
    def snapshot(self) -> NodeSnapshot:
        cpu, gpu, memory = self._reserved_totals()
        return replace(
            self._base_snapshot,
            cpu_util=max(
                self._base_snapshot.cpu_util,
                min(1.0, cpu / max(1e-9, self.node_spec.cpu_capacity)),
            ),
            gpu_util=max(
                self._base_snapshot.gpu_util,
                min(1.0, gpu / max(1e-9, self.node_spec.gpu_capacity))
                if self.node_spec.gpu_capacity > 0
                else 0.0,
            ),
            memory_util=max(
                self._base_snapshot.memory_util,
                min(1.0, memory / max(1e-9, self.node_spec.memory_gb)),
            ),
        )

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def active_reservation_count(self) -> int:
        return len(self._reservations)

    def register(self, now_ms: float) -> bool:
        newly_registered = not self._registered
        self._registered = True
        self._last_heartbeat_ms = now_ms
        return newly_registered

    def heartbeat(self, now_ms: float) -> AgentHeartbeat:
        if not self._registered:
            raise RuntimeError(f"agent {self.node_spec.node_id} is not registered")
        self._heartbeat_sequence += 1
        self._last_heartbeat_ms = now_ms
        return AgentHeartbeat(
            agent_id=self.node_spec.node_id,
            sequence=self._heartbeat_sequence,
            sampled_at_ms=now_ms,
            snapshot=self.snapshot,
            active_reservations=len(self._reservations),
        )

    def can_execute(self, task: TaskInstance) -> tuple[bool, str]:
        if not self._registered:
            return False, "agent_not_registered"
        if not self._base_snapshot.online:
            return False, "agent_offline"
        constraints = resolved_placement_constraints(task)
        if constraints.pinned_node_id:
            if self.node_spec.node_id != constraints.pinned_node_id:
                return False, "placement_constraints_reject_agent"
            if (
                constraints.allowed_node_kinds
                and self.node_spec.kind
                not in constraints.allowed_node_kinds
            ):
                return False, "placement_constraints_reject_agent"
        elif self.node_spec.node_id == task.source_node_id:
            if not constraints.allow_source_node:
                return False, "placement_constraints_reject_agent"
        else:
            if (
                self.node_spec.kind is NodeKind.ROBOT
                and not constraints.allow_other_robots
            ):
                return False, "placement_constraints_reject_agent"
            if self.node_spec.kind not in constraints.allowed_node_kinds:
                return False, "placement_constraints_reject_agent"
        if constraints.safety_required and not self.node_spec.safety_capable:
            return False, "safety_capability_required"
        capabilities = set(self.node_spec.capabilities)
        if self.node_spec.safety_capable:
            capabilities.add("local_safety")
        if not set(constraints.required_capabilities).issubset(
            capabilities
        ):
            return False, "required_capability_unavailable"
        if self.supported_task_types and task.spec.task_type not in self.supported_task_types:
            return False, "task_capability_not_declared"
        if task.spec.gpu_demand > 0 and self.node_spec.gpu_capacity <= 0:
            return False, "gpu_unavailable"
        if task.spec.gpu_demand > self.node_spec.gpu_capacity + 1e-9:
            return False, "gpu_capacity_insufficient"
        model = task.spec.model_requirement
        if model and self.node_spec.supported_models and model not in self.node_spec.supported_models:
            return False, "model_not_supported"
        return True, ""

    def reserve(
        self,
        task: TaskInstance,
        attempt_id: str,
        scheduled_start_ms: float,
    ) -> _ResourceReservation | None:
        feasible, _ = self.can_execute(task)
        if not feasible or any(
            item.attempt_id == attempt_id for item in self._reservations.values()
        ):
            return None
        if len(self._reservations) >= self.max_concurrency:
            return None
        cpu, gpu, memory = self._resource_demand(task)
        used_cpu, used_gpu, used_memory = self._reserved_totals()
        if used_cpu + cpu > self.node_spec.cpu_capacity + 1e-9:
            return None
        if used_gpu + gpu > self.node_spec.gpu_capacity + 1e-9:
            return None
        if used_memory + memory > self.node_spec.memory_gb + 1e-9:
            return None
        reservation = _ResourceReservation(
            reservation_id=f"reservation:{self.node_spec.node_id}:{attempt_id}",
            attempt_id=attempt_id,
            task_id=task.task_id,
            node_id=self.node_spec.node_id,
            scheduled_start_ms=scheduled_start_ms,
            cpu_units=cpu,
            gpu_units=gpu,
            memory_gb=memory,
        )
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def execute(
        self,
        task: TaskInstance,
        assignment: Assignment,
        reservation: _ResourceReservation,
        input_artifacts: tuple[ArtifactRef, ...],
        *,
        seed: int,
        attempt_no: int,
        inject_failure: bool = False,
    ) -> _ExecutionResult:
        if self._reservations.get(reservation.reservation_id) != reservation:
            raise RuntimeError("execution requires an active matching reservation")
        stable_seed = _stable_seed(
            seed,
            self.node_spec.node_id,
            task.workflow_id,
            task.task_id,
            str(attempt_no),
        )
        rng = random.Random(stable_seed)
        compute_ms = max(0.01, assignment.compute_ms * rng.uniform(0.96, 1.04))
        forced_failure = inject_failure or (
            task.task_id in self.fail_first_task_ids and attempt_no == 1
        )
        self.executions.append(
            ExecutionInvocation(
                attempt_id=reservation.attempt_id,
                task_id=task.task_id,
                attempt_no=attempt_no,
                input_artifacts=input_artifacts,
                injected_failure=forced_failure,
            )
        )
        outputs = () if forced_failure else self._build_outputs(task)
        energy_scale = compute_ms / max(1e-9, assignment.compute_ms)
        return _ExecutionResult(
            attempt_id=reservation.attempt_id,
            task_id=task.task_id,
            agent_id=self.node_spec.node_id,
            ok=not forced_failure,
            compute_time_ms=compute_ms,
            energy_j=max(0.0, assignment.energy_j * energy_scale),
            outputs=outputs,
            error_code="injected_first_attempt_failure" if forced_failure else "",
        )

    def release(
        self,
        reservation_id: str,
        finished_time_ms: float,
        *,
        ok: bool,
    ) -> bool:
        reservation = self._reservations.pop(reservation_id, None)
        if reservation is None:
            return False
        self._busy_ms += max(0.0, finished_time_ms - reservation.scheduled_start_ms)
        if ok:
            self._completed_attempts += 1
        else:
            self._failed_attempts += 1
        return True

    def describe(self, makespan_ms: float) -> dict[str, object]:
        available_cpu, available_gpu, available_memory = self._available_resources()
        return {
            "agent_id": self.node_spec.node_id,
            "kind": self.node_spec.kind.value,
            "architecture": self.node_spec.architecture,
            "registered": self._registered,
            "online": self._base_snapshot.online,
            "heartbeat_sequence": self._heartbeat_sequence,
            "last_heartbeat_ms": round(self._last_heartbeat_ms, 4),
            "active_reservations": len(self._reservations),
            "max_concurrency": self.max_concurrency,
            "completed_attempts": self._completed_attempts,
            "failed_attempts": self._failed_attempts,
            "busy_time_ms": round(self._busy_ms, 4),
            "utilization": round(self._busy_ms / max(1.0, makespan_ms), 4),
            "capabilities": list(self.node_spec.capabilities),
            "supported_models": list(self.node_spec.supported_models),
            "resources": {
                "cpu_capacity": self.node_spec.cpu_capacity,
                "gpu_capacity": self.node_spec.gpu_capacity,
                "memory_gb": self.node_spec.memory_gb,
                "available_cpu": round(available_cpu, 4),
                "available_gpu": round(available_gpu, 4),
                "available_memory_gb": round(available_memory, 4),
            },
        }

    def _resource_demand(self, task: TaskInstance) -> tuple[float, float, float]:
        return task_resource_demand(task, self.node_spec)

    def _reserved_totals(self) -> tuple[float, float, float]:
        return (
            sum(item.cpu_units for item in self._reservations.values()),
            sum(item.gpu_units for item in self._reservations.values()),
            sum(item.memory_gb for item in self._reservations.values()),
        )

    def _available_resources(self) -> tuple[float, float, float]:
        cpu, gpu, memory = self._reserved_totals()
        return (
            max(0.0, self.node_spec.cpu_capacity - cpu),
            max(0.0, self.node_spec.gpu_capacity - gpu),
            max(0.0, self.node_spec.memory_gb - memory),
        )

    def _build_outputs(self, task: TaskInstance) -> tuple[ArtifactRef, ...]:
        ports = task.spec.output_ports
        if not ports:
            return (
                ArtifactRef(
                    artifact_id=f"artifact:{task.workflow_id}:{task.task_id}:result",
                    producer_task_id=task.task_id,
                    node_id=self.node_spec.node_id,
                    size_mb=task.spec.output_size_mb,
                    uri=f"agent://{self.node_spec.node_id}/{task.workflow_id}/{task.task_id}/result",
                    producer_port="result",
                ),
            )
        size_per_output = task.spec.output_size_mb / max(1, len(ports))
        return tuple(
            ArtifactRef(
                artifact_id=f"artifact:{task.workflow_id}:{task.task_id}:{port.name}",
                producer_task_id=task.task_id,
                node_id=self.node_spec.node_id,
                size_mb=size_per_output,
                uri=(
                    f"agent://{self.node_spec.node_id}/{task.workflow_id}/"
                    f"{task.task_id}/{port.name}"
                ),
                producer_port=port.name,
                message_type=port.message_type,
            )
            for port in ports
        )


class InProcessRuntime:
    """Deterministic adapter that executes all registered agents in one process."""

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
    ) -> None:
        specs = tuple(node_specs)
        if not specs:
            raise ValueError("at least one runtime node is required")
        node_ids = tuple(spec.node_id for spec in specs)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("runtime node ids must be unique")

        snapshot_items = tuple(snapshots)
        snapshot_by_id = {item.node_id: item for item in snapshot_items}
        if len(snapshot_by_id) != len(snapshot_items):
            raise ValueError("runtime snapshots must have unique node ids")
        unknown_snapshots = set(snapshot_by_id) - set(node_ids)
        if unknown_snapshots:
            raise ValueError(
                f"runtime snapshots reference unknown nodes: {sorted(unknown_snapshots)}"
            )

        concurrency = dict(max_concurrency or {})
        task_types = dict(supported_task_types or {})
        unknown_configuration = (set(concurrency) | set(task_types)) - set(node_ids)
        if unknown_configuration:
            raise ValueError(
                "runtime configuration references unknown nodes: "
                f"{sorted(unknown_configuration)}"
            )

        effective_specs = tuple(
            replace(
                spec,
                max_concurrency=concurrency.get(
                    spec.node_id,
                    spec.max_concurrency,
                ),
            )
            for spec in specs
        )
        failure_ids = tuple(fail_first_task_ids)
        self._node_order = node_ids
        self._agents = {
            spec.node_id: _SimulatedAgent(
                spec,
                snapshot_by_id.get(spec.node_id),
                max_concurrency=spec.max_concurrency,
                supported_task_types=task_types.get(spec.node_id, ()),
                fail_first_task_ids=failure_ids,
            )
            for spec in effective_specs
        }
        self._pending: dict[
            str,
            tuple[_SimulatedAgent, _ResourceReservation, AttemptCompletion],
        ] = {}
        self._dispatch_by_attempt: dict[str, str] = {}
        self._consumed_dispatches: set[str] = set()
        self._terminal_attempts: set[str] = set()
        self._cancelled_attempts: dict[str, str] = {}

    @property
    def executions(self) -> tuple[ExecutionInvocation, ...]:
        """Adapter-local observations used by deterministic contract tests."""

        return tuple(
            invocation
            for node_id in self._node_order
            for invocation in self._agents[node_id].executions
        )

    async def start(self, now_ms: float) -> RuntimeInventory:
        for node_id in self._node_order:
            self._agents[node_id].register(now_ms)
        return await self.inventory(now_ms)

    async def inventory(self, now_ms: float) -> RuntimeInventory:
        heartbeats: list[AgentHeartbeat] = []
        for node_id in self._node_order:
            agent = self._agents[node_id]
            if not agent.registered:
                agent.register(now_ms)
            heartbeats.append(agent.heartbeat(now_ms))
        return RuntimeInventory(
            nodes=tuple(self._agents[node_id].node_spec for node_id in self._node_order),
            heartbeats=tuple(heartbeats),
        )

    async def dispatch(self, command: DispatchCommand) -> DispatchAck:
        task = command.task
        assignment = command.assignment
        node_id = assignment.target_node_id
        rejection = self._validate_command(command)
        agent = self._agents.get(node_id)
        if rejection or agent is None:
            return self._rejected_ack(
                command,
                node_id,
                rejection or "unknown_agent",
            )

        dispatch_id = (
            f"inprocess:{node_id}:{command.attempt_id}:{command.attempt_no}"
        )
        if (
            dispatch_id in self._pending
            or dispatch_id in self._consumed_dispatches
            or command.attempt_id in self._dispatch_by_attempt
            or command.attempt_id in self._terminal_attempts
        ):
            return self._rejected_ack(command, node_id, "duplicate_attempt")

        can_execute, reason = agent.can_execute(task)
        if not can_execute:
            return self._rejected_ack(command, node_id, reason)
        planned_demand = command.resource_reservation.demand
        expected_demand = task_resource_demand(task, agent.node_spec)
        if any(
            abs(planned - expected) > 1e-9
            for planned, expected in zip(
                (
                    planned_demand.cpu_units,
                    planned_demand.gpu_units,
                    planned_demand.memory_gb,
                ),
                expected_demand,
            )
        ):
            return self._rejected_ack(
                command,
                node_id,
                "resource_reservation_mismatch",
            )
        reservation = agent.reserve(
            task,
            command.attempt_id,
            command.resource_reservation.start_ms,
        )
        if reservation is None:
            return self._rejected_ack(
                command,
                node_id,
                "resources_unavailable",
            )

        try:
            execution = agent.execute(
                task,
                assignment,
                reservation,
                command.input_artifacts,
                seed=command.seed,
                attempt_no=command.attempt_no,
                inject_failure=command.inject_failure,
            )
        except Exception:
            agent.release(
                reservation.reservation_id,
                reservation.scheduled_start_ms,
                ok=False,
            )
            raise

        completion = AttemptCompletion(
            dispatch_id=dispatch_id,
            attempt_id=command.attempt_id,
            task_id=task.task_id,
            agent_id=node_id,
            ok=execution.ok,
            finished_time_ms=(
                command.resource_reservation.start_ms
                + execution.compute_time_ms
            ),
            compute_time_ms=execution.compute_time_ms,
            energy_j=execution.energy_j,
            outputs=execution.outputs,
            error_code=execution.error_code,
        )
        self._pending[dispatch_id] = (agent, reservation, completion)
        self._dispatch_by_attempt[command.attempt_id] = dispatch_id
        return DispatchAck(
            dispatch_id=dispatch_id,
            attempt_id=command.attempt_id,
            task_id=task.task_id,
            agent_id=node_id,
            accepted=True,
        )

    async def receive_completion(self, dispatch_id: str) -> AttemptCompletion:
        if dispatch_id in self._consumed_dispatches:
            raise RuntimeError(f"completion already consumed: {dispatch_id}")
        pending = self._pending.pop(dispatch_id, None)
        if pending is None:
            raise KeyError(f"unknown dispatch id: {dispatch_id}")
        agent, reservation, completion = pending
        if not agent.release(
            reservation.reservation_id,
            completion.finished_time_ms,
            ok=completion.ok,
        ):
            raise RuntimeError(
                f"reservation already released: {reservation.reservation_id}"
            )
        agent.heartbeat(completion.finished_time_ms)
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
        pending = self._pending.pop(dispatch_id, None)
        if pending is None:
            return False
        agent, reservation, _ = pending
        released = agent.release(
            reservation.reservation_id,
            now_ms,
            ok=False,
        )
        agent.heartbeat(now_ms)
        self._consumed_dispatches.add(dispatch_id)
        self._terminal_attempts.add(attempt_id)
        self._cancelled_attempts[attempt_id] = reason
        return released

    async def describe(self, makespan_ms: float) -> tuple[dict[str, object], ...]:
        return tuple(
            self._agents[node_id].describe(makespan_ms)
            for node_id in self._node_order
        )

    def _validate_command(self, command: DispatchCommand) -> str:
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


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
