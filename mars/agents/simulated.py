"""Deterministic local agent sessions for end-to-end scheduler exercises."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, replace
from typing import Protocol, runtime_checkable

from ..models import (
    ArtifactRef,
    Assignment,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    TaskClass,
    TaskInstance,
)


@dataclass(frozen=True)
class AgentHeartbeat:
    agent_id: str
    sequence: int
    sampled_at_ms: float
    snapshot: NodeSnapshot
    active_reservations: int


@dataclass(frozen=True)
class ResourceReservation:
    reservation_id: str
    attempt_id: str
    task_id: str
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
class AgentExecutionResult:
    attempt_id: str
    task_id: str
    agent_id: str
    ok: bool
    compute_time_ms: float
    energy_j: float
    outputs: tuple[ArtifactRef, ...]
    error_code: str = ""


@runtime_checkable
class AgentSession(Protocol):
    """Coordinator-facing contract implemented by local or future sessions."""

    @property
    def node_spec(self) -> NodeSpec: ...

    @property
    def snapshot(self) -> NodeSnapshot: ...

    def reset(self) -> None: ...

    def register(self, now_ms: float) -> bool: ...

    def heartbeat(self, now_ms: float) -> AgentHeartbeat: ...

    def can_execute(self, task: TaskInstance) -> tuple[bool, str]: ...

    def reserve(
        self,
        task: TaskInstance,
        attempt_id: str,
        scheduled_start_ms: float,
    ) -> ResourceReservation | None: ...

    def execute(
        self,
        task: TaskInstance,
        assignment: Assignment,
        reservation: ResourceReservation,
        input_artifacts: tuple[ArtifactRef, ...],
        *,
        seed: int,
        attempt_no: int,
        inject_failure: bool = False,
    ) -> AgentExecutionResult: ...

    def release(
        self,
        reservation_id: str,
        finished_time_ms: float,
        *,
        ok: bool,
    ) -> bool: ...

    def describe(self, makespan_ms: float) -> dict[str, object]: ...


class SimulatedAgent:
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
        self._reservations: dict[str, ResourceReservation] = {}
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

    def reset(self) -> None:
        self.executions.clear()
        self._registered = False
        self._heartbeat_sequence = 0
        self._last_heartbeat_ms = 0.0
        self._reservations.clear()
        self._busy_ms = 0.0
        self._completed_attempts = 0
        self._failed_attempts = 0

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
        if task.spec.task_class is TaskClass.LOCAL_SAFETY and (
            self.node_spec.node_id != task.source_node_id
            or self.node_spec.kind is not NodeKind.ROBOT
            or not self.node_spec.safety_capable
        ):
            return False, "local_safety_requires_source_robot"
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
    ) -> ResourceReservation | None:
        feasible, _ = self.can_execute(task)
        if not feasible or attempt_id in self._reservations:
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
        reservation = ResourceReservation(
            reservation_id=f"reservation:{self.node_spec.node_id}:{attempt_id}",
            attempt_id=attempt_id,
            task_id=task.task_id,
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
        reservation: ResourceReservation,
        input_artifacts: tuple[ArtifactRef, ...],
        *,
        seed: int,
        attempt_no: int,
        inject_failure: bool = False,
    ) -> AgentExecutionResult:
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
        return AgentExecutionResult(
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
        # compute_demand is a relative duration/complexity signal, not a
        # literal number of CPU cores. Reservations scale it into the target's
        # declared capacity while preserving pressure differences.
        cpu = min(
            self.node_spec.cpu_capacity,
            max(0.05, task.spec.compute_demand * 0.15),
        )
        gpu = max(0.0, task.spec.gpu_demand)
        memory = max(0.05, min(16.0, task.spec.compute_demand * 0.08))
        return cpu, gpu, memory

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


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
