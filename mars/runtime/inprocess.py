"""In-process implementation of the MARS runtime port."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from ..domain.artifact import (
    ArtifactRef,
    InputArtifactBinding,
    artifacts_from_bindings,
)
from ..domain.execution import Assignment, task_resource_demand
from ..domain.task import TaskInstance, resolved_placement_constraints
from ..domain.topology import (
    NodeKind,
    NodeSnapshot,
    NodeSpec,
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
    scheduled_finish_ms: float
    cpu_units: float
    gpu_units: float
    memory_gb: float


@dataclass(frozen=True)
class ExecutionInvocation:
    attempt_id: str
    task_id: str
    attempt_no: int
    input_artifact_bindings: tuple[InputArtifactBinding, ...]
    injected_failure: bool

    @property
    def input_artifacts(self) -> tuple[ArtifactRef, ...]:
        return artifacts_from_bindings(self.input_artifact_bindings)


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
    """One simulated Orin or edge executor with explicit capacity accounting.

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
        execution_noise: float = 0.04,
        respect_expected_accuracy: bool = False,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        if not 0.0 <= execution_noise <= 1.0:
            raise ValueError("execution_noise must be in [0, 1]")
        if snapshot is not None and snapshot.node_id != node_spec.node_id:
            raise ValueError("agent snapshot must match node_spec.node_id")
        self._node_spec = node_spec
        self._base_snapshot = snapshot or NodeSnapshot(node_spec.node_id)
        self.max_concurrency = max_concurrency
        self.supported_task_types = frozenset(supported_task_types)
        self.fail_first_task_ids = frozenset(fail_first_task_ids)
        self.execution_noise = execution_noise
        self.respect_expected_accuracy = respect_expected_accuracy
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
        """Return the dynamic snapshot at the last heartbeat time."""

        return self.snapshot_at(self._last_heartbeat_ms)

    def snapshot_at(self, now_ms: float) -> NodeSnapshot:
        cpu, gpu, memory = self._reserved_totals(now_ms)
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
        return len(self._active_reservations(self._last_heartbeat_ms))

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
            snapshot=self.snapshot_at(now_ms),
            active_reservations=len(self._active_reservations(now_ms)),
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
        scheduled_finish_ms: float,
    ) -> _ResourceReservation | None:
        feasible, _ = self.can_execute(task)
        if not feasible or any(
            item.attempt_id == attempt_id for item in self._reservations.values()
        ):
            return None
        if scheduled_finish_ms < scheduled_start_ms:
            return None
        cpu, gpu, memory = self._resource_demand(task)
        duration_ms = scheduled_finish_ms - scheduled_start_ms
        actual_start_ms = self._earliest_feasible_start(
            scheduled_start_ms,
            duration_ms,
            cpu,
            gpu,
            memory,
        )
        if actual_start_ms is None:
            return None
        reservation = _ResourceReservation(
            reservation_id=f"reservation:{self.node_spec.node_id}:{attempt_id}",
            attempt_id=attempt_id,
            task_id=task.task_id,
            node_id=self.node_spec.node_id,
            scheduled_start_ms=actual_start_ms,
            scheduled_finish_ms=actual_start_ms + duration_ms,
            cpu_units=cpu,
            gpu_units=gpu,
            memory_gb=memory,
        )
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def reschedule(
        self,
        reservation_id: str,
        *,
        earliest_start_ms: float,
        duration_ms: float,
    ) -> _ResourceReservation:
        """Resize a planned interval to the sampled execution duration."""

        reservation = self._reservations.pop(
            reservation_id,
            None,
        )
        if reservation is None:
            raise RuntimeError(
                "cannot reschedule an inactive reservation"
            )
        actual_start_ms = self._earliest_feasible_start(
            earliest_start_ms,
            duration_ms,
            reservation.cpu_units,
            reservation.gpu_units,
            reservation.memory_gb,
        )
        if actual_start_ms is None:
            self._reservations[reservation_id] = reservation
            raise RuntimeError(
                "sampled execution no longer fits node capacity"
            )
        updated = replace(
            reservation,
            scheduled_start_ms=actual_start_ms,
            scheduled_finish_ms=actual_start_ms + duration_ms,
        )
        self._reservations[reservation_id] = updated
        return updated

    def execute(
        self,
        task: TaskInstance,
        assignment: Assignment,
        reservation: _ResourceReservation,
        input_artifact_bindings: tuple[InputArtifactBinding, ...],
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
        compute_ms = max(
            0.01,
            assignment.compute_ms
            * rng.uniform(
                1.0 - self.execution_noise,
                1.0 + self.execution_noise,
            ),
        )
        injected_failure = inject_failure or (
            task.task_id in self.fail_first_task_ids and attempt_no == 1
        )
        accuracy_failure = False
        if self.respect_expected_accuracy:
            thermal_penalty = (
                0.2 if self._base_snapshot.temperature_c > 93.0 else 0.0
            )
            network_penalty = (
                0.04
                if assignment.communication_ms
                > task.spec.latency_budget_ms * 0.5
                else 0.0
            )
            deadline_penalty = (
                0.25
                if reservation.scheduled_start_ms + compute_ms
                > task.deadline_time_ms
                else 0.0
            )
            success_probability = max(
                0.05,
                task.expected_accuracy
                - thermal_penalty
                - network_penalty
                - deadline_penalty,
            )
            accuracy_failure = rng.random() >= success_probability
        forced_failure = injected_failure or accuracy_failure
        self.executions.append(
            ExecutionInvocation(
                attempt_id=reservation.attempt_id,
                task_id=task.task_id,
                attempt_no=attempt_no,
                input_artifact_bindings=input_artifact_bindings,
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
            error_code=(
                "injected_first_attempt_failure"
                if injected_failure
                else "execution_failed"
                if accuracy_failure
                else ""
            ),
        )

    def release(
        self,
        reservation_id: str,
        finished_time_ms: float,
        *,
        ok: bool,
        retain_until_finish: bool = False,
    ) -> bool:
        reservation = self._reservations.get(reservation_id)
        if reservation is None:
            return False
        if not retain_until_finish:
            self._reservations.pop(reservation_id, None)
        self._busy_ms += max(0.0, finished_time_ms - reservation.scheduled_start_ms)
        if ok:
            self._completed_attempts += 1
        else:
            self._failed_attempts += 1
        return True

    def _discard_expired_reservations(self, now_ms: float) -> None:
        expired_ids = tuple(
            reservation_id
            for reservation_id, reservation in self._reservations.items()
            if reservation.scheduled_finish_ms <= now_ms + 1e-9
        )
        for reservation_id in expired_ids:
            self._reservations.pop(reservation_id, None)

    def _discard_attempt_reservation(self, attempt_id: str) -> None:
        for reservation_id, reservation in tuple(
            self._reservations.items()
        ):
            if reservation.attempt_id == attempt_id:
                self._reservations.pop(reservation_id, None)

    def describe(self, makespan_ms: float) -> dict[str, object]:
        self._discard_expired_reservations(makespan_ms)
        active = self._active_reservations(makespan_ms)
        available_cpu, available_gpu, available_memory = (
            self._available_resources(makespan_ms)
        )
        capacity_time_ms = max(0.0, makespan_ms) * self.max_concurrency
        utilization = (
            self._busy_ms / capacity_time_ms
            if capacity_time_ms > 0.0
            else 0.0
        )
        return {
            "agent_id": self.node_spec.node_id,
            "kind": self.node_spec.kind.value,
            "architecture": self.node_spec.architecture,
            "registered": self._registered,
            "online": self._base_snapshot.online,
            "heartbeat_sequence": self._heartbeat_sequence,
            "last_heartbeat_ms": round(self._last_heartbeat_ms, 4),
            "active_reservations": len(active),
            "max_concurrency": self.max_concurrency,
            "completed_attempts": self._completed_attempts,
            "failed_attempts": self._failed_attempts,
            "busy_time_ms": round(self._busy_ms, 4),
            "utilization": round(min(1.0, max(0.0, utilization)), 4),
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

    def _active_reservations(
        self,
        now_ms: float,
    ) -> tuple[_ResourceReservation, ...]:
        return tuple(
            item
            for item in self._reservations.values()
            if (
                item.scheduled_start_ms
                <= now_ms
                < item.scheduled_finish_ms
            )
        )

    def _overlapping_reservations(
        self,
        start_ms: float,
        finish_ms: float,
    ) -> tuple[_ResourceReservation, ...]:
        if finish_ms == start_ms:
            return self._active_reservations(start_ms)
        return tuple(
            item
            for item in self._reservations.values()
            if (
                item.scheduled_start_ms < finish_ms
                and start_ms < item.scheduled_finish_ms
            )
        )

    def _fits_interval(
        self,
        start_ms: float,
        finish_ms: float,
        cpu_units: float,
        gpu_units: float,
        memory_gb: float,
    ) -> bool:
        overlapping = self._overlapping_reservations(
            start_ms,
            finish_ms,
        )
        check_points = {start_ms}
        check_points.update(
            item.scheduled_start_ms
            for item in overlapping
            if start_ms <= item.scheduled_start_ms < finish_ms
        )
        for point_ms in check_points:
            active = tuple(
                item
                for item in overlapping
                if (
                    item.scheduled_start_ms
                    <= point_ms
                    < item.scheduled_finish_ms
                )
            )
            if len(active) + 1 > self.max_concurrency:
                return False
            used_cpu = sum(item.cpu_units for item in active)
            used_gpu = sum(item.gpu_units for item in active)
            used_memory = sum(item.memory_gb for item in active)
            if used_cpu + cpu_units > self.node_spec.cpu_capacity + 1e-9:
                return False
            if used_gpu + gpu_units > self.node_spec.gpu_capacity + 1e-9:
                return False
            if used_memory + memory_gb > self.node_spec.memory_gb + 1e-9:
                return False
        return True

    def _earliest_feasible_start(
        self,
        earliest_start_ms: float,
        duration_ms: float,
        cpu_units: float,
        gpu_units: float,
        memory_gb: float,
    ) -> float | None:
        if (
            duration_ms < 0
            or cpu_units > self.node_spec.cpu_capacity + 1e-9
            or gpu_units > self.node_spec.gpu_capacity + 1e-9
            or memory_gb > self.node_spec.memory_gb + 1e-9
        ):
            return None
        cursor_ms = earliest_start_ms
        while not self._fits_interval(
            cursor_ms,
            cursor_ms + duration_ms,
            cpu_units,
            gpu_units,
            memory_gb,
        ):
            overlaps = self._overlapping_reservations(
                cursor_ms,
                cursor_ms + duration_ms,
            )
            releases = [
                item.scheduled_finish_ms
                for item in overlaps
                if item.scheduled_finish_ms > cursor_ms + 1e-9
            ]
            if not releases:
                return None
            cursor_ms = min(releases)
        return cursor_ms

    def _reserved_totals(
        self,
        now_ms: float,
    ) -> tuple[float, float, float]:
        active = self._active_reservations(now_ms)
        return (
            sum(item.cpu_units for item in active),
            sum(item.gpu_units for item in active),
            sum(item.memory_gb for item in active),
        )

    def _available_resources(
        self,
        now_ms: float,
    ) -> tuple[float, float, float]:
        cpu, gpu, memory = self._reserved_totals(now_ms)
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
        execution_noise: float = 0.04,
        respect_expected_accuracy: bool = False,
    ) -> None:
        if not 0.0 <= execution_noise <= 1.0:
            raise ValueError("execution_noise must be in [0, 1]")
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
                execution_noise=execution_noise,
                respect_expected_accuracy=respect_expected_accuracy,
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
            agent._discard_expired_reservations(now_ms)
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
            command.resource_reservation.finish_ms,
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
                command.input_artifact_bindings,
                seed=command.seed,
                attempt_no=command.attempt_no,
                inject_failure=command.inject_failure,
            )
            reservation = agent.reschedule(
                reservation.reservation_id,
                earliest_start_ms=(
                    command.resource_reservation.start_ms
                ),
                duration_ms=execution.compute_time_ms,
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
            started_time_ms=reservation.scheduled_start_ms,
            finished_time_ms=reservation.scheduled_finish_ms,
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
            retain_until_finish=True,
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
            for agent in self._agents.values():
                agent._discard_attempt_reservation(attempt_id)
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
