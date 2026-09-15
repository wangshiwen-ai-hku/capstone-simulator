"""A bounded real-execution Agent for offline, trusted-LAN hardware tests."""

from __future__ import annotations

import asyncio
import logging
import math
from time import perf_counter
from uuid import uuid4

import grpc

from interfaces.proto.mars.v1 import (
    artifact_service_pb2_grpc,
    runtime_pb2,
    runtime_service_pb2,
    runtime_service_pb2_grpc,
    workflow_pb2,
)

from .artifacts import ArtifactFiles, ArtifactService, canonical_json, fetch_artifact
from .executor import WorkloadExecutor
from .service import AgentConfig, _node_message, _rejected_ack
from .telemetry import HostTelemetry


LOGGER = logging.getLogger(__name__)
MAX_ATTEMPTS_PER_PROCESS = 1024


class ExecutionAgentService(runtime_service_pb2_grpc.AgentRuntimeServicer):
    """Own admission, cancellation and RPCs; business computation is injected.

    One worker at a time is intentional for this MVP. Exhausted history requires
    an explicit agent restart, rather than silently forgetting deduplication.
    """

    def __init__(
        self,
        config: AgentConfig,
        executor: WorkloadExecutor,
        files: ArtifactFiles,
        peers: dict[str, str],
        *,
        task_timeout_seconds: float = 30.0,
    ) -> None:
        if not math.isfinite(task_timeout_seconds) or task_timeout_seconds <= 0:
            raise ValueError("task timeout must be finite and positive")
        self.config = config
        self.node = _node_message(config)
        self.executor = executor
        self.files = files
        self.peers = dict(peers)
        self.timeout = task_timeout_seconds
        self.telemetry = HostTelemetry(gpu_info=config.node.get("gpu_info"))
        # The executor owns the actual resource requirement for each bundled
        # task. Dispatch cannot relabel a CUDA task as CPU work to bypass GPU
        # admission or its reservation; CPU-only executors default to zero.
        self._gpu_demands = dict(getattr(executor, "gpu_demands", {}))
        if any(
            task_type not in executor.ports or not math.isfinite(demand) or demand < 0
            for task_type, demand in self._gpu_demands.items()
        ):
            raise ValueError("executor GPU demands must be finite and nonnegative")
        self.instance_id = uuid4().hex
        self._sequence = 0
        self._active: dict[str, asyncio.Task] = {}
        self._accepted: dict[str, tuple[bytes, runtime_pb2.DispatchAck]] = {}
        self._history: list[runtime_pb2.AttemptCompletion] = []
        self._subscribers: set[asyncio.Queue] = set()
        self.records: list[dict] = []
        self._closing = False

    async def RegisterAgent(self, request, context):
        if (
            request.agent_id != self.config.agent_id
            or request.schema_version != "mars.runtime.v1"
        ):
            return runtime_pb2.RegisterAgentResponse(
                accepted=False,
                error_code="agent_or_schema_mismatch",
            )
        return runtime_pb2.RegisterAgentResponse(
            accepted=True,
            registration_id=f"hardware:{self.instance_id}",
            node=self.node,
            heartbeat_interval_ms=1000.0,
        )

    async def GetState(self, request, context):
        if request.agent_id != self.config.agent_id:
            return runtime_service_pb2.GetStateResponse(error_code="agent_id_mismatch")
        await self.telemetry.warmup()
        self._sequence += 1
        snapshot = self.telemetry.sample(
            self.config.agent_id,
            self._sequence,
            len(self._active),
        )
        return runtime_service_pb2.GetStateResponse(
            heartbeat=runtime_pb2.AgentHeartbeat(
                agent_id=self.config.agent_id,
                sequence=self._sequence,
                sampled_at_ms=snapshot.sampled_at_ms,
                node_snapshot=snapshot,
                active_reservation_count=len(self._active),
            ),
        )

    def _validate(self, request) -> str:
        task = request.task
        spec = task.spec
        assignment = request.assignment
        if request.schema_version != "mars.runtime.v1":
            return "unsupported_schema"
        if not request.attempt_id or not task.workflow_id or not task.task_id:
            return "missing_identity"
        if assignment.target_node_id != self.config.agent_id:
            return "wrong_target"
        if assignment.task_id != task.task_id:
            return "task_mismatch"
        if not all(
            (
                request.problem_id,
                request.snapshot_id,
                request.policy_id,
                request.policy_version,
                assignment.epoch_id,
                assignment.optimizer_id,
            )
        ):
            return "missing_plan_identity"
        if spec.task_type not in self.executor.ports:
            return "unsupported_task_type"
        if (
            spec.model_requirement
            and spec.model_requirement not in self.node.supported_models
        ):
            return "unsupported_model"
        if request.inject_failure:
            return "mock_failure_injection_not_supported"
        placement = spec.placement_constraints
        if (
            placement.pinned_node_id
            and placement.pinned_node_id != self.config.agent_id
        ):
            return "placement_mismatch"
        if (
            placement.allowed_node_kinds
            and self.node.kind not in placement.allowed_node_kinds
        ):
            return "node_kind_not_allowed"
        if (
            task.source_node_id == self.config.agent_id
            and placement.HasField("allow_source_node")
            and not placement.allow_source_node
        ):
            return "source_node_not_allowed"
        if placement.safety_required:
            return "unsupported_hardware_requirement"
        if not math.isfinite(spec.gpu_demand) or spec.gpu_demand < 0:
            return "invalid_gpu_demand"
        expected_gpu_demand = self._gpu_demands.get(spec.task_type, 0.0)
        if spec.gpu_demand > 0 and (
            not math.isfinite(self.node.gpu_capacity_units)
            or self.node.gpu_capacity_units <= 0
            or "cuda" not in self.node.capabilities
            or expected_gpu_demand <= 0
        ):
            return "unsupported_hardware_requirement"
        if abs(spec.gpu_demand - expected_gpu_demand) > 1e-6:
            return "gpu_demand_mismatch"
        if not set(placement.required_capabilities).issubset(self.node.capabilities):
            return "missing_capability"
        resource = request.resource_reservation
        if (
            not resource.reservation_id
            or resource.task_id != task.task_id
            or resource.node_id != self.config.agent_id
            or resource.epoch_id != assignment.epoch_id
        ):
            return "reservation_identity_mismatch"
        times = (
            assignment.estimated_start_time_ms,
            assignment.estimated_finish_time_ms,
            assignment.compute_time_ms,
            resource.start_time_ms,
            resource.finish_time_ms,
            resource.demand.cpu_units,
            resource.demand.gpu_units,
            resource.demand.memory_gb,
        )
        if any(not math.isfinite(value) or value < 0 for value in times):
            return "invalid_plan_time"
        if abs(resource.demand.gpu_units - spec.gpu_demand) > 1e-6:
            return "gpu_reservation_mismatch"
        if (
            assignment.estimated_finish_time_ms < assignment.estimated_start_time_ms
            or abs(resource.finish_time_ms - assignment.estimated_finish_time_ms) > 1e-6
            or abs(
                resource.finish_time_ms
                - resource.start_time_ms
                - assignment.compute_time_ms
            )
            > 1e-6
            or resource.start_time_ms < assignment.estimated_start_time_ms - 1e-6
        ):
            return "reservation_time_mismatch"
        if (
            resource.demand.cpu_units > self.node.cpu_capacity_units
            or resource.demand.gpu_units > self.node.gpu_capacity_units
            or resource.demand.memory_gb > self.node.memory_capacity_gb
        ):
            return "insufficient_capacity"
        contract = self.executor.ports[spec.task_type]
        for field in ("inputs", "outputs"):
            ports = spec.input_ports if field == "inputs" else spec.output_ports
            actual = {port.name: port.message_type for port in ports}
            if len(actual) != len(ports) or actual != contract[field]:
                return "port_contract_mismatch"
        bindings = request.input_artifact_bindings
        if len(bindings) != len(contract["inputs"]):
            return "missing_input_binding"
        seen: set[str] = set()
        for binding in bindings:
            if (
                binding.consumer_task_id != task.task_id
                or binding.consumer_port in seen
                or contract["inputs"].get(binding.consumer_port)
                != binding.artifact.message_type
            ):
                return "input_binding_mismatch"
            seen.add(binding.consumer_port)
        if self._closing:
            return "agent_stopping"
        if len(self._active) >= 1:
            return "agent_busy"
        if len(self._accepted) >= MAX_ATTEMPTS_PER_PROCESS:
            return "attempt_history_full_restart_agent"
        return ""

    async def DispatchTask(self, request, context):
        encoded = request.SerializeToString(deterministic=True)
        previous = self._accepted.get(request.attempt_id)
        if previous is not None:
            if previous[0] == encoded:
                return previous[1]
            return _rejected_ack(request, self.config.agent_id, "conflicting_attempt")
        error = self._validate(request)
        if error:
            return _rejected_ack(request, self.config.agent_id, error)
        ack = runtime_pb2.DispatchAck(
            dispatch_id=f"hardware:{self.instance_id}:{uuid4().hex}",
            attempt_id=request.attempt_id,
            task_id=request.task.task_id,
            agent_id=self.config.agent_id,
            accepted=True,
        )
        self._accepted[request.attempt_id] = (encoded, ack)
        task = asyncio.create_task(self._execute(ack, request))
        self._active[request.attempt_id] = task
        task.add_done_callback(lambda done: self._ensure_terminal(ack, request, done))
        return ack

    def _ensure_terminal(self, ack, request, task: asyncio.Task) -> None:
        # Cancellation can precede the coroutine's first instruction. Also
        # recover admission slots if telemetry/logging unexpectedly raises.
        if request.attempt_id not in self._active:
            return
        error = "cancelled" if task.cancelled() else "execution_failed"
        detail = "" if task.cancelled() else str(task.exception())[:2000]
        self._publish(
            runtime_pb2.AttemptCompletion(
                dispatch_id=ack.dispatch_id,
                attempt_id=request.attempt_id,
                task_id=request.task.task_id,
                agent_id=self.config.agent_id,
                ok=False,
                started_time_ms=request.assignment.estimated_start_time_ms,
                finished_time_ms=request.assignment.estimated_start_time_ms,
                error_code=error,
                error_message=detail,
            )
        )

    def _publish(self, completion) -> None:
        self._active.pop(completion.attempt_id, None)
        self._history.append(completion)
        for subscriber in self._subscribers:
            subscriber.put_nowait(completion)

    async def StreamCompletions(self, request, context):
        if request.agent_id != self.config.agent_id:
            await context.abort(grpc.StatusCode.NOT_FOUND, "unknown agent")
        if len(self._subscribers) >= 8:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "too many subscribers"
            )
        queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_ATTEMPTS_PER_PROCESS)
        # No await between subscribing and taking the snapshot: no replay gap.
        self._subscribers.add(queue)
        history = tuple(self._history)
        try:
            for result in history:
                yield result
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    async def CancelAttempt(self, request, context):
        task = self._active.get(request.attempt_id)
        if task is None:
            return runtime_pb2.CancelAttemptResponse(
                attempt_id=request.attempt_id,
                cancelled=False,
                error_code="not_active",
            )
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return runtime_pb2.CancelAttemptResponse(
            attempt_id=request.attempt_id, cancelled=True
        )

    async def _execute(self, ack, request) -> None:
        await self.telemetry.warmup()
        started = perf_counter()
        record = {
            "schema": "mars.hil.execution.v1",
            "agent_id": self.config.agent_id,
            "workflow_id": request.task.workflow_id,
            "task_id": request.task.task_id,
            "task_type": request.task.spec.task_type,
            "attempt_id": request.attempt_id,
            "dispatch_id": ack.dispatch_id,
            "execution_mode": (
                "real_cuda" if request.task.spec.gpu_demand > 0 else "real_cpu"
            ),
            "host": self.telemetry.identity(),
            "host_observations": {"before": self.telemetry.observe(), "after": None},
            "input_artifacts": [],
            "remote_input_bytes": 0,
            "input_fetch_ms": 0.0,
            "worker_elapsed_ms": 0.0,
            "energy_j": None,
        }
        outputs: list = []
        error = ""
        detail = ""
        try:
            outputs = await asyncio.wait_for(
                self._invoke(request, record),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            error = "execution_timeout"
        except asyncio.CancelledError:
            error = "cancelled"
        except Exception as exc:
            error = "execution_failed"
            detail = str(exc)[:2000]
        if record["host_observations"]["after"] is None:
            record["host_observations"]["after"] = self.telemetry.observe()
        elapsed = (perf_counter() - started) * 1000
        record.update(
            ok=not error, error_code=error, error_message=detail, elapsed_ms=elapsed
        )
        self.records.append(record)
        LOGGER.info("execution %s", canonical_json(record).decode())
        # Agent clocks never cross hosts. These are logical scheduler timestamps
        # anchored to the dispatch, with *measured* invocation elapsed time.
        # End-to-end physical wall time is measured separately by the HIL runner.
        logical_start = request.assignment.estimated_start_time_ms
        completion = runtime_pb2.AttemptCompletion(
            dispatch_id=ack.dispatch_id,
            attempt_id=request.attempt_id,
            task_id=request.task.task_id,
            agent_id=self.config.agent_id,
            ok=not error,
            started_time_ms=logical_start,
            finished_time_ms=logical_start + elapsed,
            compute_time_ms=record["worker_elapsed_ms"],
            # Energy is unmeasured, not an estimated value presented as real.
            # The JSON evidence explicitly represents it as null.
            energy_j=0.0,
            outputs=outputs if not error else [],
            error_code=error,
            error_message=detail,
        )
        self._publish(completion)

    async def _invoke(self, request, record: dict) -> list:
        inputs = {}
        fetch_started = perf_counter()
        for binding in request.input_artifact_bindings:
            envelope, remote_bytes = await fetch_artifact(
                binding.artifact,
                agent_id=self.config.agent_id,
                files=self.files,
                peers=self.peers,
                timeout_seconds=min(self.timeout, 10.0),
            )
            if envelope.get("workflow_id") != request.task.workflow_id:
                raise ValueError("artifact belongs to a different workflow")
            inputs[binding.consumer_port] = envelope["payload"]
            record["remote_input_bytes"] += remote_bytes
            record["input_artifacts"].append(
                {
                    "port": binding.consumer_port,
                    "sha256": binding.artifact.checksum,
                    "producer_task_id": binding.artifact.producer_task_id,
                    "producer_node_id": binding.artifact.node_id,
                    "remote_bytes": remote_bytes,
                }
            )
        record["input_fetch_ms"] = (perf_counter() - fetch_started) * 1000
        result = await self.executor.execute(
            request.task.spec.task_type, inputs, request.random_seed
        )
        # Capture before serializing any output so every immutable artifact
        # carries the same actual post-worker observation as the local record.
        record["host_observations"]["after"] = self.telemetry.observe()
        expected = self.executor.ports[request.task.spec.task_type]["outputs"]
        if (
            not isinstance(result.outputs, dict)
            or set(result.outputs) != set(expected)
            or any(not isinstance(payload, dict) for payload in result.outputs.values())
            or not math.isfinite(result.elapsed_ms)
            or result.elapsed_ms < 0
        ):
            raise ValueError("invalid executor output contract or elapsed time")
        record["worker_elapsed_ms"] = result.elapsed_ms
        outputs = []
        for port, payload in result.outputs.items():
            message_type = self.executor.ports[request.task.spec.task_type]["outputs"][
                port
            ]
            data = canonical_json(
                {
                    "schema": "mars.hil.artifact.v1",
                    "workflow_id": request.task.workflow_id,
                    "producer_task_id": request.task.task_id,
                    "producer_port": port,
                    "agent_id": self.config.agent_id,
                    "message_type": message_type,
                    "execution": record,
                    "payload": payload,
                }
            )
            digest = self.files.put(data)
            outputs.append(
                workflow_pb2.ArtifactRef(
                    artifact_id=f"sha256:{digest}",
                    producer_task_id=request.task.task_id,
                    node_id=self.config.agent_id,
                    size_mb=len(data) / 1_000_000,
                    uri=f"mars-artifact://{self.config.agent_id}/{digest}",
                    checksum=digest,
                    producer_port=port,
                    message_type=message_type,
                )
            )
        return outputs

    async def close(self) -> None:
        self._closing = True
        active = tuple(self._active.values())
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)


async def start_execution_server(service: ExecutionAgentService):
    await service.telemetry.warmup()
    server = grpc.aio.server(maximum_concurrent_rpcs=32)
    runtime_service_pb2_grpc.add_AgentRuntimeServicer_to_server(service, server)
    artifact_service_pb2_grpc.add_ArtifactStoreServicer_to_server(
        ArtifactService(service.files),
        server,
    )
    port = server.add_insecure_port(service.config.listen)
    if not port:
        raise RuntimeError(f"cannot bind agent endpoint {service.config.listen}")
    await server.start()
    return server, port
