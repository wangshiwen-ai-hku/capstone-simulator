"""CUDA admission integrity without requiring or pretending to execute a GPU."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from agent.artifacts import ArtifactFiles, fetch_artifact
from agent.executor import ExecutionResult
from agent.real_service import ExecutionAgentService
from agent.service import AgentConfig
from agent.telemetry import HostTelemetry, detected_node
from interfaces.proto.mars.v1 import common_pb2, runtime_pb2


GPU_INFO = {
    "available": True,
    "device": "cuda:0",
    "device_count": 2,
    "device_name": "preflight-test-device",
    "compute_capability": [8, 7],
    "torch_version": "test-version",
}


class DeclaredCudaExecutor:
    """Controlled admission fixture; its payload explicitly marks test data."""

    ports = {"test_cuda": {"inputs": {}, "outputs": {"result": "test.result.v1"}}}
    gpu_demands = {"test_cuda": 1.0}

    def __init__(self):
        self.calls = []

    async def execute(self, task_type, inputs, seed):
        self.calls.append((task_type, inputs, seed))
        return ExecutionResult({"result": {"test_fixture": True}}, 1.0)


def _node():
    return detected_node(
        "edge",
        gpu_info=deepcopy(GPU_INFO),
        capabilities=["test_cuda_v1"],
        supported_models=["test/model"],
    )


def _service(tmp_path, *, node=None, executor=None):
    return ExecutionAgentService(
        AgentConfig("edge_1", "127.0.0.1:0", _node() if node is None else node, {}),
        executor or DeclaredCudaExecutor(),
        ArtifactFiles(tmp_path),
        {},
    )


def _request():
    request = runtime_pb2.DispatchCommand(
        schema_version="mars.runtime.v1",
        attempt_id="gpu-attempt-1",
        attempt_number=1,
        problem_id="problem-1",
        snapshot_id="snapshot-1",
        policy_id="policy-1",
        policy_version="1",
    )
    request.task.task_id = "inference"
    request.task.workflow_id = "gpu-workflow"
    request.task.source_node_id = "robot_1"
    spec = request.task.spec
    spec.task_type = "test_cuda"
    spec.gpu_demand = 1.0
    spec.model_requirement = "test/model"
    spec.dominant_resource = common_pb2.RESOURCE_CLASS_GPU
    spec.placement_constraints.required_capabilities.extend(["test_cuda_v1"])
    spec.output_ports.add(name="result", message_type="test.result.v1")
    assignment = request.assignment
    assignment.task_id = "inference"
    assignment.target_node_id = "edge_1"
    assignment.estimated_start_time_ms = 0
    assignment.estimated_finish_time_ms = 100
    assignment.compute_time_ms = 100
    assignment.epoch_id = "epoch-1"
    assignment.optimizer_id = "test-optimizer"
    resource = request.resource_reservation
    resource.reservation_id = "reservation-1"
    resource.epoch_id = "epoch-1"
    resource.task_id = "inference"
    resource.node_id = "edge_1"
    resource.start_time_ms = 0
    resource.finish_time_ms = 100
    resource.demand.cpu_units = 1
    resource.demand.gpu_units = 1
    resource.demand.memory_gb = 0.01
    return request


def test_detected_node_preserves_cpu_default_and_requires_cuda_preflight():
    cpu = detected_node("robot")
    assert cpu["gpu_capacity"] == 0
    assert cpu["capabilities"] == ["cpu", "hil_navigation_v1"]
    assert "gpu_info" not in cpu
    with pytest.raises(ValueError, match="preflight"):
        detected_node("edge", capabilities=["cuda", "test_cuda_v1"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("available", False),
        ("device", "cpu"),
        ("device", "cuda:2"),
        ("device_count", 0),
        ("compute_capability", []),
        ("torch_version", ""),
    ],
)
def test_invalid_cuda_preflight_cannot_advertise_gpu(field, value):
    info = deepcopy(GPU_INFO)
    info[field] = value
    with pytest.raises(ValueError, match="preflight"):
        detected_node("edge", gpu_info=info)


def test_verified_device_advertises_one_worker_without_inventing_measurements():
    node = _node()
    assert node["gpu_capacity"] == 1
    assert node["capabilities"] == ["cpu", "cuda", "test_cuda_v1"]
    assert node["supported_models"] == ["test/model"]
    identity = HostTelemetry(gpu_info=node["gpu_info"]).identity()
    assert identity["cuda_device"]["device"] == "cuda:0"
    assert "gpu_utilization" not in identity["measured"]
    assert {"gpu_utilization", "power", "energy"}.issubset(identity["unavailable"])


@pytest.mark.parametrize("missing", ["capacity", "cuda", "executor_support"])
def test_gpu_admission_requires_capacity_capability_and_executor(tmp_path, missing):
    node = _node()
    executor = DeclaredCudaExecutor()
    if missing == "capacity":
        node["gpu_capacity"] = 0
    elif missing == "cuda":
        node["capabilities"].remove("cuda")
    else:
        executor.gpu_demands = {}
    service = _service(tmp_path, node=node, executor=executor)
    assert service._validate(_request()) == "unsupported_hardware_requirement"
    assert not executor.calls


@pytest.mark.parametrize("demand", [0.0, 0.5, 2.0])
def test_dispatch_gpu_demand_must_match_actual_executor_requirement(tmp_path, demand):
    service = _service(tmp_path)
    request = _request()
    request.task.spec.gpu_demand = demand
    request.resource_reservation.demand.gpu_units = demand
    assert service._validate(request) == "gpu_demand_mismatch"


@pytest.mark.parametrize("demand", [-1.0, float("nan"), float("inf")])
def test_dispatch_gpu_demand_must_be_finite_and_nonnegative(tmp_path, demand):
    request = _request()
    request.task.spec.gpu_demand = demand
    assert _service(tmp_path)._validate(request) == "invalid_gpu_demand"


@pytest.mark.parametrize("reservation", [0.0, 0.5, 2.0])
def test_reservation_must_match_gpu_task_demand(tmp_path, reservation):
    request = _request()
    request.resource_reservation.demand.gpu_units = reservation
    assert _service(tmp_path)._validate(request) == "gpu_reservation_mismatch"


def test_matched_reservation_still_requires_sufficient_capacity(tmp_path):
    node = _node()
    node["gpu_capacity"] = 0.5
    assert (
        _service(tmp_path, node=node)._validate(_request()) == "insufficient_capacity"
    )


def test_cuda_support_does_not_admit_safety_required_work(tmp_path):
    request = _request()
    request.task.spec.placement_constraints.safety_required = True
    assert _service(tmp_path)._validate(request) == "unsupported_hardware_requirement"


def test_gpu_dispatch_propagates_cuda_identity_without_claiming_measured_energy(
    tmp_path,
):
    async def run():
        service = _service(tmp_path)
        try:
            ack = await service.DispatchTask(_request(), None)
            assert ack.accepted
            await asyncio.wait_for(
                asyncio.gather(*tuple(service._active.values())), timeout=2
            )
            completion = service._history[-1]
            assert completion.ok
            assert len(service.executor.calls) == 1
            assert service.records[0]["execution_mode"] == "real_cuda"
            assert service.records[0]["energy_j"] is None
            envelope, _ = await fetch_artifact(
                completion.outputs[0],
                agent_id="edge_1",
                files=service.files,
                peers={},
            )
            assert envelope["payload"] == {"test_fixture": True}
            assert envelope["execution"]["host"]["cuda_device"] == GPU_INFO
            assert "gpu_utilization" in envelope["execution"]["host"]["unavailable"]
        finally:
            await service.close()

    asyncio.run(run())
