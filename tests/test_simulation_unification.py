from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from backend.app import simulation as simulation_api
from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
)
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import (
    GenerateSceneRequest,
    SimulateRequest,
    SimulationResponse,
)
from mars.coordinator import CentralCoordinator, CoordinatorReport
from mars.runtime import InProcessRuntime, RuntimePort


class _RecordingRuntime(InProcessRuntime):
    """Record the RuntimePort operations used by a Web simulation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []

    async def start(self, now_ms: float):
        self.calls.append("start")
        return await super().start(now_ms)

    async def inventory(self, now_ms: float):
        self.calls.append("inventory")
        return await super().inventory(now_ms)

    async def dispatch(self, command):
        self.calls.append("dispatch")
        return await super().dispatch(command)

    async def receive_completion(self, dispatch_id: str):
        self.calls.append("receive_completion")
        return await super().receive_completion(dispatch_id)

    async def cancel(
        self,
        attempt_id: str,
        reason: str,
        now_ms: float,
    ):
        self.calls.append("cancel")
        return await super().cancel(attempt_id, reason, now_ms)

    async def describe(self, makespan_ms: float):
        self.calls.append("describe")
        return await super().describe(makespan_ms)


class _RecordingCoordinator(CentralCoordinator):
    def __init__(
        self,
        *args: Any,
        report_sink: Callable[[CoordinatorReport], None],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._report_sink = report_sink

    def run(self, *args: Any, **kwargs: Any) -> CoordinatorReport:
        report = super().run(*args, **kwargs)
        self._report_sink(report)
        return report


def test_web_simulation_uses_canonical_coordinator_runtime_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            use_llm=False,
            seed=37,
        )
    )
    captured: dict[str, object] = {}

    def recording_factory(
        scene_arg,
        *,
        execution_noise: float = 0.04,
        respect_expected_accuracy: bool = False,
        link_snapshots=None,
    ) -> CentralCoordinator:
        runtime = _RecordingRuntime(
            build_node_specs(scene_arg),
            build_node_snapshots(scene_arg),
            execution_noise=execution_noise,
            respect_expected_accuracy=respect_expected_accuracy,
        )
        captured["runtime"] = runtime
        effective_links = (
            tuple(build_link_snapshots(scene_arg))
            if link_snapshots is None
            else tuple(link_snapshots)
        )
        return _RecordingCoordinator(
            runtime,
            link_specs=build_link_specs(scene_arg),
            link_snapshots=effective_links,
            report_sink=lambda report: captured.__setitem__(
                "report",
                report,
            ),
        )

    monkeypatch.setattr(
        simulation_api,
        "coordinator_for_scene",
        recording_factory,
    )

    response = simulation_api.run_simulation(
        SimulateRequest(
            scene=scene,
            algorithm="dag_deadline",
            network_jitter=0.0,
            resource_noise=0.0,
            seed=37,
        )
    )

    assert isinstance(response, SimulationResponse)
    runtime = captured["runtime"]
    assert isinstance(runtime, RuntimePort)
    assert isinstance(runtime, _RecordingRuntime)
    assert runtime.calls[0:2] == ["start", "inventory"]
    assert "dispatch" in runtime.calls
    assert "receive_completion" in runtime.calls
    assert runtime.calls[-1] == "describe"

    report = captured["report"]
    assert isinstance(report, CoordinatorReport)
    assert {
        key: response.transport[key]
        for key in (
            "active",
            "execution_path",
            "runtime_adapter",
            "network_jitter",
            "resource_noise",
            "runtime_event_count",
        )
    } == {
        "active": "in_process_runtime",
        "execution_path": "central_coordinator_runtime_port",
        "runtime_adapter": "InProcessRuntime",
        "network_jitter": 0.0,
        "resource_noise": 0.0,
        "runtime_event_count": len(report.events),
    }
    assert response.metrics.task_count == len(scene.tasks)
    assert response.workflow.workflow_id == scene.workflow_id

    canonical_by_task = {
        str(item["task_id"]): item for item in report.task_results
    }
    projected_by_task = {
        item.task_id: item for item in response.task_results
    }
    assert projected_by_task.keys() == canonical_by_task.keys()
    for task_id, projected in projected_by_task.items():
        canonical = canonical_by_task[task_id]
        assert projected.state == canonical["state"]
        assert projected.target_node_id == str(
            canonical["target_node_id"]
        )
        assert projected.mode == canonical["mode"]

    assert response.workflow.state == report.workflow["state"]
    assert response.dag.topological_order == list(
        report.workflow["topological_order"]
    )
