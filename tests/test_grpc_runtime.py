from __future__ import annotations

import asyncio
from dataclasses import replace

from agent.service import load_agent_configs, start_agent_server
from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_workflow,
)
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import GenerateSceneRequest
from mars.coordinator import CentralCoordinator
from mars.runtime import GrpcRuntimeAdapter, RuntimePort


def test_hardware_agent_example_uses_lan_bindings() -> None:
    configured = load_agent_configs("configs/mars/agents.hardware.example.json")

    assert {item.agent_id for item in configured} == {
        "robot_1",
        "robot_2",
        "edge_pc",
    }
    assert {item.listen for item in configured} == {"0.0.0.0:50051"}


def test_grpc_runtime_completes_three_agent_workflow() -> None:
    async def run() -> None:
        configured = load_agent_configs("configs/mars/agents.local.json")
        servers = []
        endpoints: dict[str, str] = {}
        for item in configured:
            server, port = await start_agent_server(
                replace(item, listen="127.0.0.1:0")
            )
            servers.append(server)
            endpoints[item.agent_id] = f"127.0.0.1:{port}"
        try:
            scene = build_deterministic_scene(
                GenerateSceneRequest(
                    robot_count=2,
                    edge_count=1,
                    use_llm=False,
                    seed=19,
                )
            )
            workflow = build_workflow(scene)
            failed_task_id = next(
                task.task_id
                for task in workflow.tasks
                if task.spec.task_type == "object_detection"
            )
            runtime = GrpcRuntimeAdapter(endpoints)
            assert isinstance(runtime, RuntimePort)
            report = await CentralCoordinator(
                runtime,
                link_specs=build_link_specs(scene),
                link_snapshots=build_link_snapshots(scene),
            ).run_async(
                workflow,
                seed=19,
                max_attempts=2,
                fail_first_task_ids=(failed_task_id,),
            )

            assert report.workflow["state"] == "succeeded"
            assert len(report.agents) == 3
            assert report.metrics["retry_count"] == 1
            retried = next(
                item
                for item in report.task_results
                if item["task_id"] == failed_task_id
            )
            assert [item["state"] for item in retried["attempts"]] == [
                "failed",
                "succeeded",
            ]
        finally:
            await asyncio.gather(*(server.stop(0) for server in servers))

    asyncio.run(run())
