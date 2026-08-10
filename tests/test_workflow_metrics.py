from mars.domain.task import TaskClass, TaskInstance, TaskSpec
from mars.domain.topology import NodeKind, NodeSnapshot, NodeSpec
from mars.domain.workflow import WorkflowSpec
from mars.workflow_metrics import evaluate_workflow_metrics


def test_workflow_metrics_add_initial_background_and_task_demand() -> None:
    node = NodeSpec("robot", NodeKind.ROBOT, 1, 1, 1, 100, 0)
    task = TaskInstance(
        "task",
        "workflow",
        "task",
        "robot",
        TaskSpec(
            "custom",
            TaskClass.REALTIME_OFFLOADABLE,
            compute_demand=1,
            gpu_demand=0.2,
            latency_budget_ms=100,
        ),
        priority=5,
    )
    metrics = evaluate_workflow_metrics(
        (
            {
                "task_id": "task",
                "target_node_id": "robot",
                "attempts": [
                    {
                        "target_node_id": "robot",
                        "start_time_ms": 0,
                        "finish_time_ms": 50,
                        "compute_time_ms": 50,
                        "communication_time_ms": 10,
                    }
                ],
            },
        ),
        WorkflowSpec("workflow", (task,)),
        (node,),
        (
            NodeSnapshot(
                "robot",
                cpu_util=0.1,
                gpu_util=0.2,
                memory_util=0.3,
            ),
        ),
        None,
    )

    assert metrics["expected_success_ratio"] == 1
    assert metrics["normalized_communication"] == 0.1
    assert metrics["peak_cpu_utilization"] == 0.25
    assert metrics["peak_gpu_utilization"] == 0.4
    assert metrics["peak_memory_utilization"] == 0.38
    assert metrics["maximum_resource_utilization"] == 0.4
    assert metrics["workflow_evaluation_objective"] == -0.1
