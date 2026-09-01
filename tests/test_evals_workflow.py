from evals import (
    AggregationRule,
    EvaluationResult,
    MetricDefinition,
    MetricObservation,
    aggregate_evaluations,
    evaluate_run_artifact,
)
from mars.coordinator import CoordinatorReport
from mars.domain.task import TaskClass, TaskInstance, TaskSpec
from mars.domain.topology import NodeKind, NodeSnapshot, NodeSpec
from mars.domain.workflow import WorkflowSpec
from mars.run_artifact import build_run_artifact


def test_workflow_evaluation_uses_run_artifact_facts() -> None:
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
    report = CoordinatorReport(
        workflow={
            "workflow_id": "workflow",
            "state": "succeeded",
            "levels": {"task": 0},
        },
        metrics={
            "transferred_mb": 0.0,
            "critical_path_ms": 50.0,
        },
        task_results=(
            {
                "task_id": "task",
                "state": "succeeded",
                "target_node_id": "robot",
                "mode": "local",
                "attempts": [
                    {
                        "target_node_id": "robot",
                        "start_time_ms": 0,
                        "finish_time_ms": 50.004,
                        "compute_time_ms": 50,
                        "communication_time_ms": 10,
                        "energy_j": 1.234,
                    }
                ],
                "outputs": [],
            },
        ),
        agents=(),
        data_edges=(),
        events=(),
        logs=(),
    )
    artifact = build_run_artifact(
        run_id="run",
        workflow=WorkflowSpec("workflow", (task,)),
        node_specs=(node,),
        node_snapshots=(
            NodeSnapshot(
                "robot",
                cpu_util=0.1,
                gpu_util=0.2,
                memory_util=0.3,
            ),
        ),
        link_specs=(),
        link_snapshots=(),
        profiles=(),
        raw_report=report,
        algorithm="heuristic",
        formulation=None,
        seed=7,
        deterministic=True,
        max_attempts=1,
        network_jitter=0,
        resource_noise=0,
    )

    metrics = evaluate_run_artifact(artifact).as_dict()

    assert metrics["expected_success_ratio"] == 1
    assert metrics["normalized_communication"] == 0.1
    assert metrics["peak_cpu_utilization"] == 1.1
    assert metrics["peak_gpu_utilization"] == 0.4
    assert metrics["peak_memory_utilization"] == 0.38
    assert metrics["maximum_resource_utilization"] == 1.1
    assert metrics["workflow_evaluation_objective"] == 1.3
    assert metrics["avg_latency_ms"] == 50.0
    assert metrics["makespan_ms"] == 50.0
    assert metrics["avg_energy_j"] == 1.23
    assert metrics["total_energy_j"] == 1.23


def test_ratio_metrics_aggregate_as_ratio_of_sums() -> None:
    definition = MetricDefinition(
        "success_rate",
        "ratio",
        AggregationRule.RATIO_OF_SUMS,
    )
    results = (
        EvaluationResult((MetricObservation(definition, 1.0, 1, 1),)),
        EvaluationResult((MetricObservation(definition, 0.5, 50, 100),)),
    )

    aggregated = aggregate_evaluations(results)

    assert aggregated["success_rate"] == 51 / 101
