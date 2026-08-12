"""Demonstrate Simulate vs Runtime workflow failure-sampling mismatch.

Simulate (`backend/app/simulation.py`) builds the coordinator with
`respect_expected_accuracy=True`, so InProcessRuntime samples task failure
from profile `failure_rate` via Assignment.success_probability.

Runtime submit (`backend/app/runtime.py`) defaults to
`respect_expected_accuracy=False`, so profile failures are not sampled unless
injected (e.g. inject_first_failure).

This script builds a deterministic scene with `local_llm_10b` tasks and an
optional amplified failure_rate on profiles so the gap is obvious in one run.

Usage (from repo root):

    PYTHONPATH=. python scripts/demo_simulate_vs_runtime_failure_sampling.py

Optional LLM scene (uses backend/.env — e.g. APIYI via CUSTOM or OPENAI vars):

    PYTHONPATH=. python scripts/demo_simulate_vs_runtime_failure_sampling.py --use-llm
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_workflow,
)
from backend.app.runtime import coordinator_for_scene, runtime_for_scene
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import GenerateSceneRequest
from backend.app.scheduling import configure_scheduling
from mars.coordinator import CentralCoordinator
from mars.profiling import ProfileCatalog, profile_catalog_from_workloads
from mars.synthetic_workloads import load_default_synthetic_workloads
from mars.workflow_metrics import evaluate_workflow_metrics
from backend.app.mars_adapter import build_node_specs, build_node_snapshots


def _high_failure_catalog(amplified_rate: float) -> ProfileCatalog:
    workloads = load_default_synthetic_workloads()
    base = profile_catalog_from_workloads(workloads)
    profiles = [
        replace(profile, failure_rate=amplified_rate)
        if profile.task_type == "local_llm_10b"
        else profile
        for profile in base.profiles
    ]
    return ProfileCatalog(list(profiles))


def _task_outcomes(task_results: list[dict[str, object]]) -> dict[str, int]:
    failed_attempts = 0
    succeeded_attempts = 0
    task_failures = 0
    for row in task_results:
        attempts = row.get("attempts", [])
        if not isinstance(attempts, list):
            continue
        task_failed = False
        for attempt in attempts:
            state = str(attempt.get("state", ""))
            if state == "failed":
                failed_attempts += 1
                task_failed = True
            elif state == "succeeded":
                succeeded_attempts += 1
        if task_failed:
            task_failures += 1
    return {
        "failed_attempts": failed_attempts,
        "succeeded_attempts": succeeded_attempts,
        "tasks_with_failed_attempt": task_failures,
    }


def _print_block(
    title: str,
    workflow_state: str,
    metrics: dict[str, object],
    task_results: list[dict[str, object]],
) -> None:
    outcomes = _task_outcomes(task_results)
    print(f"\n{'=' * 60}")
    print(title)
    print(f"  workflow_state: {workflow_state}")
    print(f"  retry_count: {metrics.get('retry_count')}")
    print(f"  attempt_count: {metrics.get('attempt_count')}")
    print(
        f"  failed/succeeded attempts: "
        f"{outcomes['failed_attempts']}/{outcomes['succeeded_attempts']}"
    )
    print(f"  tasks with ≥1 failed attempt: {outcomes['tasks_with_failed_attempt']}")
    print(
        "  expected_success_ratio (from profile at placement, not realized): "
        f"{metrics.get('expected_success_ratio')}"
    )
    print(f"  workflow_evaluation_objective: {metrics.get('workflow_evaluation_objective')}")


def _coordinator_report_metrics(
    report,
    scene,
    workflow,
    coordinator: CentralCoordinator,
    scheduling_weights,
) -> dict[str, object]:
    extra = evaluate_workflow_metrics(
        report.task_results,
        workflow,
        build_node_specs(scene),
        build_node_snapshots(scene),
        coordinator.profile_catalog,
        weights=scheduling_weights,
    )
    return {**report.metrics, **extra}


def run_demo(seed: int, amplified_rate: float, use_llm: bool) -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=1,
            edge_count=2,
            use_llm=use_llm,
            seed=99,
            task_categories=["local_llm_10b"],
        )
    )
    workflow = build_workflow(scene)
    scheduling = configure_scheduling("binary_offload", {})
    profile_catalog = _high_failure_catalog(amplified_rate)
    links = build_link_specs(scene)
    link_snaps = build_link_snapshots(scene)

    print("Demo scene")
    print(f"  tasks: {len(scene.tasks)} × local_llm_10b")
    print(f"  robots: 1, edges: 2, seed: {seed}")
    print(f"  profile failure_rate (local_llm_10b, all targets): {amplified_rate}")
    print(f"  algorithm: binary_offload, max_attempts: 3 (both paths below)")
    print(
        "  note: POST /api/simulate still uses max_attempts=1 inside simulation.py"
    )

    # --- Path A: same as Simulate API (respect_expected_accuracy=True) ---
    simulate_style = CentralCoordinator(
        runtime_for_scene(scene, respect_expected_accuracy=True),
        profile_catalog=profile_catalog,
        link_specs=links,
        link_snapshots=link_snaps,
        optimizer_registry=scheduling.registry,
        fallback_optimizer=scheduling.fallback_optimizer,
    )
    report_a = simulate_style.run(
        workflow,
        algorithm="binary_offload",
        seed=seed,
        max_attempts=3,
        deterministic=True,
    )
    metrics_a = _coordinator_report_metrics(
        report_a,
        scene,
        workflow,
        simulate_style,
        scheduling.evaluation_weights,
    )
    _print_block(
        "Simulate-style path (respect_expected_accuracy=True)",
        str(report_a.workflow.get("state", "")),
        metrics_a,
        list(report_a.task_results),
    )

    # --- Path B: Runtime API default (no respect_expected_accuracy) ---
    runtime_default = coordinator_for_scene(
        scene,
        optimizer_registry=scheduling.registry,
        fallback_optimizer=scheduling.fallback_optimizer,
    )
    runtime_default.profile_catalog = profile_catalog
    report_b = runtime_default.run(
        workflow,
        algorithm="binary_offload",
        seed=seed,
        max_attempts=3,
        deterministic=True,
    )
    metrics_b = _coordinator_report_metrics(
        report_b,
        scene,
        workflow,
        runtime_default,
        scheduling.evaluation_weights,
    )
    _print_block(
        "Runtime API path (current default: NO profile failure sampling)",
        str(report_b.workflow.get("state", "")),
        metrics_b,
        list(report_b.task_results),
    )

    print("\n" + "=" * 60)
    print("Interpretation")
    print(
        "  • expected_success_ratio often MATCHES across paths because "
        "workflow_metrics uses profile failure_rate at the chosen node, "
        "not realized attempt outcomes."
    )
    print(
        "  • Compare failed attempts / retry_count / workflow_state instead."
    )
    if (
        metrics_b.get("retry_count") == 0
        and _task_outcomes(list(report_b.task_results))["failed_attempts"] == 0
        and _task_outcomes(list(report_a.task_results))["failed_attempts"] > 0
    ):
        print(
            "  • This run shows profile failures on the Simulate-style path only; "
            "Runtime default skipped sampling."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--amplified-failure-rate",
        type=float,
        default=0.55,
        help="Override local_llm_10b profile failure_rate for a visible demo",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Generate scene via LLM (requires backend/.env credentials)",
    )
    args = parser.parse_args()
    run_demo(args.seed, args.amplified_failure_rate, args.use_llm)


if __name__ == "__main__":
    main()
