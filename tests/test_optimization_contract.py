from __future__ import annotations

from dataclasses import replace

import pytest

from mars.domain import (
    ArtifactRef,
    InputArtifactBinding,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    PlacementConstraints,
    TaskClass,
    TaskInstance,
    TaskSpec,
)
from mars.optimizers import (
    BUILTIN_METRICS,
    CandidateFidelity,
    ConstraintRelation,
    ConstraintSpec,
    HeuristicOptimizer,
    MetricScope,
    ObjectiveEvaluation,
    ObjectiveMetric,
    OptimizerRegistry,
    PlanValidationError,
    SchedulingEpoch,
    SchedulingPolicy,
    SchedulingProblem,
    SolveLimits,
    SolveStatus,
    built_in_policy,
    candidate_objective_key,
    candidate_proxy_key,
    validate_plan,
)
from mars.scheduler import (
    build_scheduling_problem,
    plan_scheduling_epoch,
)


LEGACY_ALIASES = (
    "dag_deadline",
    "rule_based",
    "local_first",
    "edge_first",
    "greedy_cost",
)


def _nodes() -> dict[str, NodeSpec]:
    return {
        "robot": NodeSpec(
            node_id="robot",
            kind=NodeKind.ROBOT,
            cpu_capacity=4,
            gpu_capacity=2,
            memory_gb=16,
            bandwidth_mbps=100,
            base_latency_ms=2,
        ),
        "edge": NodeSpec(
            node_id="edge",
            kind=NodeKind.EDGE,
            cpu_capacity=12,
            gpu_capacity=8,
            memory_gb=64,
            bandwidth_mbps=500,
            base_latency_ms=5,
        ),
    }


def _snapshots() -> dict[str, NodeSnapshot]:
    return {
        "robot": NodeSnapshot("robot", power_w=25),
        "edge": NodeSnapshot("edge", power_w=120),
    }


def _task(task_id: str = "perception") -> TaskInstance:
    return TaskInstance(
        task_id=task_id,
        workflow_id="workflow",
        name=task_id,
        source_node_id="robot",
        spec=TaskSpec(
            task_type="yolo_inference",
            task_class=TaskClass.REALTIME_OFFLOADABLE,
            compute_demand=2,
            gpu_demand=1,
            placement_constraints=PlacementConstraints(
                allowed_node_kinds=(NodeKind.ROBOT, NodeKind.EDGE),
                preferred_node_kinds=(NodeKind.EDGE,),
                allow_source_node=True,
            ),
        ),
        deadline_time_ms=10_000,
    )


def _epoch(task_id: str = "perception") -> SchedulingEpoch:
    return SchedulingEpoch(
        epoch_id=f"epoch:{task_id}",
        now_ms=0,
        ready_tasks=(_task(task_id),),
    )


def _plan_kwargs(task_id: str = "perception") -> dict[str, object]:
    return {
        "node_specs": _nodes(),
        "node_snapshots": _snapshots(),
        "parent_artifacts": {},
        "ready_time_ms": {task_id: 0},
        "link_specs": (),
        "link_snapshots": (),
    }


def _problem(
    *,
    policy: str | SchedulingPolicy = "greedy_cost",
    task_id: str = "perception",
) -> SchedulingProblem:
    return build_scheduling_problem(
        _epoch(task_id),
        policy=policy,
        **_plan_kwargs(task_id),
    )


class _RecordingOptimizer:
    def __init__(self, optimizer_id: str = "recording") -> None:
        self.optimizer_id = optimizer_id
        self.problem: SchedulingProblem | None = None

    def solve(self, problem: SchedulingProblem):
        self.problem = problem
        baseline = HeuristicOptimizer().solve(problem)
        return replace(
            baseline,
            optimizer_id=self.optimizer_id,
            assignments=tuple(
                replace(
                    assignment,
                    optimizer_id=self.optimizer_id,
                )
                for assignment in baseline.assignments
            ),
        )


class _ExplodingOptimizer:
    optimizer_id = "exploding"

    def __init__(self) -> None:
        self.problem: SchedulingProblem | None = None

    def solve(self, problem: SchedulingProblem):
        self.problem = problem
        raise RuntimeError("optimizer failed")


class _MutatingOptimizer:
    optimizer_id = "mutating"

    def __init__(self) -> None:
        self.problem: SchedulingProblem | None = None
        self.original_candidates = ()

    def solve(self, problem: SchedulingProblem):
        self.problem = problem
        task_id = problem.epoch.ready_tasks[0].task_id
        self.original_candidates = problem.candidates[task_id]
        problem.candidates[task_id] = ()  # type: ignore[index]


def test_builtin_metric_registry_is_complete_and_immutable() -> None:
    expected_proto_names = {
        ObjectiveMetric.MAKESPAN_MS: "OBJECTIVE_METRIC_MAKESPAN_MS",
        ObjectiveMetric.TOTAL_DEADLINE_VIOLATION_MS: (
            "OBJECTIVE_METRIC_TOTAL_DEADLINE_VIOLATION_MS"
        ),
        ObjectiveMetric.TOTAL_COMPLETION_TIME_MS: (
            "OBJECTIVE_METRIC_TOTAL_COMPLETION_TIME_MS"
        ),
        ObjectiveMetric.CRITICAL_PATH_FINISH_MS: (
            "OBJECTIVE_METRIC_CRITICAL_PATH_FINISH_MS"
        ),
        ObjectiveMetric.TOTAL_ENERGY_J: "OBJECTIVE_METRIC_TOTAL_ENERGY_J",
        ObjectiveMetric.TOTAL_COMMUNICATION_MS: (
            "OBJECTIVE_METRIC_TOTAL_COMMUNICATION_TIME_MS"
        ),
        ObjectiveMetric.LOCALITY_PENALTY: "OBJECTIVE_METRIC_LOCALITY_PENALTY",
        ObjectiveMetric.DROPPED_TASKS: "OBJECTIVE_METRIC_DROPPED_TASK_COUNT",
        ObjectiveMetric.NON_SOURCE_ASSIGNMENTS: (
            "OBJECTIVE_METRIC_NON_SOURCE_ASSIGNMENT_COUNT"
        ),
        ObjectiveMetric.NON_EDGE_ASSIGNMENTS: (
            "OBJECTIVE_METRIC_NON_EDGE_ASSIGNMENT_COUNT"
        ),
        ObjectiveMetric.PLACEMENT_PREFERENCE_PENALTY: (
            "OBJECTIVE_METRIC_PLACEMENT_PREFERENCE_PENALTY"
        ),
        ObjectiveMetric.RULE_MISMATCH_COUNT: (
            "OBJECTIVE_METRIC_RULE_MISMATCH_COUNT"
        ),
        ObjectiveMetric.EXPECTED_WEIGHTED_SUCCESS_RATIO: (
            "OBJECTIVE_METRIC_EXPECTED_WEIGHTED_SUCCESS_RATIO"
        ),
        ObjectiveMetric.NORMALIZED_COMMUNICATION_RATIO: (
            "OBJECTIVE_METRIC_NORMALIZED_COMMUNICATION_RATIO"
        ),
        ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION: (
            "OBJECTIVE_METRIC_MAXIMUM_RESOURCE_UTILIZATION"
        ),
        ObjectiveMetric.DEFERRED_PRIORITY_PENALTY: (
            "OBJECTIVE_METRIC_DEFERRED_PRIORITY_PENALTY"
        ),
    }

    assert set(BUILTIN_METRICS) == set(ObjectiveMetric)
    assert len(BUILTIN_METRICS) == 16
    assert {
        metric: definition.proto_enum_name
        for metric, definition in BUILTIN_METRICS.items()
    } == expected_proto_names
    assert all(
        definition.metric is metric
        and definition.semantics_version == "1"
        and isinstance(definition.scope, MetricScope)
        and isinstance(
            definition.candidate_fidelity,
            CandidateFidelity,
        )
        and callable(definition.evaluate_plan)
        for metric, definition in BUILTIN_METRICS.items()
    )
    assert {
        metric
        for metric, definition in BUILTIN_METRICS.items()
        if definition.candidate_fidelity is CandidateFidelity.PROXY
    } == {
        ObjectiveMetric.MAKESPAN_MS,
        ObjectiveMetric.CRITICAL_PATH_FINISH_MS,
        ObjectiveMetric.NORMALIZED_COMMUNICATION_RATIO,
        ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION,
    }
    assert {
        metric
        for metric, definition in BUILTIN_METRICS.items()
        if definition.scope is MetricScope.TIMELINE
    } == {ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION}

    with pytest.raises(TypeError):
        BUILTIN_METRICS[ObjectiveMetric.MAKESPAN_MS] = (  # type: ignore[index]
            BUILTIN_METRICS[ObjectiveMetric.MAKESPAN_MS]
        )
    with pytest.raises((AttributeError, TypeError)):
        BUILTIN_METRICS[ObjectiveMetric.MAKESPAN_MS].unit = (  # type: ignore[misc]
            "seconds"
        )


def test_builtin_metric_registry_preserves_raw_metric_semantics() -> None:
    problem = _problem(policy="greedy_cost")
    plan = validate_plan(problem, HeuristicOptimizer().solve(problem))
    plan_values = {
        metric: definition.evaluate_plan(problem, plan)
        for metric, definition in BUILTIN_METRICS.items()
    }

    assert plan_values == pytest.approx(
        {
            ObjectiveMetric.MAKESPAN_MS: 8.333333333333334,
            ObjectiveMetric.TOTAL_DEADLINE_VIOLATION_MS: 0.0,
            ObjectiveMetric.TOTAL_COMPLETION_TIME_MS: 8.333333333333334,
            ObjectiveMetric.CRITICAL_PATH_FINISH_MS: 8.333333333333334,
            ObjectiveMetric.TOTAL_ENERGY_J: 1.0,
            ObjectiveMetric.TOTAL_COMMUNICATION_MS: 0.0,
            ObjectiveMetric.LOCALITY_PENALTY: 2.0,
            ObjectiveMetric.DROPPED_TASKS: 0.0,
            ObjectiveMetric.NON_SOURCE_ASSIGNMENTS: 1.0,
            ObjectiveMetric.NON_EDGE_ASSIGNMENTS: 0.0,
            ObjectiveMetric.PLACEMENT_PREFERENCE_PENALTY: 0.0,
            ObjectiveMetric.RULE_MISMATCH_COUNT: 0.0,
            ObjectiveMetric.EXPECTED_WEIGHTED_SUCCESS_RATIO: 1.0,
            ObjectiveMetric.NORMALIZED_COMMUNICATION_RATIO: 0.0,
            ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION: 0.125,
            ObjectiveMetric.DEFERRED_PRIORITY_PENALTY: 0.0,
        }
    )

    task_id = problem.epoch.ready_tasks[0].task_id
    candidate = next(
        item for item in problem.candidates[task_id] if item.node_id == "robot"
    )
    assert BUILTIN_METRICS[
        ObjectiveMetric.DEFERRED_PRIORITY_PENALTY
    ].candidate_proxy is None
    candidate_values = {
        metric: definition.candidate_proxy(problem, task_id, candidate)
        for metric, definition in BUILTIN_METRICS.items()
        if definition.candidate_proxy is not None
    }
    assert candidate_values == pytest.approx(
        {
            ObjectiveMetric.MAKESPAN_MS: 28.571428571428573,
            ObjectiveMetric.TOTAL_DEADLINE_VIOLATION_MS: 0.0,
            ObjectiveMetric.TOTAL_COMPLETION_TIME_MS: 28.571428571428573,
            ObjectiveMetric.CRITICAL_PATH_FINISH_MS: 28.571428571428573,
            ObjectiveMetric.TOTAL_ENERGY_J: 0.7142857142857143,
            ObjectiveMetric.TOTAL_COMMUNICATION_MS: 0.0,
            ObjectiveMetric.LOCALITY_PENALTY: 0.0,
            ObjectiveMetric.DROPPED_TASKS: 0.0,
            ObjectiveMetric.NON_SOURCE_ASSIGNMENTS: 0.0,
            ObjectiveMetric.NON_EDGE_ASSIGNMENTS: 1.0,
            ObjectiveMetric.PLACEMENT_PREFERENCE_PENALTY: 1.0,
            ObjectiveMetric.RULE_MISMATCH_COUNT: 1.0,
            ObjectiveMetric.EXPECTED_WEIGHTED_SUCCESS_RATIO: 1.0,
            ObjectiveMetric.NORMALIZED_COMMUNICATION_RATIO: 0.0,
            ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION: 0.5,
        }
    )
    assert candidate_objective_key(
        problem,
        task_id,
        candidate,
    ) == candidate_proxy_key(problem, task_id, candidate)


def test_legacy_algorithm_aliases_are_reserved_from_solver_registry() -> None:
    class AmbiguousOptimizer:
        optimizer_id = "dag_deadline"

        def solve(self, problem: SchedulingProblem):
            raise AssertionError("reserved optimizer must not run")

    with pytest.raises(ValueError, match="reserved for policy"):
        OptimizerRegistry().register(AmbiguousOptimizer())


@pytest.mark.parametrize("algorithm", LEGACY_ALIASES)
def test_legacy_algorithm_alias_selects_heuristic_and_policy(
    algorithm: str,
) -> None:
    plan = plan_scheduling_epoch(
        _epoch(),
        optimizer=algorithm,
        **_plan_kwargs(),
    )

    assert plan.optimizer_id == "heuristic"
    assert plan.policy_id == algorithm
    assert plan.policy_version == built_in_policy(algorithm).version
    assert plan.solve_status is SolveStatus.FEASIBLE
    assert plan.optimizer_version == "1"
    assert plan.iteration_count == 1


def test_explicit_heuristic_policy_matches_legacy_alias() -> None:
    alias = plan_scheduling_epoch(
        _epoch(),
        optimizer="edge_first",
        **_plan_kwargs(),
    )
    explicit = plan_scheduling_epoch(
        _epoch(),
        optimizer="heuristic",
        policy="edge_first",
        **_plan_kwargs(),
    )

    assert explicit.problem_id == alias.problem_id
    assert explicit.snapshot_id == alias.snapshot_id
    assert explicit.policy_id == alias.policy_id == "edge_first"
    assert explicit.assignments == alias.assignments
    assert explicit.objective_evaluations == alias.objective_evaluations
    assert explicit.objective_value == alias.objective_value


def test_explicit_solve_limits_are_preserved_without_budget_ambiguity() -> None:
    limits = SolveLimits(
        solve_budget_ms=25,
        max_iterations=200,
        deterministic=True,
        random_seed=7,
    )
    problem = build_scheduling_problem(
        _epoch(),
        solve_limits=limits,
        **_plan_kwargs(),
    )

    assert problem.solve_limits is limits
    assert problem.solve_budget_ms == 25
    with pytest.raises(ValueError, match="not both"):
        build_scheduling_problem(
            _epoch(),
            solve_limits=limits,
            solve_budget_ms=10,
            **_plan_kwargs(),
        )


def test_snapshot_and_problem_ids_cover_their_exact_contract_content() -> None:
    baseline = _problem(policy="greedy_cost")
    changed_state = build_scheduling_problem(
        _epoch(),
        policy="greedy_cost",
        node_specs=_nodes(),
        node_snapshots={
            **_snapshots(),
            "robot": NodeSnapshot(
                "robot",
                cpu_util=0.7,
                power_w=25,
            ),
        },
        parent_artifacts={},
        ready_time_ms={"perception": 0},
        link_specs=(),
        link_snapshots=(),
    )
    changed_limits = build_scheduling_problem(
        _epoch(),
        policy="greedy_cost",
        solve_limits=SolveLimits(solve_budget_ms=25),
        **_plan_kwargs(),
    )
    changed_policy = _problem(policy="edge_first")

    assert changed_state.snapshot.snapshot_id != baseline.snapshot.snapshot_id
    assert changed_state.problem_id != baseline.problem_id
    assert (
        changed_limits.snapshot.snapshot_id
        == baseline.snapshot.snapshot_id
    )
    assert changed_limits.problem_id != baseline.problem_id
    assert (
        changed_policy.snapshot.snapshot_id
        == baseline.snapshot.snapshot_id
    )
    assert changed_policy.problem_id != baseline.problem_id


def test_input_bindings_are_hashed_and_shared_artifacts_transfer_once() -> None:
    artifact = ArtifactRef(
        artifact_id="shared-detections",
        producer_task_id="detector",
        node_id="robot",
        size_mb=2,
        producer_port="detections",
        message_type="example.DetectionArray",
    )
    bindings = (
        InputArtifactBinding(
            consumer_task_id="perception",
            consumer_port="primary_detections",
            artifact=artifact,
        ),
        InputArtifactBinding(
            consumer_task_id="perception",
            consumer_port="secondary_detections",
            artifact=artifact,
        ),
    )
    common = {
        key: value
        for key, value in _plan_kwargs().items()
        if key != "parent_artifacts"
    }
    baseline = build_scheduling_problem(
        _epoch(),
        input_artifact_bindings={"perception": bindings},
        **common,
    )
    changed_binding = build_scheduling_problem(
        _epoch(),
        input_artifact_bindings={
            "perception": (
                bindings[0],
                replace(
                    bindings[1],
                    consumer_port="tertiary_detections",
                ),
            )
        },
        **common,
    )

    assert baseline.input_artifact_bindings["perception"] == bindings
    feasible = next(
        candidate
        for candidate in baseline.candidates["perception"]
        if candidate.feasible
    )
    assert feasible.input_locations == ("robot",)
    assert len(feasible.transfers) == 1
    assert baseline.snapshot.snapshot_id != changed_binding.snapshot.snapshot_id
    assert baseline.problem_id != changed_binding.problem_id


def test_snapshot_maps_and_nested_candidate_sequences_are_immutable() -> None:
    problem = _problem()

    assert all(
        isinstance(node.capabilities, tuple)
        and isinstance(node.supported_models, tuple)
        for node in problem.node_specs
    )
    assert all(
        isinstance(candidate.input_locations, tuple)
        and isinstance(candidate.transfers, tuple)
        and all(
            isinstance(transfer.path_link_ids, tuple)
            for transfer in candidate.transfers
        )
        for candidates in problem.candidates.values()
        for candidate in candidates
    )
    with pytest.raises(TypeError):
        problem.candidates["perception"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        problem.node_available_ms["robot"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        problem.link_available_ms["missing"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        problem.critical_tail_ms["perception"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        problem.candidates["perception"][0] = (  # type: ignore[index]
            problem.candidates["perception"][0]
        )


def test_domain_sequences_are_copied_from_mutable_inputs() -> None:
    capabilities = ["inference"]
    models = ["yolo"]
    allowed_kinds = [NodeKind.EDGE]
    dependencies = ["upstream"]
    node = NodeSpec(
        node_id="edge",
        kind=NodeKind.EDGE,
        cpu_capacity=4,
        gpu_capacity=2,
        memory_gb=16,
        bandwidth_mbps=100,
        base_latency_ms=2,
        capabilities=capabilities,  # type: ignore[arg-type]
        supported_models=models,  # type: ignore[arg-type]
    )
    placement = PlacementConstraints(
        allowed_node_kinds=allowed_kinds,  # type: ignore[arg-type]
    )
    task = TaskInstance(
        task_id="consumer",
        workflow_id="workflow",
        name="consumer",
        source_node_id="edge",
        spec=TaskSpec(
            task_type="consumer",
            task_class=TaskClass.REALTIME_OFFLOADABLE,
            placement_constraints=placement,
        ),
        dependency_task_ids=dependencies,  # type: ignore[arg-type]
    )
    capabilities.append("mutated")
    models.append("mutated")
    allowed_kinds.append(NodeKind.ROBOT)
    dependencies.append("mutated")

    assert node.capabilities == ("inference",)
    assert node.supported_models == ("yolo",)
    assert placement.allowed_node_kinds == (NodeKind.EDGE,)
    assert task.dependency_task_ids == ("upstream",)


def test_failed_mutation_is_repaired_without_corrupting_fallback_input() -> None:
    optimizer = _MutatingOptimizer()
    registry = OptimizerRegistry()
    registry.register(optimizer)

    plan = plan_scheduling_epoch(
        _epoch(),
        optimizer=optimizer.optimizer_id,
        registry=registry,
        policy="edge_first",
        **_plan_kwargs(),
    )

    assert optimizer.problem is not None
    assert (
        optimizer.problem.candidates["perception"]
        == optimizer.original_candidates
    )
    assert plan.optimizer_id == "heuristic"
    assert plan.policy_id == "edge_first"
    assert plan.assignments[0].target_node_id in {
        item.node_id for item in optimizer.original_candidates
    }
    assert plan.diagnostics["repaired_from_optimizer"] == "mutating"
    assert "TypeError" in str(plan.diagnostics["repair_reason"])


@pytest.mark.parametrize(
    "field",
    ("problem_id", "snapshot_id", "policy_id", "policy_version"),
)
def test_validator_rejects_wrong_plan_correlation(field: str) -> None:
    problem = _problem()
    raw_plan = HeuristicOptimizer().solve(problem)

    with pytest.raises(PlanValidationError, match=field):
        validate_plan(problem, replace(raw_plan, **{field: "wrong"}))


def test_validator_recomputes_forged_objective_fields() -> None:
    problem = _problem()
    raw_plan = HeuristicOptimizer().solve(problem)
    expected = validate_plan(problem, raw_plan)
    forged = replace(
        raw_plan,
        objective_value=-123_456,
        objective_key=(-123_456,),
        objective_evaluations=(
            ObjectiveEvaluation(
                objective_id="forged",
                metric=ObjectiveMetric.TOTAL_ENERGY_J,
                priority_order=0,
                raw_value=-1,
                normalized_value=-1,
                weighted_value=-1,
            ),
        ),
    )

    validated = validate_plan(problem, forged)

    assert validated.objective_value == expected.objective_value
    assert validated.objective_value != forged.objective_value
    assert validated.objective_key == expected.objective_key
    assert validated.objective_value == validated.objective_key[0]
    assert (
        validated.objective_evaluations
        == expected.objective_evaluations
    )
    assert all(
        item.objective_id != "forged"
        for item in validated.objective_evaluations
    )


def test_validator_rejects_forged_locality_inputs() -> None:
    problem = _problem(policy="dag_deadline")
    raw_plan = HeuristicOptimizer().solve(problem)
    assignment = raw_plan.assignments[0]
    assert assignment.input_locations
    forged = replace(
        raw_plan,
        assignments=(
            replace(assignment, input_locations=()),
        ),
    )

    with pytest.raises(PlanValidationError, match="input locations"):
        validate_plan(problem, forged)


def test_validator_rejects_custom_hard_constraint_violation() -> None:
    policy = replace(
        built_in_policy("greedy_cost"),
        policy_id="must-drop-negative-tasks",
        constraints=(
            ConstraintSpec(
                constraint_id="impossible_drop_bound",
                metric=ObjectiveMetric.DROPPED_TASKS,
                relation=ConstraintRelation.LESS_THAN_OR_EQUAL,
                bound=-1,
                hard=True,
            ),
        ),
    )
    problem = _problem(policy=policy)
    raw_plan = HeuristicOptimizer().solve(problem)

    with pytest.raises(
        PlanValidationError,
        match="impossible_drop_bound",
    ):
        validate_plan(problem, raw_plan)


@pytest.mark.parametrize(
    "status",
    (SolveStatus.INFEASIBLE, SolveStatus.ERROR),
)
def test_validator_rejects_noncommittable_solve_status(
    status: SolveStatus,
) -> None:
    problem = _problem()
    raw_plan = HeuristicOptimizer().solve(problem)

    with pytest.raises(PlanValidationError, match="not committable"):
        validate_plan(
            problem,
            replace(raw_plan, solve_status=status),
        )


def test_heuristic_honors_max_iterations_with_a_partial_incumbent() -> None:
    epoch = SchedulingEpoch(
        epoch_id="limited-epoch",
        now_ms=0,
        ready_tasks=(_task("first"), _task("second")),
    )
    plan = plan_scheduling_epoch(
        epoch,
        optimizer="heuristic",
        policy="greedy_cost",
        solve_limits=SolveLimits(max_iterations=1),
        node_specs=_nodes(),
        node_snapshots=_snapshots(),
        parent_artifacts={},
        ready_time_ms={"first": 0, "second": 0},
        link_specs=(),
        link_snapshots=(),
    )

    assert plan.solve_status is SolveStatus.ITERATION_LIMIT
    assert plan.iteration_count == 1
    assert len(plan.assignments) == 1
    assert len(plan.deferred_task_ids) == 1
    assert {
        plan.assignments[0].task_id,
        plan.deferred_task_ids[0],
    } == {"first", "second"}


def test_soft_constraint_penalty_enters_its_objective_key_priority() -> None:
    policy = replace(
        built_in_policy("greedy_cost"),
        policy_id="energy-penalty",
        constraints=(
            ConstraintSpec(
                constraint_id="energy_over_zero",
                metric=ObjectiveMetric.TOTAL_ENERGY_J,
                relation=ConstraintRelation.LESS_THAN_OR_EQUAL,
                bound=0,
                hard=False,
                violation_penalty=10,
                priority_order=0,
            ),
        ),
    )
    problem = _problem(policy=policy)
    validated = validate_plan(
        problem,
        HeuristicOptimizer().solve(problem),
    )
    primary_objectives = sum(
        item.weighted_value
        for item in validated.objective_evaluations
        if item.priority_order == 0
    )
    penalty = validated.constraint_evaluations[0].penalty

    assert penalty > 0
    assert validated.objective_key[0] == pytest.approx(
        primary_objectives + penalty
    )


def test_custom_optimizer_receives_canonical_problem_and_exact_policy() -> None:
    policy = replace(
        built_in_policy("greedy_cost"),
        policy_id="team-policy",
        version="2026-07-23",
    )
    optimizer = _RecordingOptimizer()

    plan = plan_scheduling_epoch(
        _epoch(),
        optimizer=optimizer,
        policy=policy,
        **_plan_kwargs(),
    )

    assert isinstance(optimizer.problem, SchedulingProblem)
    assert optimizer.problem.policy is policy
    assert optimizer.problem.problem_id == plan.problem_id
    assert optimizer.problem.snapshot.snapshot_id == plan.snapshot_id
    assert plan.optimizer_id == optimizer.optimizer_id
    assert plan.policy_id == policy.policy_id
    assert plan.policy_version == policy.version


def test_fallback_receives_same_problem_and_exact_policy() -> None:
    policy = replace(
        built_in_policy("dag_deadline"),
        policy_id="deadline-team-policy",
        version="2",
    )
    selected = _ExplodingOptimizer()
    fallback = _RecordingOptimizer("fallback-recording")

    plan = plan_scheduling_epoch(
        _epoch(),
        optimizer=selected,
        fallback_optimizer=fallback,
        policy=policy,
        **_plan_kwargs(),
    )

    assert selected.problem is fallback.problem
    assert fallback.problem is not None
    assert fallback.problem.policy is policy
    assert plan.optimizer_id == fallback.optimizer_id
    assert plan.policy_id == policy.policy_id
    assert plan.policy_version == policy.version
    assert plan.diagnostics["repaired_from_optimizer"] == "exploding"
