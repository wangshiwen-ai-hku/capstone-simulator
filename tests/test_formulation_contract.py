from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from mars.domain import (
    ExecutionMode,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    PlacementConstraints,
    TaskClass,
    TaskInstance,
    TaskSpec,
)
from mars.optimizers import (
    BinaryOffloadOptimizer,
    ConstraintRelation,
    ConstraintSpec,
    FormulationCompatibilityError,
    FormulationDomainError,
    FormulationRegistry,
    FormulationSpec,
    HeuristicOptimizer,
    ObjectiveAggregation,
    ObjectiveMetric,
    ObjectiveSpec,
    OneHotPlacementFormulation,
    OptimizationDirection,
    OptimizerContinuation,
    OptimizerRegistry,
    OptimizerSolveState,
    PlanValidationError,
    SchedulingEpoch,
    SchedulingPolicy,
    SchedulingSolveRequest,
    SolveLimits,
    SolveStatus,
    SolveTracePhase,
    build_solve_request,
    built_in_formulation_registry,
    metric_contract_id,
    prepare_solve,
    compile_solve_request,
    validate_plan,
)
from mars.scheduler import build_scheduling_problem, plan_scheduling_epoch


class _FailingFormulatedOptimizer(HeuristicOptimizer):
    optimizer_id = "failing_formulated"
    optimizer_config_digest = "failing-formulated.v1"

    def solve_formulated(self, prepared):
        raise RuntimeError("intentional formulated failure")


class _RelaxingFallbackOptimizer(HeuristicOptimizer):
    optimizer_id = "relaxing_fallback"
    optimizer_config_digest = "relaxing-fallback.v1"

    def solve_formulated(self, prepared):
        raise RuntimeError("intentional preserved-domain failure")


class _SameStackFailingOptimizer(HeuristicOptimizer):
    optimizer_id = "same-stack-failing"
    optimizer_config_digest = "same-stack-failing.v1"
    default_formulation_id = "one_hot_placement"

    def __init__(self) -> None:
        self.calls = 0

    def solve_formulated(self, prepared):
        self.calls += 1
        raise RuntimeError("intentional same-stack failure")


class _ForgingUnformulatedOptimizer(HeuristicOptimizer):
    optimizer_id = "forging-unformulated"
    optimizer_config_digest = "forging-unformulated.v1"

    def solve(self, problem):
        return replace(
            super().solve(problem),
            optimizer_id=self.optimizer_id,
            optimizer_version=self.optimizer_version,
            solve_request_id="forged-request",
            formulation_id="forged-formulation",
            formulation_version="99",
            formulation_digest="forged-digest",
        )


def _task(task_id: str) -> TaskInstance:
    return TaskInstance(
        task_id=task_id,
        workflow_id="formulation-workflow",
        name=task_id,
        source_node_id="z_robot",
        spec=TaskSpec(
            task_type="placement-test",
            task_class=TaskClass.REALTIME_OFFLOADABLE,
            compute_demand=1.0,
            input_size_mb=0.0,
            placement_constraints=PlacementConstraints(
                allowed_node_kinds=(NodeKind.ROBOT, NodeKind.EDGE),
                allow_source_node=True,
                allow_other_robots=False,
            ),
        ),
        deadline_time_ms=10_000.0,
    )


def _inputs(
    *,
    policy: str | SchedulingPolicy = "binary_offload",
    solve_limits: SolveLimits | None = None,
) -> dict[str, object]:
    tasks = (_task("task_b"), _task("task_a"))
    return {
        "epoch": SchedulingEpoch("formulation-epoch", 0.0, tasks),
        "node_specs": {
            "z_robot": NodeSpec(
                "z_robot",
                NodeKind.ROBOT,
                cpu_capacity=8.0,
                gpu_capacity=2.0,
                memory_gb=16.0,
                bandwidth_mbps=200.0,
                base_latency_ms=1.0,
                max_concurrency=4,
            ),
            "a_edge": NodeSpec(
                "a_edge",
                NodeKind.EDGE,
                cpu_capacity=16.0,
                gpu_capacity=8.0,
                memory_gb=64.0,
                bandwidth_mbps=1_000.0,
                base_latency_ms=1.0,
                max_concurrency=4,
            ),
        },
        "node_snapshots": {
            "z_robot": NodeSnapshot("z_robot", power_w=20.0),
            "a_edge": NodeSnapshot("a_edge", power_w=100.0),
        },
        "parent_artifacts": {},
        "ready_time_ms": {task.task_id: 0.0 for task in tasks},
        "policy": policy,
        "solve_limits": solve_limits or SolveLimits(solve_budget_ms=1_000.0),
    }


def _problem(
    *,
    policy: str | SchedulingPolicy = "binary_offload",
    solve_limits: SolveLimits | None = None,
):
    values = _inputs(policy=policy, solve_limits=solve_limits)
    epoch = values.pop("epoch")
    assert isinstance(epoch, SchedulingEpoch)
    return build_scheduling_problem(epoch, **values)


def test_formulation_spec_is_frozen_canonical_and_serializable() -> None:
    first = FormulationSpec(
        formulation_id="test",
        formulation_version="1",
        materializer_id="timeline",
        materializer_version="2",
        options={"beta": 2, "alpha": True},
    )
    second = FormulationSpec(
        formulation_id="test",
        formulation_version="1",
        materializer_id="timeline",
        materializer_version="2",
        options={"alpha": True, "beta": 2},
    )

    assert first.formulation_digest == second.formulation_digest
    assert hash(first) == hash(second)
    assert first.as_dict() == second.as_dict()
    with pytest.raises(FrozenInstanceError):
        first.formulation_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.options["alpha"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="unique after normalization"):
        FormulationSpec(
            formulation_id="test",
            formulation_version="1",
            materializer_id="timeline",
            materializer_version="2",
            options={"alpha": True, " alpha ": False},
        )


def test_optimizer_rejects_same_formulation_id_with_changed_contract() -> None:
    changed_version = replace(
        OneHotPlacementFormulation().spec,
        formulation_version="2",
    )
    changed_materializer = replace(
        OneHotPlacementFormulation().spec,
        materializer_version="2",
    )

    assert not BinaryOffloadOptimizer.supports_formulation(changed_version)
    assert not HeuristicOptimizer.supports_formulation(changed_version)
    assert not BinaryOffloadOptimizer.supports_formulation(
        changed_materializer
    )


@pytest.mark.parametrize(
    "optimizer",
    (BinaryOffloadOptimizer(), HeuristicOptimizer()),
)
def test_formulated_optimizer_rejects_foreign_config_identity(
    optimizer,
) -> None:
    problem = _problem()
    formulation = OneHotPlacementFormulation()
    request = build_solve_request(problem, optimizer, formulation)
    forged = replace(request, optimizer_config_digest="foreign-config")
    prepared = compile_solve_request(forged, formulation)

    with pytest.raises(ValueError, match="version/config"):
        optimizer.solve_formulated(prepared)


def test_builtin_registry_exposes_one_hot_placement() -> None:
    registry = built_in_formulation_registry()

    assert registry.ids() == ("one_hot_placement",)
    assert isinstance(
        registry.resolve("one_hot_placement"),
        OneHotPlacementFormulation,
    )
    assert registry.specs()[0].options == {
        "assignment_cardinality": "exactly_one",
        "allow_drop": False,
        "allow_defer": False,
        "allow_split": False,
        "allow_replication": False,
        "task_order": "epoch",
        "candidate_order": "node_id",
    }


def test_registry_replacement_updates_canonical_spec_and_existing_aliases() -> None:
    class StubFormulation:
        def __init__(self, version: str) -> None:
            self.spec = FormulationSpec(
                formulation_id="stub",
                formulation_version=version,
                materializer_id="stub-materializer",
                materializer_version="1",
            )

        def compile(self, problem):
            raise NotImplementedError

        def materialize(self, problem, model, decision, *, optimizer_id):
            raise NotImplementedError

        def evaluate(self, problem, model, plan):
            raise NotImplementedError

        def validate_plan_domain(self, problem, model, plan):
            raise NotImplementedError

    registry = FormulationRegistry()
    registry.register(StubFormulation("1"), aliases=("legacy-stub",))
    registry.register(StubFormulation("2"), replace=True)

    assert registry.resolve("stub").spec.formulation_version == "2"
    assert registry.resolve("legacy-stub").spec.formulation_version == "2"
    assert tuple(spec.formulation_version for spec in registry.specs()) == (
        "2",
    )

    replacement = FormulationRegistry()
    replacement.register(StubFormulation("3"))
    registry.extend(replacement, replace=True)

    assert registry.resolve("stub").spec.formulation_version == "3"
    assert registry.resolve("legacy-stub").spec.formulation_version == "3"
    assert tuple(spec.formulation_version for spec in registry.specs()) == (
        "3",
    )


def test_one_hot_compile_has_stable_groups_and_complete_policy_contract() -> None:
    problem = _problem()
    prepared = prepare_solve(
        problem,
        BinaryOffloadOptimizer(),
        OneHotPlacementFormulation(),
    )
    model = prepared.model

    assert model.problem_id == problem.problem_id
    assert model.metric_contract_id == problem.metric_contract_id
    assert model.ordered_task_ids == ("task_b", "task_a")
    assert tuple(
        tuple(candidate.node_id for candidate in options)
        for options in model.candidate_options
    ) == (("a_edge", "z_robot"), ("a_edge", "z_robot"))
    assert model.total_decisions == 4
    assert model.objective_ids == tuple(
        item.objective_id for item in problem.policy.objectives
    )
    assert model.constraint_ids == tuple(
        item.constraint_id for item in problem.policy.constraints
    )


@pytest.mark.parametrize("metric", tuple(ObjectiveMetric))
@pytest.mark.parametrize(
    "aggregation",
    (ObjectiveAggregation.WEIGHTED_SUM, ObjectiveAggregation.LEXICOGRAPHIC),
)
def test_one_hot_exact_compile_covers_every_policy_metric_role(
    metric: ObjectiveMetric,
    aggregation: ObjectiveAggregation,
) -> None:
    policy = SchedulingPolicy(
        policy_id=f"coverage-{metric.value}-{aggregation.value}",
        version="1",
        objectives=(
            ObjectiveSpec(
                objective_id="objective",
                metric=metric,
                direction=OptimizationDirection.MAXIMIZE,
            ),
        ),
        constraints=(
            ConstraintSpec(
                constraint_id="hard-upper",
                metric=metric,
                relation=ConstraintRelation.LESS_THAN_OR_EQUAL,
                bound=1_000_000.0,
            ),
            ConstraintSpec(
                constraint_id="soft-lower",
                metric=metric,
                relation=ConstraintRelation.GREATER_THAN_OR_EQUAL,
                bound=-1_000_000.0,
                hard=False,
                violation_penalty=10.0,
            ),
        ),
        objective_aggregation=aggregation,
    )

    model = OneHotPlacementFormulation().compile(_problem(policy=policy))

    assert model.objective_ids == ("objective",)
    assert model.constraint_ids == ("hard-upper", "soft-lower")
    assert (metric.value, "1") in model.referenced_metric_versions


def test_compile_fails_closed_on_metric_semantics_mismatch() -> None:
    problem = _problem()

    with pytest.raises(
        FormulationCompatibilityError,
        match="metric contract",
    ):
        OneHotPlacementFormulation().compile(
            replace(problem, metric_contract_id="metric-contract:stale")
        )


def test_problem_and_solve_request_identities_are_independent() -> None:
    problem = _problem()
    formulation = OneHotPlacementFormulation()
    binary_request = build_solve_request(
        problem,
        BinaryOffloadOptimizer(),
        formulation,
    )
    heuristic_request = build_solve_request(
        problem,
        HeuristicOptimizer(),
        formulation,
    )
    variant = FormulationSpec(
        formulation_id="one_hot_variant",
        formulation_version="1",
        materializer_id="serial_transfer_earliest_resource",
        materializer_version="1",
    )
    variant_request = SchedulingSolveRequest(
        problem=problem,
        formulation_spec=variant,
        optimizer_id=binary_request.optimizer_id,
        optimizer_version=binary_request.optimizer_version,
        optimizer_config_digest=binary_request.optimizer_config_digest,
    )

    assert binary_request.problem.problem_id == heuristic_request.problem.problem_id
    assert binary_request.solve_request_id != heuristic_request.solve_request_id
    assert binary_request.solve_request_id != variant_request.solve_request_id
    assert build_solve_request(
        problem,
        BinaryOffloadOptimizer(),
        formulation,
    ).solve_request_id == binary_request.solve_request_id


def test_solve_request_accepts_an_equal_rehydrated_problem_value() -> None:
    problem = _problem()
    optimizer = BinaryOffloadOptimizer()
    formulation = OneHotPlacementFormulation()
    prepared = prepare_solve(problem, optimizer, formulation)
    plan = optimizer.solve_formulated(prepared)
    equal_problem = replace(problem)

    validated = validate_plan(
        equal_problem,
        plan,
        solve_request=replace(
            prepared.request,
            problem=equal_problem,
        ),
    )

    assert validated.solve_request_id == prepared.request.solve_request_id


def test_continuation_contract_excludes_snapshot_but_binds_policy_and_solver() -> None:
    problem = _problem()
    formulation = OneHotPlacementFormulation()
    optimizer = BinaryOffloadOptimizer()
    original = build_solve_request(problem, optimizer, formulation)
    next_snapshot = build_solve_request(
        replace(problem, problem_id="next-frame-problem"),
        optimizer,
        formulation,
    )
    changed_policy = replace(
        problem.policy,
        objectives=(
            replace(problem.policy.objectives[0], weight=3.0),
            *problem.policy.objectives[1:],
        ),
    )
    changed_policy_request = build_solve_request(
        replace(
            problem,
            problem_id="changed-policy-problem",
            policy=changed_policy,
            metric_contract_id=metric_contract_id(changed_policy),
        ),
        optimizer,
        formulation,
    )
    changed_optimizer = BinaryOffloadOptimizer()
    changed_optimizer.optimizer_config_digest = "bounded-exhaustive.v2"
    changed_optimizer_request = build_solve_request(
        problem,
        changed_optimizer,
        formulation,
    )
    changed_seed_request = build_solve_request(
        replace(
            problem,
            problem_id="changed-seed-problem",
            solve_limits=replace(problem.solve_limits, random_seed=17),
        ),
        optimizer,
        formulation,
    )
    changed_schema_request = build_solve_request(
        replace(
            problem,
            problem_id="changed-schema-problem",
            schema_version="mars.scheduling-problem.v2",
        ),
        optimizer,
        formulation,
    )

    assert original.solve_request_id != next_snapshot.solve_request_id
    assert (
        original.continuation_contract_id
        == next_snapshot.continuation_contract_id
    )
    assert (
        original.continuation_contract_id
        != changed_policy_request.continuation_contract_id
    )
    assert (
        original.continuation_contract_id
        != changed_optimizer_request.continuation_contract_id
    )
    assert (
        original.continuation_contract_id
        != changed_seed_request.continuation_contract_id
    )
    assert (
        original.continuation_contract_id
        != changed_schema_request.continuation_contract_id
    )


def test_binary_default_and_explicit_one_hot_are_equivalent() -> None:
    values = _inputs()
    epoch = values.pop("epoch")
    assert isinstance(epoch, SchedulingEpoch)
    default = plan_scheduling_epoch(
        epoch,
        optimizer="binary_offload",
        fallback_optimizer=None,
        **values,
    )
    explicit = plan_scheduling_epoch(
        epoch,
        optimizer="binary_offload",
        formulation="one_hot_placement",
        fallback_optimizer=None,
        **values,
    )

    assert default.assignments == explicit.assignments
    assert default.objective_key == explicit.objective_key
    assert default.solve_request_id == explicit.solve_request_id
    assert explicit.formulation_id == "one_hot_placement"
    assert explicit.metric_contract_id == metric_contract_id(explicit_policy := _problem().policy)
    assert explicit.policy_id == explicit_policy.policy_id


def test_optimal_is_scoped_to_exhausted_formulation_domain() -> None:
    full = BinaryOffloadOptimizer().solve(_problem())
    limited = BinaryOffloadOptimizer().solve(
        _problem(
            solve_limits=SolveLimits(
                solve_budget_ms=1_000.0,
                max_iterations=1,
            )
        )
    )

    assert full.solve_status is SolveStatus.OPTIMAL
    assert full.diagnostics["formulation_exhausted"] is True
    assert full.iteration_count == full.diagnostics["total_combinations"]
    assert limited.solve_status is SolveStatus.ITERATION_LIMIT
    assert limited.diagnostics["formulation_exhausted"] is False
    assert limited.iteration_count < limited.diagnostics["total_combinations"]


def test_formulated_trace_and_continuation_bind_complete_identity() -> None:
    problem = _problem()
    optimizer = BinaryOffloadOptimizer()
    formulation = OneHotPlacementFormulation()
    request = build_solve_request(problem, optimizer, formulation)
    state = OptimizerSolveState(session_id="formulated-state")

    optimizer.solve_with_state(problem, state)
    contexts = {entry.context for entry in state.entries}
    assert len(contexts) == 1
    context = contexts.pop()
    assert context.solve_request_id == request.solve_request_id
    assert (
        context.continuation_contract_id
        == request.continuation_contract_id
    )
    assert context.metric_contract_id == problem.metric_contract_id
    assert context.formulation_id == formulation.spec.formulation_id
    assert context.formulation_digest == formulation.spec.formulation_digest

    continuation = OptimizerContinuation(
        optimizer_id=optimizer.optimizer_id,
        optimizer_version=optimizer.optimizer_version,
        optimizer_config_digest=optimizer.optimizer_config_digest,
        schema_version="binary.continuation.v1",
        source_solve_request_id=request.solve_request_id,
        continuation_contract_id=request.continuation_contract_id,
        updated_problem_id=problem.problem_id,
        metric_contract_id=problem.metric_contract_id,
        formulation_id=formulation.spec.formulation_id,
        formulation_version=formulation.spec.formulation_version,
        formulation_digest=formulation.spec.formulation_digest,
        payload={"next_combination": 2},
    )
    state.set_continuation(continuation)

    assert state.continuation_for_request(request) == continuation
    assert (
        state.continuation_for(
            optimizer.optimizer_id,
            optimizer_version="different",
            optimizer_config_digest=optimizer.optimizer_config_digest,
            continuation_contract_id=request.continuation_contract_id,
            formulation_id=formulation.spec.formulation_id,
            formulation_version=formulation.spec.formulation_version,
            formulation_digest=formulation.spec.formulation_digest,
            metric_contract_id=problem.metric_contract_id,
        )
        is None
    )


@pytest.mark.parametrize(
    "values",
    (
        {"objective_key": (float("nan"),)},
        {"objective_components": {"score": float("inf")}},
        {"details": {"opaque": object()}},
    ),
)
def test_solve_trace_rejects_non_serializable_or_non_finite_values(
    values,
) -> None:
    problem = _problem()
    state = OptimizerSolveState(session_id="invalid-trace")
    context = state.begin(problem, optimizer_id="test")

    with pytest.raises(ValueError):
        state.record(context, SolveTracePhase.ITERATION, **values)


def test_explicit_formulation_is_rejected_by_unformulated_plugin() -> None:
    class StatelessPlugin:
        optimizer_id = "stateless"

        def solve(self, problem):
            return HeuristicOptimizer().solve(problem)

    with pytest.raises(
        FormulationCompatibilityError,
        match="does not implement",
    ):
        plan_scheduling_epoch(
            _inputs().pop("epoch"),
            optimizer=StatelessPlugin(),
            formulation="one_hot_placement",
            fallback_optimizer=None,
            **{
                key: value
                for key, value in _inputs().items()
                if key != "epoch"
            },
        )


def test_one_hot_domain_validation_rejects_drop_even_if_policy_allows_it() -> None:
    problem = _problem()
    optimizer = BinaryOffloadOptimizer()
    prepared = prepare_solve(
        problem,
        optimizer,
        OneHotPlacementFormulation(),
    )
    valid = optimizer.solve_formulated(prepared)
    dropped = replace(
        valid.assignments[0],
        target_node_id="",
        execution_mode=ExecutionMode.DROP,
        estimated_finish_ms=valid.assignments[0].estimated_start_ms,
        compute_ms=0.0,
        communication_ms=0.0,
        energy_j=0.0,
        transfer_link_ids=(),
        output_size_mb=0.0,
        success_probability=0.0,
    )
    forged = replace(
        valid,
        assignments=(dropped, *valid.assignments[1:]),
        node_reservations=tuple(
            item
            for item in valid.node_reservations
            if item.task_id != dropped.task_id
        ),
        transfer_reservations=tuple(
            item
            for item in valid.transfer_reservations
            if item.task_id != dropped.task_id
        ),
    )

    with pytest.raises(FormulationDomainError, match="compiled candidate"):
        prepared.formulation.validate_plan_domain(
            problem,
            prepared.model,
            forged,
        )


def test_fallback_preserves_requested_formulation_and_exact_identity() -> None:
    registry = OptimizerRegistry()
    registry.register(_FailingFormulatedOptimizer())
    state = OptimizerSolveState(session_id="preserved-fallback")
    values = _inputs()
    epoch = values.pop("epoch")

    plan = plan_scheduling_epoch(
        epoch,
        optimizer="failing_formulated",
        formulation="one_hot_placement",
        registry=registry,
        solve_state=state,
        **values,
    )

    assert plan.formulation_id == "one_hot_placement"
    assert plan.diagnostics["formulation_changed"] is False
    assert plan.diagnostics["formulation_relaxed"] is False
    assert plan.diagnostics["repaired_from_formulation_digest"] == (
        plan.diagnostics["fallback_formulation_digest"]
    )
    fallback = state.invocation_summaries("heuristic")[-1]
    assert fallback["terminal_phase"] == "fallback"
    assert fallback["formulation_id"] == "one_hot_placement"


def test_fallback_relaxation_is_explicit_after_preservation_fails() -> None:
    registry = OptimizerRegistry()
    registry.register(_FailingFormulatedOptimizer())
    registry.register(_RelaxingFallbackOptimizer())
    state = OptimizerSolveState(session_id="relaxed-fallback")
    values = _inputs()
    epoch = values.pop("epoch")

    plan = plan_scheduling_epoch(
        epoch,
        optimizer="failing_formulated",
        formulation="one_hot_placement",
        fallback_optimizer="relaxing_fallback",
        registry=registry,
        solve_state=state,
        **values,
    )

    assert plan.formulation_id == ""
    assert plan.diagnostics["formulation_changed"] is True
    assert plan.diagnostics["formulation_relaxed"] is True
    assert plan.diagnostics["fallback_attempt_count"] == 2
    fallback = state.invocation_summaries("relaxing_fallback")[-1]
    assert fallback["terminal_phase"] == "fallback"
    assert fallback["formulation_relaxed"] is True


def test_formulated_binary_fallback_retains_exhaustion_trace() -> None:
    registry = OptimizerRegistry()
    registry.register(_FailingFormulatedOptimizer())
    state = OptimizerSolveState(session_id="binary-fallback")
    values = _inputs()
    epoch = values.pop("epoch")

    plan = plan_scheduling_epoch(
        epoch,
        optimizer="failing_formulated",
        formulation="one_hot_placement",
        fallback_optimizer="binary_offload",
        registry=registry,
        solve_state=state,
        **values,
    )

    summary = state.invocation_summaries("binary_offload")[-1]
    assert plan.diagnostics["formulation_exhausted"] is True
    assert summary["terminal_phase"] == "fallback"
    assert summary["formulation_exhausted"] is True
    assert summary["enumerated_combinations"] == 4
    assert summary["total_combinations"] == 4


def test_fallback_does_not_repeat_the_rejected_optimizer_stack() -> None:
    optimizer = _SameStackFailingOptimizer()
    values = _inputs()
    epoch = values.pop("epoch")

    with pytest.raises(RuntimeError, match="fallback.*also failed"):
        plan_scheduling_epoch(
            epoch,
            optimizer=optimizer,
            formulation="one_hot_placement",
            fallback_optimizer=optimizer,
            **values,
        )

    assert optimizer.calls == 1


def test_unformulated_optimizer_cannot_forge_formulation_provenance() -> None:
    values = _inputs()
    epoch = values.pop("epoch")

    with pytest.raises(
        PlanValidationError,
        match="cannot claim formulation provenance",
    ):
        plan_scheduling_epoch(
            epoch,
            optimizer=_ForgingUnformulatedOptimizer(),
            formulation=None,
            fallback_optimizer=None,
            **values,
        )


def test_late_time_limited_fallback_cannot_cross_shared_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]

    class SlowFailure(HeuristicOptimizer):
        optimizer_id = "slow-primary"
        optimizer_config_digest = "slow-primary.v1"

        def solve_formulated(self, prepared):
            now[0] += 0.007
            raise RuntimeError("primary consumed most of the budget")

    class LateIncumbent(HeuristicOptimizer):
        optimizer_id = "late-incumbent"
        optimizer_config_digest = "late-incumbent.v1"

        def solve_formulated(self, prepared):
            plan = super().solve_formulated(prepared)
            now[0] += 0.008
            return replace(plan, solve_status=SolveStatus.TIME_LIMIT)

    monkeypatch.setattr("mars.scheduler.perf_counter", lambda: now[0])
    monkeypatch.setattr(
        "mars.optimizers.formulation.perf_counter",
        lambda: now[0],
    )
    monkeypatch.setattr(
        "mars.optimizers.heuristics.perf_counter",
        lambda: now[0],
    )
    values = _inputs(
        solve_limits=SolveLimits(solve_budget_ms=10.0),
    )
    epoch = values.pop("epoch")

    with pytest.raises(RuntimeError, match="fallback.*also failed"):
        plan_scheduling_epoch(
            epoch,
            optimizer=SlowFailure(),
            formulation="one_hot_placement",
            fallback_optimizer=LateIncumbent(),
            **values,
        )

    assert now[0] == pytest.approx(0.015)
