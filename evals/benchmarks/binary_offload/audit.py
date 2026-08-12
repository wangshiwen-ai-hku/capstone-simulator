"""Pure projections of production scheduling and solver audit data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json

from .spec import FALLBACK_OPTIMIZER


def _json_counter(values) -> str:
    return json.dumps(
        dict(sorted(Counter(str(value) for value in values).items())),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_mapping(value: Mapping[object, object]) -> str:
    return json.dumps(
        {str(key): value[key] for key in sorted(value, key=str)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def scheduling_audit(report) -> dict[str, object]:
    """Read effective scheduling facts emitted by the production coordinator."""

    scheduling = report.workflow.get("scheduling")
    if not isinstance(scheduling, Mapping):
        raise RuntimeError(
            "simulation report does not expose workflow.scheduling; refusing "
            "to infer effective optimizers or fallback from task placements"
        )
    effective_optimizers = scheduling.get("effective_optimizers")
    effective_policies = scheduling.get("effective_policies")
    if not isinstance(effective_optimizers, Mapping):
        raise RuntimeError("workflow.scheduling.effective_optimizers is missing")
    if not isinstance(effective_policies, Mapping):
        raise RuntimeError("workflow.scheduling.effective_policies is missing")
    requested = str(scheduling.get("requested_algorithm", ""))
    if not requested:
        raise RuntimeError("workflow.scheduling.requested_algorithm is missing")
    solve_limits = scheduling.get("solve_limits")
    if not isinstance(solve_limits, Mapping):
        raise RuntimeError("workflow.scheduling.solve_limits is missing")
    fallback_count = int(scheduling.get("fallback_count", 0))
    return {
        "requested_algorithm": requested,
        "effective_algorithm": (
            next(iter(effective_optimizers))
            if len(effective_optimizers) == 1
            else "mixed"
        ),
        "effective_optimizers": _json_mapping(effective_optimizers),
        "effective_policies": _json_mapping(effective_policies),
        "effective_solver_statuses": (
            _json_mapping(scheduling["solve_statuses"])
            if isinstance(scheduling.get("solve_statuses"), Mapping)
            else "not_exposed_by_simulation_report"
        ),
        "effective_termination_reasons": (
            _json_mapping(scheduling["termination_reasons"])
            if isinstance(scheduling.get("termination_reasons"), Mapping)
            else "not_exposed_by_simulation_report"
        ),
        "fallback_enabled": True,
        "fallback_optimizer": FALLBACK_OPTIMIZER,
        "fallback_count": fallback_count,
        "fallback_used": fallback_count > 0,
        "requested_seed": int(scheduling.get("requested_seed", 0)),
        "deterministic_execution": bool(
            scheduling.get("deterministic", False)
        ),
        "execution_seed": int(scheduling.get("execution_seed", 0)),
        "solve_budget_ms": float(solve_limits["solve_budget_ms"]),
        "max_iterations": int(solve_limits["max_iterations"]),
        "solver_deterministic": bool(solve_limits["deterministic"]),
        "solver_random_seed": int(solve_limits["random_seed"]),
    }


def optimizer_invocation_summaries(
    report,
) -> tuple[Mapping[str, object], ...]:
    """Read caller-owned optimizer summaries emitted by the coordinator."""

    scheduling = report.workflow.get("scheduling")
    if not isinstance(scheduling, Mapping):
        raise RuntimeError("workflow.scheduling is missing")
    solve_state = scheduling.get("optimizer_solve_state")
    if not isinstance(solve_state, Mapping):
        raise RuntimeError(
            "workflow.scheduling.optimizer_solve_state is missing"
        )
    raw_summaries = solve_state.get("invocation_summaries")
    if not isinstance(raw_summaries, (list, tuple)):
        raise RuntimeError(
            "optimizer_solve_state.invocation_summaries is missing"
        )
    if any(not isinstance(item, Mapping) for item in raw_summaries):
        raise RuntimeError("optimizer invocation summaries must be mappings")
    return tuple(raw_summaries)


def solver_audit(
    requested_algorithm: str,
    invocation_summaries: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    """Describe requested binary solves without confusing them with fallback."""

    summaries = (
        tuple(
            item
            for item in invocation_summaries
            if item.get("optimizer_id") == "binary_offload"
        )
        if requested_algorithm == "binary_offload"
        else ()
    )
    budgets = sorted({float(item["solve_budget_ms"]) for item in summaries})
    iteration_limits = sorted(
        {int(item["max_iterations"]) for item in summaries}
    )
    if summaries:
        statuses = _json_counter(item["solve_status"] for item in summaries)
        reasons = _json_counter(
            item["termination_reason"] for item in summaries
        )
    elif requested_algorithm == "binary_offload":
        statuses = "{}"
        reasons = "{}"
    else:
        statuses = "not_exposed_for_policy_alias"
        reasons = "not_exposed_for_policy_alias"
    return {
        "requested_solver_invocations": len(summaries),
        "requested_solver_history_epochs": len(summaries),
        "requested_solver_statuses": statuses,
        "requested_termination_reasons": reasons,
        "observed_solve_budgets_ms": json.dumps(
            budgets,
            separators=(",", ":"),
        ),
        "observed_iteration_limits": json.dumps(
            iteration_limits,
            separators=(",", ":"),
        ),
        "requested_placement_search_exhaustive": (
            all(
                bool(item["placement_search_exhaustive"])
                for item in summaries
            )
            if summaries
            else ""
        ),
        "search_scope": (
            "receding_horizon_ready_epoch"
            if requested_algorithm == "binary_offload"
            else "policy_alias_per_ready_epoch"
        ),
        "global_workflow_exact": False,
    }


__all__ = [
    "optimizer_invocation_summaries",
    "scheduling_audit",
    "solver_audit",
]
