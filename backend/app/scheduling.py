"""Scheduling capability metadata and request-scoped optimizer wiring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from mars.optimizers import (
    BinaryOffloadOptimizer,
    HeuristicOptimizer,
    OptimizerRegistry,
    SolveLimits,
)
from mars.workflow_metrics import WorkflowEvaluationWeights


DEFAULT_SUCCESS_WEIGHT = 1.0
DEFAULT_COMMUNICATION_WEIGHT = 1.0
DEFAULT_UTILIZATION_WEIGHT = 2.0
DEFAULT_FORMULATION = BinaryOffloadOptimizer.default_formulation_id
OPTIMIZER_FORMULATIONS = {
    "binary_offload": tuple(
        sorted(BinaryOffloadOptimizer.supported_formulation_ids)
    ),
    "heuristic": tuple(sorted(HeuristicOptimizer.supported_formulation_ids)),
}
SUPPORTED_FORMULATIONS = OPTIMIZER_FORMULATIONS["binary_offload"]


@dataclass(frozen=True)
class SchedulingConfiguration:
    registry: OptimizerRegistry | None
    fallback_optimizer: str | None
    formulation: str | None
    optimizer_options: Mapping[str, float]
    evaluation_weights: WorkflowEvaluationWeights


def configure_scheduling(
    algorithm: str,
    optimizer_options: Mapping[str, float] | None = None,
    *,
    formulation: str | None = None,
    legacy_beta: float | None = None,
) -> SchedulingConfiguration:
    """Validate API options and construct one request-local optimizer."""

    options = dict(optimizer_options or {})
    if algorithm != "binary_offload":
        resolved_formulation = (
            formulation.strip() if formulation is not None else None
        )
        if formulation is not None and not resolved_formulation:
            raise ValueError("formulation must be non-blank")
        supported = OPTIMIZER_FORMULATIONS["heuristic"]
        if resolved_formulation not in (None, *supported):
            raise ValueError(
                f"unsupported heuristic formulation: "
                f"{resolved_formulation!r}; supported formulations: "
                f"{list(supported)!r}"
            )
        if options:
            raise ValueError(
                f"optimizer_options are not supported by algorithm {algorithm!r}"
            )
        # Older dashboard builds sent beta for every algorithm. It never
        # configured heuristic policies, so accepting and ignoring it here is
        # the compatible behavior while structured options remain strict.
        return SchedulingConfiguration(
            registry=None,
            fallback_optimizer="heuristic",
            formulation=resolved_formulation,
            optimizer_options={},
            evaluation_weights=WorkflowEvaluationWeights(),
        )

    resolved_formulation = (
        formulation.strip()
        if formulation is not None
        else DEFAULT_FORMULATION
    )
    if not resolved_formulation:
        raise ValueError("formulation must be non-blank")
    if resolved_formulation not in SUPPORTED_FORMULATIONS:
        raise ValueError(
            "unsupported binary_offload formulation: "
            f"{resolved_formulation!r}; supported formulations: "
            f"{list(SUPPORTED_FORMULATIONS)!r}"
        )

    if legacy_beta is not None:
        configured = options.get("communication_weight")
        if configured is not None and not math.isclose(
            configured,
            legacy_beta,
        ):
            raise ValueError(
                "beta and optimizer_options.communication_weight "
                "must agree when both are provided"
            )
        options.setdefault("communication_weight", legacy_beta)

    allowed = {
        "success_weight",
        "communication_weight",
        "utilization_weight",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(
            f"unknown binary_offload optimizer options: {sorted(unknown)}"
        )
    success = options.get("success_weight", DEFAULT_SUCCESS_WEIGHT)
    communication = options.get(
        "communication_weight",
        DEFAULT_COMMUNICATION_WEIGHT,
    )
    utilization = options.get(
        "utilization_weight",
        DEFAULT_UTILIZATION_WEIGHT,
    )
    registry = OptimizerRegistry()
    registry.register(
        BinaryOffloadOptimizer(
            alpha=success,
            beta=communication,
            gamma=utilization,
        )
    )
    normalized = {
        "success_weight": success,
        "communication_weight": communication,
        "utilization_weight": utilization,
    }
    return SchedulingConfiguration(
        registry=registry,
        fallback_optimizer="heuristic",
        formulation=resolved_formulation,
        optimizer_options=normalized,
        evaluation_weights=WorkflowEvaluationWeights(
            success=success,
            communication=communication,
            utilization=utilization,
        ),
    )


def scheduling_capabilities() -> dict[str, object]:
    """Return additive, UI-consumable scheduling metadata."""

    default_limits = SolveLimits()
    stable = (
        ("dag_deadline", "DAG deadline"),
        ("rule_based", "Rule based"),
        ("local_first", "Local first"),
        ("edge_first", "Edge first"),
        ("greedy_cost", "Greedy cost"),
    )
    algorithms: list[dict[str, object]] = [
        {
            "id": algorithm_id,
            "label": label,
            "kind": "policy_alias",
            "stability": "stable",
            "execution_paths": ["runtime", "simulation"],
            "parameters": {},
            "compatibility": {
                "supported_node_kinds": ["robot", "edge"],
                "supports_multiple_nodes": True,
                "requires_source_candidate": False,
            },
            "default_formulation": None,
            "supported_formulations": list(
                OPTIMIZER_FORMULATIONS["heuristic"]
            ),
        }
        for algorithm_id, label in stable
    ]
    algorithms.append(
        {
            "id": "binary_offload",
            "label": "Bounded exhaustive placement",
            "kind": "optimizer",
            "stability": "experimental",
            "execution_paths": ["runtime", "simulation"],
            "parameters": {
                "success_weight": {
                    "type": "number",
                    "label": "Expected-success trade-off",
                    "default": DEFAULT_SUCCESS_WEIGHT,
                    "minimum": 0.0,
                    "step": 0.1,
                    "description": (
                        "Weight on the priority-weighted probability of "
                        "successful execution."
                    ),
                },
                "communication_weight": {
                    "type": "number",
                    "label": "Communication trade-off",
                    "default": DEFAULT_COMMUNICATION_WEIGHT,
                    "minimum": 0.0,
                    "step": 0.1,
                    "description": (
                        "Weight on communication time normalized by task "
                        "latency budgets."
                    ),
                },
                "utilization_weight": {
                    "type": "number",
                    "label": "Peak-utilization trade-off",
                    "default": DEFAULT_UTILIZATION_WEIGHT,
                    "minimum": 0.0,
                    "step": 0.1,
                    "description": (
                        "Weight on peak CPU, GPU, or memory utilization."
                    ),
                },
            },
            "compatibility": {
                # The optimizer core accepts every NodeKind, while the Web
                # process-local runtime currently implements robot and edge.
                "supported_node_kinds": ["robot", "edge"],
                "supports_multiple_nodes": True,
                "requires_source_candidate": False,
            },
            "search": {
                "scope": "ready_set",
                "strategy": "bounded_exhaustive",
                "solve_budget_ms": default_limits.solve_budget_ms,
                "max_iterations": default_limits.max_iterations,
                "fallback_optimizer": "heuristic",
            },
            "default_formulation": DEFAULT_FORMULATION,
            "supported_formulations": list(SUPPORTED_FORMULATIONS),
        }
    )
    return {
        "schema_version": "mars.scheduling-capabilities.v1",
        "algorithms": algorithms,
    }


__all__ = [
    "DEFAULT_FORMULATION",
    "SchedulingConfiguration",
    "SUPPORTED_FORMULATIONS",
    "configure_scheduling",
    "scheduling_capabilities",
]
