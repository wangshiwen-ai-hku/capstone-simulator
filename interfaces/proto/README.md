# MARS protocol contracts

`mars/v1/` contains transport-neutral, data-only Protocol Buffer contracts.
They can be carried by gRPC, DDS, deployment-specific middleware, files, or
replay logs. No RPC service, generated stub, or transport implementation is
defined here.

## Contract boundaries

| File | Contract |
| --- | --- |
| `common.proto` | Shared enums and typed diagnostics |
| `workflow.proto` | Task, workflow DAG, data ports, and artifact references |
| `topology.proto` | Node/link inventory, dynamic state, and reservations |
| `optimization.proto` | Shared optimization problem and scheduling plan |
| `runtime.proto` | Registration, heartbeat, dispatch, completion, and cancellation messages |
| `profiling.proto` | Raw execution observations and aggregated profile snapshots |

The optimization boundary is:

```text
SchedulingProblem
+-- SchedulingSnapshot  immutable captured facts
+-- SchedulingPolicy    objectives and domain constraints
+-- SolveLimits         algorithm-independent solve limits
```

This v1 Problem is a rolling-horizon epoch contract: `SchedulingEpoch`
contains the currently ready tasks, while the workflow layer enforces the full
DAG and `critical_tail_estimates` carries downstream look-ahead. A future
whole-DAG optimizer would require an additive workflow/DAG context rather
than silently reinterpreting this epoch contract.

An optimizer translates that common problem into its own mathematical form.
Algorithm parameters such as ADMM `rho`, primal-dual step size, or MIP gap
configuration remain optimizer-local. `SchedulingPlan` carries
`problem_id`, `snapshot_id`, policy identity, and optimizer identity so a
result can be reproduced and compared.

Iteration traces and warm-start continuations are currently caller-owned
`OptimizerSolveState` data included in the in-process coordinator report; they
are intentionally outside the v1 PC-to-agent runtime contract. A future remote
optimizer transport must add a versioned mapping instead of serializing an
optimizer-specific payload implicitly.

`SchedulingPlan.objective_key` is the comparison value: one component for a
weighted sum, or one component per lexicographic priority group. Soft
constraint penalties enter their declared priority group. The legacy
`objective_value` is the complete weighted sum or only the first
lexicographic component.

Safety placement, node capacity, and link feasibility remain domain
invariants enforced by common validation. A policy may add stricter typed
constraints or define soft penalties, but it cannot relax those invariants.

Business payloads such as detection bounding boxes do not belong in these
control-plane files. `ArtifactRef.message_type` names the payload Proto, while
the payload bytes travel through the selected data plane.
`InputArtifactBinding` binds that reusable Artifact to an exact consumer task
and input port, so fan-out and multiple same-type inputs remain unambiguous.
The same immutable bindings are carried through the Python Snapshot digest and
`DispatchCommand`; `input_artifacts` is only a read-only, de-duplicated payload
projection used by the in-process executor.

## Source of truth and mapping

These Proto files are the source of truth for serialized cross-process and
cross-language data. The current Python dataclasses remain the executable
in-process domain contracts and validation source:

| Proto area | Current Python mapping |
| --- | --- |
| task and workflow | `mars/domain/task.py`, `mars/domain/workflow.py`, `mars/domain/artifact.py` |
| topology and transfers | `mars/domain/topology.py`, `mars/domain/transfer.py` |
| assignments and completion | `mars/domain/execution.py` |
| optimization facts and plans | `mars/optimizers/base.py` |
| optimization policy and limits | `mars/optimizers/policy.py` |
| runtime | `mars/runtime/base.py` |
| profiling | `mars/profiling.py` |

`mars/models.py` is a compatibility re-export and contains no independent
domain definitions.

The Python planning path already uses `SchedulingSnapshot`,
`SchedulingPolicy`, `SolveLimits`, objective/constraint evaluations, and Plan
correlation identifiers. Freshness metadata, raw profiling observations, and
active runtime reservation details remain forward contracts. New objective
metrics or constraint variants are added only after the Python evaluator or a
negotiated optimizer capability can execute them. An adapter must map all
fields explicitly; generated Proto classes must not replace domain validation.
Changes that affect both representations require mapping contract tests before
a transport is enabled.

The ObjectiveMetric contract test compares the complete Python and Proto
catalogs and locks existing numeric wire assignments; adding a metric therefore
requires an executable `MetricDefinition` and an explicit Proto value together.

Presence-aware scalar fields use Proto `optional`: absence maps to the
documented domain default, while an explicitly supplied `false` or zero is
preserved. Adapters must reject blank identifiers, `UNSPECIFIED` enums, missing
required measurements, and values outside the Python domain model's ranges.

Field names include their units (`_ms`, `_mb`, `_mbps`, `_j`, `_watts`,
`_ratio`). Timeline values are relative to the enclosing control-plane or
simulation timeline unless an integration explicitly establishes a wall-clock
epoch.

## Ownership

| Contract | Primary owner | Required review |
| --- | --- | --- |
| common and workflow | Core Infra / Interfaces | Scheduler and Runtime |
| topology and runtime | Agent Runtime & Simulation | Core Infra / Interfaces |
| optimization | Scheduler & Optimizer | Core Infra / Interfaces |
| profiling | Agent Runtime & Simulation | Scheduler & Optimizer |
| business payload Proto | Owning workload team | Core Infra / Interfaces |

Compatibility rules:

- keep enum zero values as `UNSPECIFIED`;
- never reuse a published field number or enum value;
- add compatible fields within `mars.v1`;
- use a new package version for breaking semantic changes.

Syntax can be checked from the repository root without generating stubs:

```sh
protoc -I . \
  --include_imports \
  --descriptor_set_out=/tmp/mars-v1.pb \
  interfaces/proto/mars/v1/*.proto
```
