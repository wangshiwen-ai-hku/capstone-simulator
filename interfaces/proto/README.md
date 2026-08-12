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
SchedulingSolveRequest
+-- SchedulingProblem
|   +-- SchedulingSnapshot  immutable captured facts
|   +-- SchedulingPolicy    objectives and domain constraints
|   +-- SolveLimits         algorithm-independent solve limits
|   +-- metric_contract_id  referenced metric-semantics fingerprint
+-- FormulationSpec         versioned decision domain and materializer
+-- OptimizerSpec           search implementation and configuration identity
```

This v1 Problem is a rolling-horizon epoch contract: `SchedulingEpoch`
contains the currently ready tasks, while the workflow layer enforces the full
DAG and `critical_tail_estimates` carries downstream look-ahead. A future
whole-DAG optimizer would require an additive workflow/DAG context rather
than silently reinterpreting this epoch contract.

`problem_id` identifies what is being solved and does not include a formulation
or optimizer. `metric_contract_id` binds the semantic versions of every metric
referenced by the Policy, so changing executable scoring semantics also changes
the Problem identity. `solve_request_id` additionally binds the versioned
Formulation, materializer, optimizer, and optimizer configuration digest; it is
deterministic, while an in-process trace `solve_id` identifies one invocation.
`continuation_contract_id` deliberately excludes the changing Snapshot while
binding the complete Policy, metric semantics, Formulation/materializer, and
optimizer version/config. It also binds the Problem/Snapshot schema versions
and deterministic/random-seed semantics while excluding work budgets; warm
state is resumed only when this ID matches.

An optimizer searches the selected Formulation compiled from the common
Problem. Algorithm parameters such as ADMM `rho`, primal-dual step size, or MIP
gap remain optimizer-local; `OptimizerSpec.optimizer_config_digest` binds their
identity without serializing implementation objects. `SchedulingPlan` carries
Problem, metric contract, solve request, formulation, policy, and optimizer
identity so a result can be reproduced and compared.
`SolveLimits.solve_budget_ms` is a cooperative orchestration deadline. A
compiled in-process solver receives the shared absolute deadline and late
results are rejected; hard interruption of an uncooperative synchronous
plug-in requires transport or worker-process isolation rather than a Proto
field.

The `one_hot_placement` v1 Formulation selects exactly one feasible candidate
per ready task in epoch order, with candidates ordered by node ID. Drop, defer,
split, replication, alternate task orders, and free start-time decisions are
outside that decision domain. `SOLVE_STATUS_OPTIMAL` therefore means optimal
within the Formulation and materializer recorded by the Plan, not over every
Plan shape accepted by common validation.

Iteration traces and warm-start continuations are currently caller-owned
`OptimizerSolveState` data included in the in-process coordinator report; they
are intentionally outside the v1 PC-to-agent runtime contract. A future remote
optimizer transport must add a versioned mapping instead of serializing an
optimizer-specific payload implicitly.

PC-to-agent dispatch contains only the deterministic `solve_request_id` in
addition to the already-validated Plan fragment. Agents do not receive a
compiled formulation, metric evaluator, or optimizer configuration.
The dispatched `Assignment` carries output size and expected success
probability because the runtime uses both to create artifacts and sample task
completion; an absent Proto success probability maps to the Python default of
`1.0`, while an explicit zero remains zero.

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
| metric semantics and contract IDs | `mars/optimizers/evaluation.py` |
| formulation and solve-request identity | `mars/optimizers/formulation.py`, `mars/optimizers/formulations/` |
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
`metric_contract_id` fingerprints only metrics referenced by a Policy, so an
unrelated catalog addition does not invalidate an existing Problem identity.

`FormulationSpec` is declarative and versioned. Its digest covers formulation
and materializer versions plus typed configuration. Compiled solver models,
callables, trace state, and warm-start payloads are process-local and must never
be encoded as formulation options.

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
