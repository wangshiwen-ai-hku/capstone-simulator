# Runtime Adapter and Execution-Agent Architecture

This document defines the boundary between MARS scheduling and task execution.
The design keeps `CentralCoordinator` independent of the process or transport
used by an execution node.

## Terminology

- **Authoring Assistant** is the application-layer assistant that helps a user
  create a workflow. It is not part of runtime execution.
- **Execution Agent** is the runtime abstraction for one node that accepts work,
  accounts for resources, and reports state and outcomes.
- **Runtime Adapter** implements the coordinator-facing `RuntimePort` and maps
  aggregate runtime operations to one or more execution agents.

## Implemented in-process path

```text
CentralCoordinator
        |
        v
RuntimePort
        |
        v
InProcessRuntimeAdapter
        |
        v
SimulationEnvironment
        |
        +----> SimulatedExecutionAgent (robot)
        +----> SimulatedExecutionAgent (edge)
```

Responsibilities are deliberately separated:

| Component | Responsibility |
|---|---|
| `CentralCoordinator` | Owns workflow progress, planning epochs, validated-plan commit, retry, and completion reconciliation. |
| `RuntimePort` | Defines transport-neutral inventory, dispatch, completion, cancellation, and reporting semantics. |
| `InProcessRuntimeAdapter` | Privately owns the simulation environment, validates runtime commands, routes them to a node, correlates attempts with dispatches, and aggregates inventory and reports. |
| `SimulationEnvironment` | Builds and owns simulated nodes and centralizes deterministic sampling, virtual execution behavior, and failure injection. |
| `ExecutionAgent` | Defines the transport-neutral, single-node execution protocol. |
| `SimulatedExecutionAgent` | Implements that protocol using virtual resources and deterministic simulated execution. |

The public `ExecutionAgent` surface consists of `node_spec` and `registered`,
plus asynchronous `register`, `heartbeat`, `dispatch`, `receive_completion`,
and `cancel` operations. This is the behavioral shape that both a local
executor and a future remote-agent proxy must preserve. Aggregate diagnostics
remain a `RuntimePort` responsibility because a long-lived node proxy does not
own a workflow-wide reporting horizon.

`InProcessRuntime` remains as a compatibility subclass/name for existing
callers, preserving its import and constructor behavior. New code should use
`InProcessRuntimeAdapter`; removal of the compatibility name, if ever
appropriate, requires a separately announced deprecation cycle.
The adapter intentionally does not expose its mutable environment or agents;
all lifecycle operations must pass through `RuntimePort`. `SimulationEnvironment`
remains public for standalone simulator construction and component testing.

## Future network adapters

The following shapes are extension guidance, **not implemented features** in
this branch:

```text
CentralCoordinator                   CentralCoordinator
        |                                    |
     RuntimePort                          RuntimePort
        |                                    |
 GrpcRuntimeAdapter                  DDSRuntimeAdapter
        |                                    |
 GrpcExecutionAgentProxy             DDSExecutionAgentProxy
 (ExecutionAgent)                    (ExecutionAgent)
        |                                    |
 gRPC channel / stub                 DDS participant / topics
        |                                    |
 node-side AgentRuntime service      node-side execution bridge
        |                                    |
 real node executor                  real node executor
```

A `GrpcRuntimeAdapter` would aggregate one client proxy per remote node. Each
proxy would implement `ExecutionAgent` while owning channels, stubs,
serialization, deadlines, and stream recovery. Its node-side service would
translate RPC messages into the same execution semantics. A `DDSRuntimeAdapter`
would similarly aggregate proxy agents that own participants, topics,
quality-of-service policy, and message correlation. Its node-side bridge would
translate DDS samples into those same semantics.

Transport-specific agent classes are therefore client proxies, not alternative
business-level executors. Explicit names such as `GrpcExecutionAgentProxy` and
`DDSExecutionAgentProxy` avoid the ambiguity of `GrpcAgent` and `DDSAgent`.
The adapter and service/bridge own transport concerns; the execution semantics
remain those of `ExecutionAgent`. A deployment-specific host must not redefine
task, resource, acknowledgement, completion, cancellation, or error semantics.

This simulator refactor intentionally does not add empty gRPC or DDS proxy
classes. It provides their shared protocol and the in-process reference
implementation; a later transport integration should add the proxies and
service/bridge together with end-to-end tests.

## Required semantic parity

Every future `RuntimePort` implementation must preserve:

- node identity, capabilities, snapshots, and liveness information;
- validated assignment and reservation boundaries;
- stable attempt and dispatch correlation identifiers;
- acknowledgement, completion, cancellation, and error lifecycle semantics;
- artifact bindings and auditable scheduling identifiers.

Internal mechanics need not match. The simulator may use virtual time and
failure injection, while a deployed executor may use hardware telemetry, wall
time, and middleware-specific recovery.

## Contract tests and migration

`tests/runtime_contract.py` supplies reusable conformance harnesses for both
`RuntimePort` and `ExecutionAgent`. They cover registration, heartbeat,
capabilities, reporting, dispatch/completion correlation, duplicate dispatch,
cancellation, unknown or offline nodes, and capacity rejection. The in-process
adapter and simulated agent run them through `tests/test_runtime_boundaries.py`,
which also verifies multi-node routing. Every future network adapter and proxy
should instantiate the same harnesses. Transport tests then add concerns such
as serialization, connection loss, ordering, and quality of service.

Migration is intentionally incremental:

1. Prefer `InProcessRuntimeAdapter` in new code.
2. Keep `InProcessRuntime` working for existing imports and constructors.
3. Integrate future transports only through `RuntimePort`.
4. Require contract-test parity before a network adapter is production-ready.
