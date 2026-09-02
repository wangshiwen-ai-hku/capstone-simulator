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
| runtime | `mars/runtime/base.py`, `mars/runtime/agent.py` |
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

---

# MARS 协议契约（中文说明）

`mars/v1/` 包含与传输机制无关、仅承载数据的 Protocol Buffer 契约。
这些契约可通过 gRPC、DDS、部署专用中间件、文件或回放日志传输。
此处不定义任何 RPC 服务、生成的存根或传输实现。

## 契约边界

| 文件 | 契约 |
| --- | --- |
| `common.proto` | 共享枚举和类型化诊断信息 |
| `workflow.proto` | 任务、工作流有向无环图（DAG）、数据端口和制品引用 |
| `topology.proto` | 节点/链路清单、动态状态和预留信息 |
| `optimization.proto` | 共享优化问题和调度计划 |
| `runtime.proto` | 注册、心跳、分派、完成和取消消息 |
| `profiling.proto` | 原始执行观测和聚合后的性能剖析快照 |

优化边界如下：

```text
SchedulingSolveRequest
+-- SchedulingProblem
|   +-- SchedulingSnapshot  不可变的已捕获事实
|   +-- SchedulingPolicy    目标与领域约束
|   +-- SolveLimits         与算法无关的求解限制
|   +-- metric_contract_id  被引用指标语义的指纹
+-- FormulationSpec         带版本的决策域与物化器
+-- OptimizerSpec           搜索实现与配置标识
```

此 v1 `Problem` 是一种滚动时域调度轮次契约：`SchedulingEpoch` 包含当前已就绪的任务，
工作流层负责约束完整 DAG，而 `critical_tail_estimates` 则提供对下游任务的前瞻信息。
未来若要支持全 DAG 优化器，应当增补工作流/DAG 上下文，不能在不作说明的情况下
重新解释这一轮次契约。

`problem_id` 标识待求解的问题，不包含模型表述或优化器。
`metric_contract_id` 绑定 `Policy` 所引用的每项指标的语义版本，因此，只要可执行的
评分语义发生变化，`Problem` 标识也会随之变化。`solve_request_id` 还会绑定带版本的
`Formulation`、物化器、优化器以及优化器配置摘要；该标识是确定性的，
而进程内跟踪标识 `solve_id` 则用于标识某一次调用。
`continuation_contract_id` 会有意排除不断变化的 `Snapshot`，同时绑定完整的 `Policy`、
指标语义、`Formulation`/物化器以及优化器版本/配置。它还会绑定 `Problem`/`Snapshot`
的模式版本和确定性/随机种子语义，但不包含工作预算；只有该 ID 匹配时才会恢复
热启动状态。

优化器会搜索由公共 `Problem` 编译而成的选定 `Formulation`。ADMM `rho`、
原始—对偶步长或 MIP gap 等算法参数仍由优化器在本地管理；
`OptimizerSpec.optimizer_config_digest` 绑定这些参数的标识，而无需序列化实现对象。
`SchedulingPlan` 携带 `Problem`、指标契约、求解请求、模型表述、策略和优化器的标识，
以便复现和比较结果。
`SolveLimits.solve_budget_ms` 是一个协作式编排截止时间。编译后的进程内求解器会收到
共享的绝对截止时间，逾期结果将被拒绝；若要强制中断不配合的同步插件，必须依靠
传输层或工作进程隔离，而不能通过某个 Proto 字段来实现。

v1 `one_hot_placement` 模型表述按照各已就绪任务在当前调度轮次中的既定顺序，为每个
任务恰好选择一个可行候选项，候选项按节点 ID 排序。丢弃、延后、拆分、复制、采用其他
任务排序方式以及自由决定开始时间，均不属于该决策域。因此，`SOLVE_STATUS_OPTIMAL`
表示在计划所记录的模型表述和物化器范围内达到最优，而不是在公共验证接受的所有计划
形态中达到最优。

迭代跟踪和热启动延续状态当前属于调用方持有的 `OptimizerSolveState` 数据，包含在
进程内协调器报告中；它们被有意排除在 v1 PC-to-agent 运行时契约之外。未来若引入
远程优化器传输，必须增加带版本的映射，不能隐式序列化优化器专用有效载荷。

PC-to-agent 分派除已验证的计划片段外，仅包含确定性的 `solve_request_id`。Agent 不会
收到已编译的模型表述、指标求值器或优化器配置。分派的 `Assignment` 携带输出大小和
预期成功概率，因为运行时会使用这两者创建制品并对任务完成结果进行采样；Proto 中
缺失的成功概率会映射为 Python 默认值 `1.0`，而显式给出的零值仍保持为零。

`SchedulingPlan.objective_key` 是用于比较的值：对于加权和，它包含一个分量；对于
字典序策略，每个优先级组各包含一个分量。软约束惩罚会计入其声明的优先级组。旧版
`objective_value` 表示完整的加权和，或者仅表示第一个字典序分量。

安全放置、节点容量和链路可行性仍是由公共验证强制执行的领域不变量。策略可以增加
更严格的类型化约束或定义软惩罚，但不能放宽这些不变量。

检测边界框等业务有效载荷不应放入这些控制平面文件中。
`ArtifactRef.message_type` 指定有效载荷 Proto，而有效载荷字节则通过选定的数据平面
传输。`InputArtifactBinding` 会将可复用的制品精确绑定到消费方任务及其输入端口，
从而确保扇出和多个同类型输入仍然含义明确。同一组不可变绑定会贯穿 Python 快照摘要
和 `DispatchCommand`；`input_artifacts` 只是进程内执行器使用的只读、去重有效载荷投影。

## 事实来源与映射

这些 Proto 文件是跨进程和跨语言序列化数据的事实来源。当前 Python 数据类仍是
可执行的进程内领域契约和验证来源：

| Proto 范畴 | 当前 Python 映射 |
| --- | --- |
| 任务和工作流 | `mars/domain/task.py`, `mars/domain/workflow.py`, `mars/domain/artifact.py` |
| 拓扑和传输 | `mars/domain/topology.py`, `mars/domain/transfer.py` |
| 任务分配与完成 | `mars/domain/execution.py` |
| 优化事实和计划 | `mars/optimizers/base.py` |
| 优化策略和限制 | `mars/optimizers/policy.py` |
| 指标语义和契约 ID | `mars/optimizers/evaluation.py` |
| 模型表述和求解请求标识 | `mars/optimizers/formulation.py`, `mars/optimizers/formulations/` |
| 运行时 | `mars/runtime/base.py`, `mars/runtime/agent.py` |
| 性能剖析 | `mars/profiling.py` |

`mars/models.py` 是一个兼容性重导出模块，不包含独立的领域定义。

Python 规划路径已使用 `SchedulingSnapshot`、`SchedulingPolicy`、`SolveLimits`、
目标/约束求值以及计划关联标识符。新鲜度元数据、原始性能剖析观测和活跃运行时资源
预留详情仍属于前瞻性契约。只有在 Python 求值器或经过协商的优化器能力能够执行新目标
指标或约束变体后，才能添加它们。适配器必须显式映射所有字段；生成的 Proto 类不得
取代领域验证。凡是同时影响两种表示的变更，都必须先完成映射契约测试，之后才能启用
传输。

`ObjectiveMetric` 契约测试会比较完整的 Python 和 Proto 指标目录，并锁定既有的
序列化数值编号；因此，添加指标时必须同时提供可执行的 `MetricDefinition` 和显式的
Proto 值。`metric_contract_id` 只对 `Policy` 引用的指标生成指纹，所以添加无关的
目录项不会使既有 `Problem` 标识失效。

`FormulationSpec` 是声明式且带版本的。其摘要覆盖模型表述和物化器的版本以及类型化配置。
编译后的求解器模型、可调用对象、跟踪状态和热启动有效载荷属于进程本地内容，绝不能
编码为模型表述选项。

需要感知存在性的标量字段使用 Proto `optional`：字段缺失时映射为文档所述的领域
默认值，而显式提供的 `false` 或零值则予以保留。适配器必须拒绝空白标识符、
`UNSPECIFIED` 枚举、缺失的必要测量值，以及超出 Python 领域模型取值范围的值。

字段名中包含其单位（`_ms`、`_mb`、`_mbps`、`_j`、`_watts`、`_ratio`）。除非某项
集成明确建立了挂钟时间基准，否则时间轴数值均相对于其所属的控制平面或模拟时间轴。

## 所有权

| 契约 | 主要负责人 | 必需评审方 |
| --- | --- | --- |
| 公共类型与工作流 | Core Infra / Interfaces | Scheduler 和 Runtime |
| 拓扑与运行时 | Agent Runtime & Simulation | Core Infra / Interfaces |
| 优化 | Scheduler & Optimizer | Core Infra / Interfaces |
| 性能剖析 | Agent Runtime & Simulation | Scheduler & Optimizer |
| 业务有效载荷 Proto | 对应工作负载团队 | Core Infra / Interfaces |

兼容性规则：

- 枚举的零值必须保持为 `UNSPECIFIED`；
- 绝不重复使用已经发布的字段编号或枚举值；
- 在 `mars.v1` 内添加兼容字段；
- 对破坏语义兼容性的变更使用新的包版本。

无需生成存根，即可从仓库根目录检查语法：

```sh
protoc -I . \
  --include_imports \
  --descriptor_set_out=/tmp/mars-v1.pb \
  interfaces/proto/mars/v1/*.proto
```
