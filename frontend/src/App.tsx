import { useEffect, useMemo, useState } from 'react';
import {
  bootstrapRuntime,
  generateScene,
  getRuntimeWorkflow,
  health,
  simulate,
  submitRuntimeWorkflow,
} from './api';
import type {
  Algorithm,
  BenchmarkScene,
  Difficulty,
  GenerateSceneRequest,
  RuntimeStatus,
  RuntimeWorkflowRun,
  ScenarioType,
  SimulationResponse,
  TaskCategory,
} from './types';
import { TASK_CATEGORIES } from './types';

const scenarioOptions: ScenarioType[] = ['warehouse', 'hospital', 'campus', 'factory', 'disaster', 'custom'];
const difficultyOptions: Difficulty[] = ['easy', 'medium', 'hard', 'stress'];
const algorithmOptions: Algorithm[] = ['dag_deadline', 'rule_based', 'local_first', 'edge_first', 'greedy_cost'];

const taskClassLabels: Record<string, string> = {
  local_safety: '端侧安全关键',
  realtime_offloadable: '可卸载实时推理',
  edge_heavy: '边缘优先重计算',
};

function pct(x: number) {
  return `${(x * 100).toFixed(1)}%`;
}

function metricLabel(key: string) {
  const map: Record<string, string> = {
    task_count: 'Tasks',
    success_rate: 'Success',
    deadline_miss_rate: 'Deadline Miss',
    avg_latency_ms: 'Avg Latency',
    p95_latency_ms: 'P95 Latency',
    p99_latency_ms: 'P99 Latency',
    avg_energy_j: 'Avg Energy',
    total_energy_j: 'Total Energy',
    bandwidth_mb: 'Bandwidth',
    makespan_ms: 'Makespan',
    edge_offload_ratio: 'Offload',
    safety_violation_count: 'Safety Violations',
    skipped_task_count: 'Skipped',
    workflow_success_rate: 'Workflow Success',
    critical_path_ms: 'Critical Path',
    dag_depth: 'DAG Depth',
  };
  return map[key] ?? key;
}

function formatMetric(key: string, value: number) {
  if (key.includes('rate') || key.includes('ratio')) return pct(value);
  if (key.includes('latency') || key.includes('makespan') || key.includes('critical_path')) return `${value.toFixed(1)} ms`;
  if (key.includes('energy')) return `${value.toFixed(1)} J`;
  if (key.includes('bandwidth')) return `${value.toFixed(1)} MB`;
  return `${value}`;
}

function runtimeDemoCompatibilityIssue(
  scene: BenchmarkScene | null,
  runtimeStatus: RuntimeStatus | null,
) {
  if (!scene) return null;
  if (!runtimeStatus) {
    return '本地 Agent 运行时尚未就绪；当前 Scene 仍可运行 benchmark 仿真。';
  }

  const robotCount = scene.nodes.filter((node) => node.kind === 'robot').length;
  const edgeCount = scene.nodes.filter((node) => node.kind === 'edge').length;
  const unsupportedCount = scene.nodes.length - robotCount - edgeCount;
  const requiredRobots = runtimeStatus.topology.orin_agents;
  const requiredEdges = runtimeStatus.topology.edge_agents;
  if (
    robotCount !== requiredRobots
    || edgeCount !== requiredEdges
    || unsupportedCount > 0
  ) {
    return `Agent 演示仅支持 ${requiredRobots} 个 Orin + ${requiredEdges} 个 Edge；当前 Scene 为 ${robotCount} 个 Orin + ${edgeCount} 个 Edge${unsupportedCount ? `，另有 ${unsupportedCount} 个不支持的节点` : ''}。仍可运行 benchmark 仿真。`;
  }
  return null;
}

export default function App() {
  const [providerInfo, setProviderInfo] = useState<string>('checking backend...');
  const [scenarioType, setScenarioType] = useState<ScenarioType>('warehouse');
  const [customScene, setCustomScene] = useState('');
  const [robotCount, setRobotCount] = useState(2);
  const [edgeCount, setEdgeCount] = useState(1);
  const [difficulty, setDifficulty] = useState<Difficulty>('medium');
  const [useLlm, setUseLlm] = useState(false);
  const [seed, setSeed] = useState(7);
  const [taskCategories, setTaskCategories] = useState<TaskCategory[]>([
    'localization',
    'environment_understanding',
    'object_detection',
    'semantic_segmentation',
    'local_planning',
    'obstacle_avoidance',
    'local_control',
    'local_llm_7b',
  ]);
  const [algorithm, setAlgorithm] = useState<Algorithm>('dag_deadline');
  const [scene, setScene] = useState<BenchmarkScene | null>(null);
  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus | null>(null);
  const [runtimeRun, setRuntimeRun] = useState<RuntimeWorkflowRun | null>(null);
  const [loading, setLoading] = useState(false);
  const [runtimeLoading, setRuntimeLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<'overview' | 'dag' | 'tasks' | 'runtime' | 'json' | 'logs'>('overview');

  useEffect(() => {
    health()
      .then((h) => setProviderInfo(
        `MARS ${h.mars_version} · local agent runtime · LLM ${h.llm_configured ? 'configured' : 'fallback'}`,
      ))
      .catch((e) => setProviderInfo(`backend unavailable: ${e.message}`));
    bootstrapRuntime()
      .then(setRuntimeStatus)
      .catch(() => setRuntimeStatus(null));
  }, []);

  const requestPayload: GenerateSceneRequest = useMemo(() => ({
    scenario_type: scenarioType,
    custom_scene: customScene || undefined,
    robot_count: robotCount,
    edge_count: edgeCount,
    task_categories: taskCategories,
    difficulty,
    seed,
    use_llm: useLlm,
  }), [scenarioType, customScene, robotCount, edgeCount, taskCategories, difficulty, seed, useLlm]);
  const runtimeDemoIssue = useMemo(
    () => runtimeDemoCompatibilityIssue(scene, runtimeStatus),
    [scene, runtimeStatus],
  );
  const runtimeDemoTopology = runtimeStatus
    ? `${runtimeStatus.topology.orin_agents} Orin + ${runtimeStatus.topology.edge_agents} Edge`
    : '本地 Agent';

  function toggleTaskCategory(cat: TaskCategory) {
    setTaskCategories((prev) => {
      if (prev.includes(cat)) return prev.filter((x) => x !== cat);
      return [...prev, cat];
    });
  }

  async function onGenerate() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      if (taskCategories.length === 0) throw new Error('请至少选择一种原子任务类别。');
      const s = await generateScene(requestPayload);
      setScene(s);
      setTab('overview');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onSimulate() {
    if (!scene) return;
    setLoading(true);
    setError(null);
    try {
      const r = await simulate(scene, algorithm);
      setResult(r);
      setTab('overview');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onRuntimeRun() {
    if (!scene || runtimeDemoIssue) return;
    setRuntimeLoading(true);
    setError(null);
    setRuntimeRun(null);
    setTab('runtime');
    try {
      const accepted = await submitRuntimeWorkflow(scene, algorithm, seed);
      let run: RuntimeWorkflowRun | null = null;
      for (let attempt = 0; attempt < 100; attempt += 1) {
        run = await getRuntimeWorkflow(accepted.run_id);
        setRuntimeRun(run);
        if (run.status === 'succeeded' || run.status === 'failed') break;
        await new Promise((resolve) => window.setTimeout(resolve, 80));
      }
      if (!run || (run.status !== 'succeeded' && run.status !== 'failed')) {
        throw new Error('本地运行时未在预期时间内返回结果。');
      }
      if (run.status === 'failed') {
        throw new Error(run.error || '本地 Agent 工作流运行失败。');
      }
      setRuntimeStatus(await bootstrapRuntime());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRuntimeLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Capstone · Multi-Agent Robot Scheduling</p>
          <h1>端—边 DAG 调度与机器人 Benchmark</h1>
          <p className="subtitle">统一 MARS 控制内核、DAG 工作流与三类任务约束，评估延迟、能耗、数据移动和工作流成功率。</p>
        </div>
        <div className="status-pill">{providerInfo}</div>
      </header>

      <main className="layout">
        <aside className="panel controls">
          <h2>生成控制</h2>
          <label>场景</label>
          <select value={scenarioType} onChange={(e) => setScenarioType(e.target.value as ScenarioType)}>
            {scenarioOptions.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>

          {scenarioType === 'custom' && (
            <textarea value={customScene} onChange={(e) => setCustomScene(e.target.value)} placeholder="描述自定义场景，例如：医院药房配送 + 网络拥塞 + 边缘服务器过载" />
          )}

          <div className="grid2">
            <div>
              <label>机器人数量</label>
              <input type="number" min={1} max={50} value={robotCount} onChange={(e) => setRobotCount(Number(e.target.value))} />
            </div>
            <div>
              <label>边缘 PC 数量</label>
              <input type="number" min={1} max={8} value={edgeCount} onChange={(e) => setEdgeCount(Number(e.target.value))} />
            </div>
          </div>

          <label>任务难度</label>
          <select value={difficulty} onChange={(e) => setDifficulty(e.target.value as Difficulty)}>
            {difficultyOptions.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>

          <label>原子任务类别</label>
          <div className="chips">
            {TASK_CATEGORIES.map((cat) => (
              <button
                type="button"
                key={cat}
                className={taskCategories.includes(cat) ? 'chip selected' : 'chip'}
                onClick={() => toggleTaskCategory(cat)}
              >
                {cat}
              </button>
            ))}
          </div>

          <div className="grid2">
            <label className="checkbox-line"><input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} /> 使用 LLM</label>
            <div>
              <label>Seed</label>
              <input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </div>
          </div>

          <button className="primary" onClick={onGenerate} disabled={loading}>{loading ? '处理中...' : '生成 Scene / Benchmark'}</button>

          <hr />
          <h2>算法测试</h2>
          <label>调度算法</label>
          <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value as Algorithm)}>
            {algorithmOptions.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
          <button className="secondary" onClick={onSimulate} disabled={!scene || loading}>运行仿真 / 评估</button>
          <button
            className="primary runtime-run"
            onClick={onRuntimeRun}
            disabled={!scene || Boolean(runtimeDemoIssue) || runtimeLoading}
          >
            {runtimeLoading ? 'Agent 运行中...' : `运行 ${runtimeDemoTopology} 演示`}
          </button>
          <p className={runtimeDemoIssue ? 'control-note warning' : 'control-note'}>
            {runtimeDemoIssue ?? '中央 Scheduler 动态分配任务，并注入一次可恢复失败来展示 retry。'}
          </p>
        </aside>

        <section className="panel workspace">
          {error && <div className="error">{error}</div>}
          {!scene && <EmptyState />}
          {scene && (
            <>
              <div className="scene-header">
                <div>
                  <h2>{scene.title}</h2>
                  <p>{scene.natural_language_description}</p>
                </div>
                <div className={`difficulty ${scene.difficulty}`}>{scene.difficulty}</div>
              </div>
              <nav className="tabs">
                <button className={tab === 'overview' ? 'active' : ''} onClick={() => setTab('overview')}>概览</button>
                <button className={tab === 'dag' ? 'active' : ''} onClick={() => setTab('dag')}>DAG</button>
                <button className={tab === 'tasks' ? 'active' : ''} onClick={() => setTab('tasks')}>任务</button>
                <button className={tab === 'runtime' ? 'active' : ''} onClick={() => setTab('runtime')}>Agent 运行</button>
                <button className={tab === 'json' ? 'active' : ''} onClick={() => setTab('json')}>JSON</button>
                <button className={tab === 'logs' ? 'active' : ''} onClick={() => setTab('logs')}>日志</button>
              </nav>

              {tab === 'overview' && <Overview scene={scene} result={result} />}
              {tab === 'dag' && <DagView scene={scene} result={result} runtimeRun={runtimeRun} />}
              {tab === 'tasks' && <Tasks scene={scene} result={result} />}
              {tab === 'runtime' && <RuntimeView status={runtimeStatus} run={runtimeRun} />}
              {tab === 'json' && <pre className="json-view">{JSON.stringify({ scene, result }, null, 2)}</pre>}
              {tab === 'logs' && <Logs result={result} />}
            </>
          )}
        </section>
      </main>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty">
      <h2>先生成一个 benchmark scene</h2>
      <p>推荐先用 warehouse / medium / 2 Orin / 1 edge，展示三类任务、定位 fan-out、动态分配和数据移动。</p>
    </div>
  );
}

function Overview({ scene, result }: { scene: BenchmarkScene; result: SimulationResponse | null }) {
  return (
    <div className="overview">
      <div className="cards">
        <div className="card"><span>Nodes</span><strong>{scene.nodes.length}</strong></div>
        <div className="card"><span>Robots</span><strong>{scene.nodes.filter((n) => n.kind === 'robot').length}</strong></div>
        <div className="card"><span>Tasks</span><strong>{scene.tasks.length}</strong></div>
        <div className="card"><span>Data edges</span><strong>{scene.data_edges.length}</strong></div>
      </div>

      {result && (
        <>
          <h3>仿真指标 · {result.algorithm}</h3>
          <div className="metrics-grid">
            {Object.entries(result.metrics).map(([key, value]) => (
              <div className="metric" key={key}>
                <span>{metricLabel(key)}</span>
                <strong>{formatMetric(key, value as number)}</strong>
              </div>
            ))}
          </div>
          <h3>三类任务汇总</h3>
          <div className="class-grid">
            {Object.entries(result.task_class_summary).map(([taskClass, summary]) => (
              <div className={`class-card ${taskClass}`} key={taskClass}>
                <span>{taskClassLabels[taskClass] ?? taskClass}</span>
                <strong>{summary.task_count}</strong>
                <small>success {pct(summary.success_rate)} · edge {pct(summary.edge_offload_ratio)} · avg {summary.avg_latency_ms.toFixed(1)} ms</small>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="split">
        <div>
          <h3>Stressors</h3>
          <ul>{scene.stressors.map((s) => <li key={s}>{s}</li>)}</ul>
        </div>
        <div>
          <h3>Success Criteria</h3>
          <ul>{scene.success_criteria.map((s) => <li key={s}>{s}</li>)}</ul>
        </div>
      </div>

      <h3>节点资源</h3>
      <div className="node-list">
        {scene.nodes.map((n) => {
          const r = scene.initial_resources.find((x) => x.node_id === n.id);
          return (
            <div className="node-card" key={n.id}>
              <strong>{n.display_name}</strong>
              <span>{n.kind} · CPU {n.cpu_capacity} · GPU {n.gpu_capacity} · {n.memory_gb}GB</span>
              {r && <span>util CPU {pct(r.cpu_util)} · GPU {pct(r.gpu_util)} · Temp {r.temperature_c.toFixed(1)}℃ · Net {r.network_latency_ms.toFixed(1)}ms</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Tasks({ scene, result }: { scene: BenchmarkScene; result: SimulationResponse | null }) {
  const resultMap = new Map(result?.task_results.map((r) => [r.task_id, r]) ?? []);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Task</th>
            <th>Class / DAG</th>
            <th>Source</th>
            <th>Priority</th>
            <th>Budget</th>
            <th>Model</th>
            <th>Target</th>
            <th>Latency</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {scene.tasks.map((t) => {
            const r = resultMap.get(t.id);
            return (
              <tr key={t.id}>
                <td><strong>{t.id}</strong><br /><span>{t.task_type}</span></td>
                <td><span className={`task-class ${t.task_class}`}>{taskClassLabels[t.task_class]}</span><span>stage {t.stage_index} · deps {t.dependencies.join(', ') || 'root'}</span></td>
                <td>{t.source_robot_id}</td>
                <td>{t.priority}</td>
                <td>{t.latency_budget_ms.toFixed(0)} ms</td>
                <td>{t.model_requirement}</td>
                <td>{r?.target_node_id ?? '-'}</td>
                <td>{r ? `${r.total_latency_ms.toFixed(1)} ms` : '-'}</td>
                <td>{r ? <span className={r.success ? 'ok' : 'bad'}>{r.state}{r.deadline_missed ? ' · deadline' : ''}</span> : '-'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function DagView({
  scene,
  result,
  runtimeRun,
}: {
  scene: BenchmarkScene;
  result: SimulationResponse | null;
  runtimeRun: RuntimeWorkflowRun | null;
}) {
  const runtimeResult = runtimeRun?.result ?? null;
  const levels = result?.dag.levels ?? runtimeResult?.workflow.levels ?? Object.fromEntries(scene.tasks.map((task) => [task.id, task.stage_index]));
  const maxLevel = Math.max(0, ...Object.values(levels));
  const critical = new Set(result?.workflow.critical_path ?? runtimeResult?.workflow.critical_path ?? []);
  const resultMap = new Map(result?.task_results.map((item) => [item.task_id, item]) ?? []);
  const runtimeMap = new Map(runtimeResult?.task_results.map((item) => [item.task_id, item]) ?? []);
  return (
    <div className="dag-view">
      <div className="dag-summary">
        <span>{scene.workflow_id}</span>
        <span>{result?.dag.valid === false ? 'invalid' : 'valid DAG'}</span>
        <span>{(result?.dag.edges.length ?? scene.tasks.reduce((sum, task) => sum + task.dependencies.length, 0))} edges</span>
        <span>failure: {scene.failure_policy}</span>
      </div>
      <div className="dag-stages">
        {Array.from({ length: maxLevel + 1 }, (_, level) => (
          <section className="dag-stage" key={level}>
            <div className="stage-label">Stage {level}</div>
            <div className="stage-nodes">
              {scene.tasks.filter((task) => levels[task.id] === level).map((task) => {
                const run = resultMap.get(task.id) ?? runtimeMap.get(task.id);
                return (
                  <article className={`dag-node ${task.task_class} ${critical.has(task.id) ? 'critical' : ''}`} key={task.id}>
                    <div><strong>{task.id}</strong><span>{run?.state ?? 'pending'}</span></div>
                    <p>{task.task_type}</p>
                    <small>{task.dependencies.length ? `← ${task.dependencies.join(', ')}` : 'root'} · {run?.target_node_id || task.source_robot_id}</small>
                  </article>
                );
              })}
            </div>
          </section>
        ))}
      </div>
      {scene.data_edges.length > 0 && (
        <>
          <h3>Typed data flow</h3>
          <div className="data-edge-list">
            {scene.data_edges.map((edge) => (
              <div className="data-edge" key={`${edge.producer_task}.${edge.producer_port}-${edge.consumer_task}.${edge.consumer_port}`}>
                <code>{edge.producer_task}.{edge.producer_port}</code>
                <span>→</span>
                <code>{edge.consumer_task}.{edge.consumer_port}</code>
                <small>{edge.message_type}</small>
              </div>
            ))}
          </div>
        </>
      )}
      <p className="muted">高亮边框表示关键路径；运行后节点会显示终态与实际执行位置。</p>
    </div>
  );
}

function Logs({ result }: { result: SimulationResponse | null }) {
  if (!result) return <p className="muted">运行仿真后这里会显示调度日志。</p>;
  return <pre className="json-view">{result.logs.join('\n')}</pre>;
}

function RuntimeView({ status, run }: { status: RuntimeStatus | null; run: RuntimeWorkflowRun | null }) {
  const result = run?.result ?? null;
  const agents = result?.agents ?? status?.agents ?? [];
  return (
    <div className="runtime-view">
      <div className="runtime-topology">
        <div className="topology-node scheduler-node">
          <span>Control plane</span>
          <strong>MARS Central Scheduler</strong>
          <small>{status?.status ?? 'starting'}</small>
        </div>
        <div className="topology-arrow">→</div>
        <div className="topology-agents">
          {agents.map((agent) => (
            <div className={`topology-node ${agent.kind}`} key={agent.agent_id}>
              <span>{agent.kind === 'robot' ? 'Simulated Orin Agent' : 'Simulated Edge Agent'}</span>
              <strong>{agent.agent_id}</strong>
              <small>{agent.registered && agent.online ? 'registered · online' : 'offline'} · heartbeat {agent.heartbeat_sequence}</small>
            </div>
          ))}
        </div>
      </div>

      {!run && <p className="runtime-hint">运行本地三 Agent 演示后，这里会展示 assignment、数据传输、失败重试、Artifact 和最终结果。</p>}
      {run && !result && <p className="runtime-hint">{run.workflow_id} · {run.status}</p>}

      {result && (
        <>
          <h3>Runtime result · {result.workflow.state}</h3>
          <div className="metrics-grid runtime-metrics">
            {Object.entries(result.metrics).map(([key, value]) => (
              <div className="metric" key={key}>
                <span>{runtimeMetricLabel(key)}</span>
                <strong>{runtimeMetricValue(key, value)}</strong>
              </div>
            ))}
          </div>

          <h3>Agent execution</h3>
          <div className="agent-runtime-grid">
            {result.agents.map((agent) => (
              <div className="agent-runtime-card" key={agent.agent_id}>
                <div><strong>{agent.agent_id}</strong><span className={agent.online ? 'ok' : 'bad'}>{agent.online ? 'online' : 'offline'}</span></div>
                <p>{agent.architecture} · max concurrency {agent.max_concurrency}</p>
                <small>completed {agent.completed_attempts} · failed attempts {agent.failed_attempts} · utilization {pct(agent.utilization)}</small>
              </div>
            ))}
          </div>

          <h3>Assignments and attempts</h3>
          <div className="table-wrap runtime-table">
            <table>
              <thead>
                <tr><th>Task</th><th>Class</th><th>Final placement</th><th>Attempts</th><th>Outputs</th><th>State</th></tr>
              </thead>
              <tbody>
                {result.task_results.map((task) => (
                  <tr key={task.task_id}>
                    <td><strong>{task.task_id}</strong><br /><span>{task.task_type}</span></td>
                    <td><span className={`task-class ${task.task_class}`}>{taskClassLabels[task.task_class]}</span></td>
                    <td>{task.target_node_id || '-'}<span>{task.mode || 'not assigned'}</span></td>
                    <td>
                      {task.attempts.length === 0 ? '-' : task.attempts.map((attempt) => (
                        <span className={`attempt-line ${attempt.state}`} key={attempt.attempt_id}>
                          #{attempt.attempt_no} {attempt.target_node_id} · {attempt.state}
                          {attempt.error_code ? ` · ${attempt.error_code}` : ''}
                        </span>
                      ))}
                    </td>
                    <td>{task.outputs.map((output) => `${output.producer_port} (${output.size_mb.toFixed(3)} MB)`).join(', ') || '-'}</td>
                    <td><span className={task.state === 'succeeded' ? 'ok' : 'bad'}>{task.state}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3>Control-plane events</h3>
          <div className="event-list">
            {result.events.map((event) => (
              <div className={`runtime-event ${event.event_type}`} key={event.sequence}>
                <code>#{event.sequence} · {event.time_ms.toFixed(1)} ms</code>
                <strong>{event.event_type}</strong>
                <span>{event.message}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function runtimeMetricLabel(key: string) {
  const labels: Record<string, string> = {
    task_count: 'Tasks',
    succeeded_task_count: 'Succeeded',
    failed_task_count: 'Failed',
    success_rate: 'Success',
    attempt_count: 'Attempts',
    retry_count: 'Retries',
    retry_success_count: 'Recovered',
    transferred_mb: 'Transferred',
    transfer_time_ms: 'Transfer time',
    total_energy_j: 'Energy',
    makespan_ms: 'Makespan',
    edge_offload_ratio: 'Edge offload',
    safety_violation_count: 'Safety violations',
    critical_path_ms: 'Critical path',
  };
  return labels[key] ?? key;
}

function runtimeMetricValue(key: string, value: number) {
  if (key.includes('rate') || key.includes('ratio')) return pct(value);
  if (key.endsWith('_ms')) return `${value.toFixed(1)} ms`;
  if (key.endsWith('_mb')) return `${value.toFixed(3)} MB`;
  if (key.endsWith('_j')) return `${value.toFixed(2)} J`;
  return `${value}`;
}
