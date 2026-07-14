import { useEffect, useMemo, useState } from 'react';
import { generateScene, health, simulate } from './api';
import type { Algorithm, BenchmarkScene, Difficulty, GenerateSceneRequest, ScenarioType, SimulationResponse, TaskCategory } from './types';
import { TASK_CATEGORIES } from './types';

const scenarioOptions: ScenarioType[] = ['warehouse', 'hospital', 'campus', 'factory', 'disaster', 'custom'];
const difficultyOptions: Difficulty[] = ['easy', 'medium', 'hard', 'stress'];
const algorithmOptions: Algorithm[] = ['rule_based', 'local_first', 'edge_first', 'greedy_cost', 'external'];

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
  };
  return map[key] ?? key;
}

function formatMetric(key: string, value: number) {
  if (key.includes('rate') || key.includes('ratio')) return pct(value);
  if (key.includes('latency') || key.includes('makespan')) return `${value.toFixed(1)} ms`;
  if (key.includes('energy')) return `${value.toFixed(1)} J`;
  if (key.includes('bandwidth')) return `${value.toFixed(1)} MB`;
  return `${value}`;
}

export default function App() {
  const [providerInfo, setProviderInfo] = useState<string>('checking backend...');
  const [scenarioType, setScenarioType] = useState<ScenarioType>('warehouse');
  const [customScene, setCustomScene] = useState('');
  const [robotCount, setRobotCount] = useState(4);
  const [edgeCount, setEdgeCount] = useState(1);
  const [difficulty, setDifficulty] = useState<Difficulty>('medium');
  const [useLlm, setUseLlm] = useState(true);
  const [seed, setSeed] = useState(7);
  const [taskCategories, setTaskCategories] = useState<TaskCategory[]>(['object_detection', 'path_planning', 'vla_inference']);
  const [algorithm, setAlgorithm] = useState<Algorithm>('greedy_cost');
  const [externalSchedulerUrl, setExternalSchedulerUrl] = useState('');
  const [scene, setScene] = useState<BenchmarkScene | null>(null);
  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<'overview' | 'tasks' | 'json' | 'logs'>('overview');

  useEffect(() => {
    health()
      .then((h) => setProviderInfo(`${h.provider}/${h.model} · LLM ${h.llm_configured ? 'configured' : 'fallback generator'}`))
      .catch((e) => setProviderInfo(`backend unavailable: ${e.message}`));
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
      const r = await simulate(scene, algorithm, externalSchedulerUrl);
      setResult(r);
      setTab('overview');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Capstone · Multi-Agent Robot Scheduling</p>
          <h1>多机器人模拟环境 / Benchmark 任务生成器</h1>
          <p className="subtitle">用 LLM 或离线规则生成真实场景，接入调度算法，评估鲁棒性、延迟、能耗、带宽和成功率。</p>
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
          {algorithm === 'external' && (
            <input value={externalSchedulerUrl} onChange={(e) => setExternalSchedulerUrl(e.target.value)} placeholder="http://localhost:9000/schedule" />
          )}
          <button className="secondary" onClick={onSimulate} disabled={!scene || loading}>运行仿真 / 评估</button>
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
                <button className={tab === 'tasks' ? 'active' : ''} onClick={() => setTab('tasks')}>任务</button>
                <button className={tab === 'json' ? 'active' : ''} onClick={() => setTab('json')}>JSON</button>
                <button className={tab === 'logs' ? 'active' : ''} onClick={() => setTab('logs')}>日志</button>
              </nav>

              {tab === 'overview' && <Overview scene={scene} result={result} />}
              {tab === 'tasks' && <Tasks scene={scene} result={result} />}
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
      <p>推荐先用 warehouse / hard / 4 robots / 1 edge PC / object_detection + path_planning + vla_inference，作为第一次汇报 Demo。</p>
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
        <div className="card"><span>Stressors</span><strong>{scene.stressors.length}</strong></div>
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
            <th>Source</th>
            <th>Priority</th>
            <th>Budget</th>
            <th>Model</th>
            <th>Fallback</th>
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
                <td>{t.source_robot_id}</td>
                <td>{t.priority}</td>
                <td>{t.latency_budget_ms.toFixed(0)} ms</td>
                <td>{t.model_requirement}</td>
                <td>{t.fallback_policy}</td>
                <td>{r?.target_node_id ?? '-'}</td>
                <td>{r ? `${r.total_latency_ms.toFixed(1)} ms` : '-'}</td>
                <td>{r ? <span className={r.success ? 'ok' : 'bad'}>{r.success ? 'success' : 'failed'}{r.deadline_missed ? ' · timeout' : ''}</span> : '-'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Logs({ result }: { result: SimulationResponse | null }) {
  if (!result) return <p className="muted">运行仿真后这里会显示调度日志。</p>;
  return <pre className="json-view">{result.logs.join('\n')}</pre>;
}
