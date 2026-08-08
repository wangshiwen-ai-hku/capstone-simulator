import { useEffect, useMemo, useState } from 'react';
import {
  generateScene,
  getRuntimeWorkflow,
  health,
  simulate,
  submitRuntimeWorkflow,
} from './api';
import { canonicalDag } from './dag';
import type {
  Algorithm,
  BenchmarkScene,
  Difficulty,
  GenerateSceneRequest,
  PlacementConstraintsSpec,
  RuntimeWorkflowRun,
  ScenarioType,
  SimulationResponse,
  TaskCategory,
  TaskClass,
} from './types';
import { TASK_CATEGORIES } from './types';

const scenarioOptions: ScenarioType[] = ['warehouse', 'hospital', 'campus', 'factory', 'disaster', 'custom'];
const difficultyOptions: Difficulty[] = ['easy', 'medium', 'hard', 'stress'];
const algorithmOptions: Algorithm[] = ['dag_deadline', 'rule_based', 'local_first', 'edge_first', 'greedy_cost', 'binary_offload'];
const MAX_ROBOTS = 50;
const MIN_EDGE_NODES = 0;
const MAX_EDGE_NODES = 8;
const MAX_SEED = 2_147_483_647;
const GRAPH_NODE_LIMIT = 120;
const TASK_PAGE_SIZE = 50;
const PLACEMENT_PAGE_SIZE = 24;
const EDGE_PAGE_SIZE = 50;
const EVENT_PAGE_SIZE = 100;
const LOG_PAGE_SIZE = 200;
const RUNTIME_POLL_TIMEOUT_MS = 60_000;
const RUNTIME_POLL_INTERVAL_MS = 500;

const generationSourceLabels: Record<BenchmarkScene['generation_source'], string> = {
  deterministic: 'Deterministic',
  llm: 'LLM',
  deterministic_fallback: 'Deterministic fallback',
};

const tabOptions = [
  ['overview', 'Overview'],
  ['dag', 'DAG'],
  ['tasks', 'Tasks'],
  ['runtime', 'Agent Runtime'],
  ['json', 'JSON'],
  ['logs', 'Logs'],
] as const;

type TabId = typeof tabOptions[number][0];

const taskClassLabels: Record<TaskClass, string> = {
  local_safety: 'Local safety reporting cohort',
  realtime_offloadable: 'Real-time offloadable reporting cohort',
  edge_heavy: 'Edge-heavy reporting cohort',
};

type ConstraintBadgeTone = 'pin' | 'safety' | 'allowed' | 'preferred' | 'capability' | 'property' | 'warning';

interface ConstraintBadge {
  label: string;
  tone: ConstraintBadgeTone;
}

function reportClassLabel(taskClass?: TaskClass | null) {
  return taskClass ? taskClassLabels[taskClass] : 'No reporting cohort';
}

function placementBadges(
  placement?: PlacementConstraintsSpec | null,
  limit = Number.POSITIVE_INFINITY,
) {
  if (!placement) {
    return [{ label: 'Placement constraints not declared', tone: 'warning' as const }];
  }

  const badges: ConstraintBadge[] = [];
  if (placement.pinned_node_id) badges.push({ label: `Pinned to ${placement.pinned_node_id}`, tone: 'pin' });
  if (placement.pin_to_source) badges.push({ label: 'Pinned to source', tone: 'pin' });
  if (placement.safety_required) badges.push({ label: 'Safety-capable node', tone: 'safety' });
  if (placement.allowed_node_kinds.length) {
    badges.push({ label: `Allowed: ${placement.allowed_node_kinds.join(' / ')}`, tone: 'allowed' });
  }
  badges.push({
    label: placement.allow_source_node ? 'Source allowed' : 'Source excluded',
    tone: placement.allow_source_node ? 'allowed' : 'property',
  });
  if (placement.preferred_node_kinds.length) {
    badges.push({ label: `Preferred: ${placement.preferred_node_kinds.join(' / ')}`, tone: 'preferred' });
  }
  placement.required_capabilities.forEach((capability) => {
    badges.push({ label: `Capability: ${capability}`, tone: 'capability' });
  });
  if (placement.allow_other_robots) badges.push({ label: 'Cross-robot placement allowed', tone: 'property' });
  if (placement.stateful) badges.push({ label: 'Stateful', tone: 'property' });
  if (!placement.idempotent) badges.push({ label: 'Non-idempotent', tone: 'property' });
  if (!placement.allow_fallback) badges.push({ label: 'Fallback disabled', tone: 'property' });
  if (placement.splittable) badges.push({ label: 'Splittable', tone: 'property' });
  if (placement.replicable) badges.push({ label: 'Replicable', tone: 'property' });
  if (!badges.length) badges.push({ label: 'General placement', tone: 'allowed' });

  if (badges.length <= limit) return badges;
  return [
    ...badges.slice(0, Math.max(0, limit - 1)),
    { label: `+${badges.length - Math.max(0, limit - 1)}`, tone: 'property' as const },
  ];
}

function PlacementBadges({
  placement,
  limit,
}: {
  placement?: PlacementConstraintsSpec | null;
  limit?: number;
}) {
  return (
    <div className="constraint-badges">
      {placementBadges(placement, limit).map((badge, index) => (
        <span className={`constraint-badge ${badge.tone}`} key={`${badge.label}-${index}`}>
          {badge.label}
        </span>
      ))}
    </div>
  );
}

function boundedInteger(value: string, minimum: number, maximum: number, fallback: number) {
  if (value.trim() === '') return minimum;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(maximum, Math.max(minimum, Math.trunc(parsed)));
}

function pageCount(itemCount: number, pageSize: number) {
  return Math.max(1, Math.ceil(itemCount / pageSize));
}

function safePage(page: number, itemCount: number, pageSize: number) {
  return Math.min(page, pageCount(itemCount, pageSize) - 1);
}

function pageItems<T>(items: T[], page: number, pageSize: number) {
  const resolvedPage = safePage(page, items.length, pageSize);
  return items.slice(resolvedPage * pageSize, (resolvedPage + 1) * pageSize);
}

function Pagination({
  page,
  itemCount,
  pageSize,
  onPageChange,
  label,
}: {
  page: number;
  itemCount: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  label: string;
}) {
  const pages = pageCount(itemCount, pageSize);
  const resolvedPage = safePage(page, itemCount, pageSize);
  if (pages <= 1) return null;
  return (
    <nav className="pagination" aria-label={`${label} pagination`}>
      <button
        type="button"
        onClick={() => onPageChange(resolvedPage - 1)}
        disabled={resolvedPage === 0}
      >
        Previous
      </button>
      <span aria-live="polite">
        Page {resolvedPage + 1} of {pages} | {itemCount} items
      </span>
      <button
        type="button"
        onClick={() => onPageChange(resolvedPage + 1)}
        disabled={resolvedPage >= pages - 1}
      >
        Next
      </button>
    </nav>
  );
}

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
    total_solver_time_ms: 'Total Solver Time',
    max_solver_time_ms: 'Max Solver Time',
    scheduling_epoch_count: 'Scheduling Epochs',
    expected_success_reward: 'Expected Success Reward',
    communication_time_ms: 'Communication Time',
    peak_cpu_utilization: 'Peak CPU Utilization',
    peak_gpu_utilization: 'Peak GPU Utilization',
    peak_memory_utilization: 'Peak Memory Utilization',
    maximum_resource_utilization: 'Umax',
    workflow_evaluation_objective: 'Evaluation Objective',
  };
  return map[key] ?? key;
}

function formatMetric(key: string, value: number) {
  if (key.includes('rate') || key.includes('ratio') || key.includes('utilization')) return pct(value);
  if (key.includes('latency') || key.includes('makespan') || key.includes('critical_path') || key.includes('time_ms')) return `${value.toFixed(1)} ms`;
  if (key.includes('energy')) return `${value.toFixed(1)} J`;
  if (key.includes('bandwidth')) return `${value.toFixed(1)} MB`;
  return `${value}`;
}

function runtimeCompatibilityIssue(
  scene: BenchmarkScene | null,
) {
  if (!scene) return null;
  const unsupportedCount = scene.nodes.filter((node) => node.kind === 'cloud').length;
  if (unsupportedCount > 0) return 'The in-process Agent runtime supports robot and edge nodes only.';
  return null;
}

export default function App() {
  const [providerInfo, setProviderInfo] = useState<string>('Connecting to the MARS API...');
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
  const [beta, setBeta] = useState(0.01);
  const [scene, setScene] = useState<BenchmarkScene | null>(null);
  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [runtimeRun, setRuntimeRun] = useState<RuntimeWorkflowRun | null>(null);
  const [loading, setLoading] = useState(false);
  const [runtimeLoading, setRuntimeLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>('overview');

  useEffect(() => {
    health()
      .then((h) => setProviderInfo(
        `MARS ${h.mars_version} | Agent Runtime | LLM ${h.llm_configured ? 'configured' : 'not configured'}`,
      ))
      .catch((e) => setProviderInfo(`MARS API unavailable: ${e.message}`));
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
  const runtimeIssue = useMemo(
    () => runtimeCompatibilityIssue(scene),
    [scene],
  );

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
    setRuntimeRun(null);
    try {
      if (taskCategories.length === 0) throw new Error('Select at least one task type.');
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
    if (!scene || runtimeIssue) return;
    setLoading(true);
    setError(null);
    setRuntimeRun(null);
    try {
      const r = await simulate(scene, algorithm, seed, beta);
      setResult(r);
      setTab('overview');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onRuntimeRun() {
    if (!scene || runtimeIssue) return;
    setRuntimeLoading(true);
    setError(null);
    setResult(null);
    setRuntimeRun(null);
    setTab('runtime');
    try {
      const accepted = await submitRuntimeWorkflow(scene, algorithm, seed, beta);
      let run: RuntimeWorkflowRun | null = null;
      const pollingStartedAt = Date.now();
      while (Date.now() - pollingStartedAt < RUNTIME_POLL_TIMEOUT_MS) {
        run = await getRuntimeWorkflow(accepted.run_id);
        setRuntimeRun(run);
        if (run.status === 'succeeded' || run.status === 'failed') break;
        await new Promise((resolve) => window.setTimeout(resolve, RUNTIME_POLL_INTERVAL_MS));
      }
      if (!run || (run.status !== 'succeeded' && run.status !== 'failed')) {
        throw new Error(
          `Run ${accepted.run_id} did not finish within 60 seconds. Automatic polling stopped; server-side execution may still be active.`,
        );
      }
      if (run.status === 'failed') {
        throw new Error(run.error || 'The in-process Agent workflow failed.');
      }
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
          <p className="eyebrow">MARS | Multi-Agent Robot Scheduling</p>
          <h1>Robot-Edge Workflow Scheduling</h1>
          <p className="subtitle">Build typed DAG workflows and inspect placement constraints, scheduling results, and Agent runtime state.</p>
        </div>
        <div className="status-pill" role="status" aria-live="polite">{providerInfo}</div>
      </header>

      <main className="layout">
        <aside className="panel controls">
          <h2>Workflow Generation</h2>
          <label htmlFor="scenario-type">Scenario</label>
          <select id="scenario-type" value={scenarioType} onChange={(e) => setScenarioType(e.target.value as ScenarioType)}>
            {scenarioOptions.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>

          {scenarioType === 'custom' && (
            <>
              <label htmlFor="custom-scene">Scenario Description</label>
              <textarea id="custom-scene" value={customScene} onChange={(e) => setCustomScene(e.target.value)} placeholder="Hospital pharmacy delivery with network congestion and edge-server overload" />
            </>
          )}

          <div className="grid2">
            <div>
              <label htmlFor="robot-count">Robot Count</label>
              <input
                id="robot-count"
                type="number"
                min={1}
                max={MAX_ROBOTS}
                value={robotCount}
                onChange={(e) => setRobotCount(
                  boundedInteger(e.target.value, 1, MAX_ROBOTS, robotCount),
                )}
              />
            </div>
            <div>
              <label htmlFor="edge-count">Edge PC Count</label>
              <input
                id="edge-count"
                type="number"
                min={MIN_EDGE_NODES}
                max={MAX_EDGE_NODES}
                value={edgeCount}
                onChange={(e) => setEdgeCount(
                  boundedInteger(e.target.value, MIN_EDGE_NODES, MAX_EDGE_NODES, edgeCount),
                )}
              />
            </div>
          </div>

          <label htmlFor="difficulty">Workload Difficulty</label>
          <select id="difficulty" value={difficulty} onChange={(e) => setDifficulty(e.target.value as Difficulty)}>
            {difficultyOptions.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>

          <fieldset className="control-fieldset">
            <legend>Task Types</legend>
            <div className="chips">
              {TASK_CATEGORIES.map((cat) => {
                const selected = taskCategories.includes(cat);
                return (
                  <button
                    type="button"
                    key={cat}
                    className={selected ? 'chip selected' : 'chip'}
                    aria-pressed={selected}
                    onClick={() => toggleTaskCategory(cat)}
                  >
                    {cat}
                  </button>
                );
              })}
            </div>
          </fieldset>

          <div className="grid2">
            <label className="checkbox-line" htmlFor="use-llm"><input id="use-llm" type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} /> Use LLM</label>
            <div>
              <label htmlFor="seed">Seed</label>
              <input
                id="seed"
                type="number"
                min={0}
                max={MAX_SEED}
                value={seed}
                onChange={(e) => setSeed(
                  boundedInteger(e.target.value, 0, MAX_SEED, seed),
                )}
              />
            </div>
          </div>

          <button className="primary" aria-busy={loading} onClick={onGenerate} disabled={loading || runtimeLoading}>{loading ? 'Processing...' : 'Generate Workflow'}</button>

          <hr />
          <h2>Scheduling and Execution</h2>
          <label htmlFor="algorithm">Scheduling Policy</label>
          <select id="algorithm" value={algorithm} onChange={(e) => setAlgorithm(e.target.value as Algorithm)}>
            {algorithmOptions.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
          {algorithm === 'binary_offload' && (
            <>
              <label htmlFor="binary-beta">Communication Weight (Beta)</label>
              <input id="binary-beta" type="number" min={0} step={0.0001} value={beta} onChange={(e) => setBeta(Math.max(0, Number(e.target.value) || 0))} />
            </>
          )}
          <button className="secondary" onClick={onSimulate} disabled={!scene || Boolean(runtimeIssue) || loading || runtimeLoading}>Run Scheduling Simulation</button>
          <button
            className="primary runtime-run"
            onClick={onRuntimeRun}
            disabled={!scene || Boolean(runtimeIssue) || runtimeLoading || loading}
            aria-busy={runtimeLoading}
            aria-describedby="runtime-note"
          >
            {runtimeLoading ? 'Agent Running...' : 'Submit to Agent Runtime'}
          </button>
          <p id="runtime-note" className={runtimeIssue ? 'control-note warning' : 'control-note'}>
            {runtimeIssue ?? 'The central scheduler assigns tasks and records data transfers, execution attempts, and retry events.'}
          </p>
        </aside>

        <section className="panel workspace">
          {error && <div className="error" role="alert" aria-live="assertive">{error}</div>}
          {!scene && <EmptyState />}
          {scene && (
            <>
              <div className="scene-header">
                <div>
                  <h2>{scene.title}</h2>
                  <p>{scene.natural_language_description}</p>
                  {scene.generation_note && (
                    <p className="generation-note">{scene.generation_note}</p>
                  )}
                </div>
                <div className="scene-badges">
                  <div className={`generation-source ${scene.generation_source}`}>
                    {generationSourceLabels[scene.generation_source]}
                  </div>
                  <div className={`difficulty ${scene.difficulty}`}>{scene.difficulty}</div>
                </div>
              </div>
              <nav className="tabs" aria-label="Workflow views" role="tablist">
                {tabOptions.map(([tabId, label]) => (
                  <button
                    type="button"
                    role="tab"
                    id={`tab-${tabId}`}
                    aria-controls={`panel-${tabId}`}
                    aria-selected={tab === tabId}
                    className={tab === tabId ? 'active' : ''}
                    onClick={() => setTab(tabId)}
                    key={tabId}
                  >
                    {label}
                  </button>
                ))}
              </nav>

              <div
                id={`panel-${tab}`}
                role="tabpanel"
                aria-labelledby={`tab-${tab}`}
              >
                {tab === 'overview' && <Overview scene={scene} result={result} />}
                {tab === 'dag' && <DagView scene={scene} result={result} runtimeRun={runtimeRun} />}
                {tab === 'tasks' && <Tasks scene={scene} result={result} />}
                {tab === 'runtime' && <RuntimeView scene={scene} run={runtimeRun} />}
                {tab === 'json' && <pre className="json-view">{JSON.stringify({ scene, result, runtimeRun }, null, 2)}</pre>}
                {tab === 'logs' && <Logs result={result} />}
              </div>
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
      <h2>Generate a workflow to inspect its scheduling structure</h2>
      <p>Select a topology scale and task types to view the DAG, placement constraints, and execution results.</p>
    </div>
  );
}

function Overview({ scene, result }: { scene: BenchmarkScene; result: SimulationResponse | null }) {
  const [placementPage, setPlacementPage] = useState(0);
  const resolvedPlacementPage = safePage(
    placementPage,
    scene.tasks.length,
    PLACEMENT_PAGE_SIZE,
  );
  const visiblePlacementTasks = pageItems(
    scene.tasks,
    resolvedPlacementPage,
    PLACEMENT_PAGE_SIZE,
  );
  return (
    <div className="overview">
      <div className="cards">
        <div className="card"><span>Compute Nodes</span><strong>{scene.nodes.length}</strong></div>
        <div className="card"><span>Robots</span><strong>{scene.nodes.filter((n) => n.kind === 'robot').length}</strong></div>
        <div className="card"><span>Tasks</span><strong>{scene.tasks.length}</strong></div>
        <div className="card"><span>Typed Data Edges</span><strong>{scene.data_edges.length}</strong></div>
      </div>

      <div className="section-heading">
        <div>
          <h3>Task Placement Constraints</h3>
          <p>Each task uses task_type to identify its work and independent constraint dimensions to declare valid execution locations and runtime properties.</p>
        </div>
      </div>
      <div className="placement-overview">
        {visiblePlacementTasks.map((task) => (
          <article className="placement-card" key={task.id}>
            <div>
              <strong>{task.task_type}</strong>
              <span>{task.id}</span>
            </div>
            <PlacementBadges placement={task.placement_constraints} limit={5} />
          </article>
        ))}
      </div>
      <Pagination
        page={resolvedPlacementPage}
        itemCount={scene.tasks.length}
        pageSize={PLACEMENT_PAGE_SIZE}
        onPageChange={setPlacementPage}
        label="Task placement constraints"
      />

      {result && (
        <>
          <h3>Scheduling Metrics | {result.algorithm}</h3>
          <div className="metrics-grid">
            {Object.entries(result.metrics).map(([key, value]) => (
              <div className="metric" key={key}>
                <span>{metricLabel(key)}</span>
                <strong>{formatMetric(key, value as number)}</strong>
              </div>
            ))}
          </div>
          <div className="section-heading">
            <div>
              <h3>Reporting Cohort Statistics</h3>
              <p>TaskClass aggregates results; PlacementConstraints determine placement.</p>
            </div>
          </div>
          <div className="class-grid">
            {Object.entries(result.task_class_summary).map(([taskClass, summary]) => (
              summary ? (
                <div className={`class-card ${taskClass}`} key={taskClass}>
                  <span>{reportClassLabel(taskClass as TaskClass)}</span>
                  <strong>{summary.task_count}</strong>
                  <small>success {pct(summary.success_rate)} | edge {pct(summary.edge_offload_ratio)} | avg {summary.avg_latency_ms.toFixed(1)} ms</small>
                </div>
              ) : null
            ))}
          </div>
        </>
      )}

      <div className="split">
        <div>
          <h3>Scenario Stressors</h3>
          <ul>{scene.stressors.map((s) => <li key={s}>{s}</li>)}</ul>
        </div>
        <div>
          <h3>Success Criteria</h3>
          <ul>{scene.success_criteria.map((s) => <li key={s}>{s}</li>)}</ul>
        </div>
      </div>

      <h3>Node Resources</h3>
      <div className="node-list">
        {scene.nodes.map((n) => {
          const r = scene.initial_resources.find((x) => x.node_id === n.id);
          return (
            <div className="node-card" key={n.id}>
              <strong>{n.display_name}</strong>
              <span>{n.kind} | CPU {n.cpu_capacity} | GPU {n.gpu_capacity} | {n.memory_gb}GB</span>
              {r && <span>util CPU {pct(r.cpu_util)} | GPU {pct(r.gpu_util)} | Temp {r.temperature_c.toFixed(1)} C | Net {r.network_latency_ms.toFixed(1)}ms</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Tasks({ scene, result }: { scene: BenchmarkScene; result: SimulationResponse | null }) {
  const [taskPage, setTaskPage] = useState(0);
  const resultMap = new Map(result?.task_results.map((r) => [r.task_id, r]) ?? []);
  const dag = canonicalDag(scene);
  const resolvedTaskPage = safePage(taskPage, scene.tasks.length, TASK_PAGE_SIZE);
  const visibleTasks = pageItems(scene.tasks, resolvedTaskPage, TASK_PAGE_SIZE);
  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Task Type</th>
              <th>Placement Constraints</th>
              <th>DAG Inputs</th>
              <th>Priority</th>
              <th>Budget</th>
              <th>Model</th>
              <th>Assignment</th>
              <th>Latency</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {visibleTasks.map((t) => {
              const r = resultMap.get(t.id);
              const parents = dag.parents[t.id] ?? [];
              return (
                <tr key={t.id}>
                  <td>
                    <strong>{t.task_type}</strong>
                    <span>{t.id}</span>
                  </td>
                  <td><PlacementBadges placement={t.placement_constraints} /></td>
                  <td>
                    level {dag.levels[t.id] ?? 0}
                    <span>{parents.length ? `depends on ${parents.join(', ')}` : 'root task'}</span>
                    <span>source {t.source_robot_id}</span>
                  </td>
                  <td>{t.priority}</td>
                  <td>{t.latency_budget_ms.toFixed(0)} ms</td>
                  <td>{t.model_requirement}</td>
                  <td>{r?.target_node_id ?? '-'}</td>
                  <td>{r ? `${r.total_latency_ms.toFixed(1)} ms` : '-'}</td>
                  <td>{r ? <span className={r.success ? 'ok' : 'bad'}>{r.state}{r.deadline_missed ? ' | deadline' : ''}</span> : '-'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Pagination
        page={resolvedTaskPage}
        itemCount={scene.tasks.length}
        pageSize={TASK_PAGE_SIZE}
        onPageChange={setTaskPage}
        label="Task table"
      />
    </>
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
  const [viewMode, setViewMode] = useState<'graph' | 'levels'>('graph');
  const [levelPage, setLevelPage] = useState(0);
  const [edgePage, setEdgePage] = useState(0);
  const runtimeResult = runtimeRun?.result ?? null;
  const dag = useMemo(() => canonicalDag(scene), [scene]);
  const levels = dag.levels;
  const maxLevel = Math.max(0, ...Object.values(levels));
  const graphAvailable = scene.tasks.length <= GRAPH_NODE_LIMIT;
  const effectiveViewMode = graphAvailable ? viewMode : 'levels';
  const critical = new Set(result?.workflow.critical_path ?? runtimeResult?.workflow.critical_path ?? []);
  const resultMap = new Map(result?.task_results.map((item) => [item.task_id, item]) ?? []);
  const runtimeMap = new Map(runtimeResult?.task_results.map((item) => [item.task_id, item]) ?? []);
  const nodeWidth = 248;
  const nodeHeight = 190;
  const levelGap = 112;
  const nodeGap = 28;
  const marginX = 48;
  const marginY = 44;
  const groupedTasks = Array.from({ length: maxLevel + 1 }, () => [] as typeof scene.tasks);
  scene.tasks.forEach((task) => {
    groupedTasks[levels[task.id] ?? 0]?.push(task);
  });
  const maxTasksInLevel = Math.max(1, ...groupedTasks.map((tasks) => tasks.length));
  const graphWidth = Math.max(
    960,
    marginX * 2 + ((maxLevel + 1) * nodeWidth) + (maxLevel * levelGap),
  );
  const graphHeight = Math.max(
    520,
    marginY * 2 + (maxTasksInLevel * nodeHeight) + ((maxTasksInLevel - 1) * nodeGap),
  );
  const positions = new Map<string, { x: number; y: number }>();
  if (graphAvailable) {
    groupedTasks.forEach((tasks, level) => {
      const levelHeight = (tasks.length * nodeHeight) + (Math.max(0, tasks.length - 1) * nodeGap);
      const startY = (graphHeight - levelHeight) / 2;
      tasks.forEach((task, index) => {
        positions.set(task.id, {
          x: marginX + level * (nodeWidth + levelGap),
          y: startY + index * (nodeHeight + nodeGap),
        });
      });
    });
  }

  const typedPairCounts = new Map<string, number>();
  dag.graphEdges.forEach((edge) => {
    if (edge.kind !== 'data') return;
    const key = `${edge.from}\u0000${edge.to}`;
    typedPairCounts.set(key, (typedPairCounts.get(key) ?? 0) + 1);
  });
  const typedPairSeen = new Map<string, number>();
  const taskById = new Map(scene.tasks.map((task) => [task.id, task]));
  const topologicalIds = new Set(dag.topologicalOrder);
  const orderedIds = [
    ...dag.topologicalOrder,
    ...scene.tasks
      .map((task) => task.id)
      .filter((taskId) => !topologicalIds.has(taskId)),
  ];
  const orderedTasks = orderedIds.flatMap((taskId) => {
    const task = taskById.get(taskId);
    return task ? [task] : [];
  });
  const resolvedLevelPage = safePage(levelPage, orderedTasks.length, TASK_PAGE_SIZE);
  const visibleLevelTasks = pageItems(orderedTasks, resolvedLevelPage, TASK_PAGE_SIZE);
  const visibleLevelGroups = new Map<number, typeof scene.tasks>();
  visibleLevelTasks.forEach((task) => {
    const level = levels[task.id] ?? 0;
    const tasks = visibleLevelGroups.get(level) ?? [];
    tasks.push(task);
    visibleLevelGroups.set(level, tasks);
  });
  const resolvedEdgePage = safePage(edgePage, scene.data_edges.length, EDGE_PAGE_SIZE);
  const visibleDataEdges = pageItems(scene.data_edges, resolvedEdgePage, EDGE_PAGE_SIZE);

  return (
    <div className="dag-view">
      <div className="dag-toolbar">
        <div className="dag-summary">
          <span>{scene.workflow_id}</span>
          <span className={dag.valid && result?.dag.valid !== false ? 'valid' : 'invalid'}>
            {dag.valid && result?.dag.valid !== false ? 'valid DAG' : 'invalid DAG'}
          </span>
          <span>{dag.dependencyEdges.length} task.dependencies</span>
          <span>{scene.data_edges.length} typed DataEdges</span>
          <span>failure: {scene.failure_policy}</span>
        </div>
        <div className="view-switch" role="group" aria-label="DAG view mode">
          <button
            type="button"
            className={effectiveViewMode === 'graph' ? 'active' : ''}
            aria-pressed={effectiveViewMode === 'graph'}
            disabled={!graphAvailable}
            title={graphAvailable ? undefined : `Graph view supports up to ${GRAPH_NODE_LIMIT} task nodes`}
            onClick={() => setViewMode('graph')}
          >
            Graph
          </button>
          <button
            type="button"
            className={effectiveViewMode === 'levels' ? 'active' : ''}
            aria-pressed={effectiveViewMode === 'levels'}
            onClick={() => setViewMode('levels')}
          >
            Levels
          </button>
        </div>
      </div>

      {!graphAvailable && (
        <p className="scale-notice" role="status">
          This workflow contains {scene.tasks.length} tasks. Graph view supports up to {GRAPH_NODE_LIMIT} nodes; Levels view paginates the complete task set.
        </p>
      )}

      <div className="dag-legend">
        <span><i className="legend-line dependency" />Dependency without a data contract</span>
        <span><i className="legend-line data" />Typed DataEdge</span>
        <span><i className="legend-node critical" />Critical path</span>
        <span><i className="legend-node assigned" />Assigned / executed</span>
      </div>

      {effectiveViewMode === 'graph' ? (
        <div className="dag-graph-scroll">
          <svg
            className="dag-graph"
            width={graphWidth}
            height={graphHeight}
            viewBox={`0 0 ${graphWidth} ${graphHeight}`}
            role="img"
            aria-label={`${scene.workflow_id} directed acyclic graph`}
          >
            <title>{scene.workflow_id} directed acyclic graph</title>
            <defs>
              <marker id="arrow-dependency" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" className="dependency-arrow" />
              </marker>
              <marker id="arrow-data" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" className="data-arrow" />
              </marker>
            </defs>
            {groupedTasks.map((_, level) => {
              const x = marginX + level * (nodeWidth + levelGap) - 18;
              return (
                <g className="level-guide" key={level}>
                  <text x={x + 18} y={24}>LEVEL {level}</text>
                  <line x1={x} x2={x} y1={34} y2={graphHeight - 24} />
                </g>
              );
            })}
            <g className="graph-edges">
              {dag.graphEdges.map((edge) => {
                const source = positions.get(edge.from);
                const target = positions.get(edge.to);
                if (!source || !target) return null;
                const pairKey = `${edge.from}\u0000${edge.to}`;
                const pairCount = edge.kind === 'data' ? (typedPairCounts.get(pairKey) ?? 1) : 1;
                const pairIndex = edge.kind === 'data' ? (typedPairSeen.get(pairKey) ?? 0) : 0;
                if (edge.kind === 'data') typedPairSeen.set(pairKey, pairIndex + 1);
                const offset = edge.kind === 'data' ? (pairIndex - ((pairCount - 1) / 2)) * 22 : 0;
                const x1 = source.x + nodeWidth;
                const y1 = source.y + nodeHeight / 2;
                const x2 = target.x;
                const y2 = target.y + nodeHeight / 2;
                const curve = Math.max(48, (x2 - x1) * 0.42);
                const path = `M ${x1} ${y1} C ${x1 + curve} ${y1 + offset}, ${x2 - curve} ${y2 + offset}, ${x2} ${y2}`;
                const labelX = (x1 + x2) / 2;
                const labelY = (y1 + y2) / 2 + offset - 8;
                const label = edge.label ?? '';
                const labelWidth = Math.min(170, Math.max(72, label.length * 7.2 + 18));
                return (
                  <g className={`graph-edge ${edge.kind}`} key={edge.id}>
                    <path d={path} markerEnd={`url(#arrow-${edge.kind})`} />
                    {edge.kind === 'data' && (
                      <g className="edge-label">
                        <title>{label}</title>
                        <rect x={labelX - labelWidth / 2} y={labelY - 11} width={labelWidth} height={20} rx={8} />
                        <text x={labelX} y={labelY + 3}>{label.length > 22 ? `${label.slice(0, 20)}...` : label}</text>
                      </g>
                    )}
                  </g>
                );
              })}
            </g>
            <g className="graph-nodes">
              {scene.tasks.map((task) => {
                const position = positions.get(task.id);
                if (!position) return null;
                const run = runtimeMap.get(task.id) ?? resultMap.get(task.id);
                const state = run?.state ?? 'not-run';
                const stateClass = state.toLowerCase().replace(/[^a-z0-9_-]/g, '-');
                return (
                  <foreignObject
                    x={position.x}
                    y={position.y}
                    width={nodeWidth}
                    height={nodeHeight}
                    key={task.id}
                  >
                    <article className={`graph-task-node status-${stateClass} ${critical.has(task.id) ? 'critical' : ''}`}>
                      <div className="graph-node-heading">
                        <span>{task.id}</span>
                        <strong>{state === 'not-run' ? 'Not run' : state}</strong>
                      </div>
                      <h4>{task.task_type}</h4>
                      <div className="graph-assignment">
                        <span>assignment</span>
                        <strong>{run?.target_node_id || 'Unassigned'}</strong>
                      </div>
                      <PlacementBadges placement={task.placement_constraints} limit={4} />
                      <div className="graph-node-footer">
                        <span>source {task.source_robot_id}</span>
                        <span>priority {task.priority}</span>
                      </div>
                    </article>
                  </foreignObject>
                );
              })}
            </g>
          </svg>
        </div>
      ) : (
        <>
          <div className="dag-stages">
            {[...visibleLevelGroups.entries()]
              .sort(([left], [right]) => left - right)
              .map(([level, tasks]) => (
                <section className="dag-stage" key={level}>
                  <div className="stage-label">Level {level}</div>
                  <div className="stage-nodes">
                    {tasks.map((task) => {
                      const run = runtimeMap.get(task.id) ?? resultMap.get(task.id);
                      const parents = dag.parents[task.id] ?? [];
                      return (
                        <article className={`dag-node ${critical.has(task.id) ? 'critical' : ''} ${run ? 'assigned' : ''}`} key={task.id}>
                          <div><strong>{task.task_type}</strong><span>{run?.state ?? 'Not run'}</span></div>
                          <p>{task.id}</p>
                          <PlacementBadges placement={task.placement_constraints} limit={4} />
                          <small>{parents.length ? `depends on ${parents.join(', ')}` : 'root'} | {run?.target_node_id || 'Unassigned'}</small>
                        </article>
                      );
                    })}
                  </div>
                </section>
              ))}
          </div>
          <Pagination
            page={resolvedLevelPage}
            itemCount={orderedTasks.length}
            pageSize={TASK_PAGE_SIZE}
            onPageChange={setLevelPage}
            label="DAG level tasks"
          />
        </>
      )}

      {scene.data_edges.length > 0 && (
        <section className="data-contracts">
          <div className="section-heading">
            <div>
              <h3>DataEdge Contracts</h3>
              <p>Ports and message_type define the data interface between producers and consumers.</p>
            </div>
          </div>
          <div className="data-edge-list">
            {visibleDataEdges.map((edge, index) => (
              <div className="data-edge" key={`${edge.producer_task}.${edge.producer_port}-${edge.consumer_task}.${edge.consumer_port}-${index}`}>
                <code>{edge.producer_task}.{edge.producer_port}</code>
                <span>-&gt;</span>
                <code>{edge.consumer_task}.{edge.consumer_port}</code>
                <small>{edge.message_type}</small>
              </div>
            ))}
          </div>
          <Pagination
            page={resolvedEdgePage}
            itemCount={scene.data_edges.length}
            pageSize={EDGE_PAGE_SIZE}
            onPageChange={setEdgePage}
            label="DataEdge contracts"
          />
        </section>
      )}
    </div>
  );
}

function Logs({ result }: { result: SimulationResponse | null }) {
  const [logPage, setLogPage] = useState(0);
  if (!result) return <p className="muted">Scheduling logs appear here after a simulation run.</p>;
  const resolvedLogPage = safePage(logPage, result.logs.length, LOG_PAGE_SIZE);
  const visibleLogs = pageItems(result.logs, resolvedLogPage, LOG_PAGE_SIZE);
  return (
    <>
      <pre className="json-view">{visibleLogs.join('\n')}</pre>
      <Pagination
        page={resolvedLogPage}
        itemCount={result.logs.length}
        pageSize={LOG_PAGE_SIZE}
        onPageChange={setLogPage}
        label="Scheduling logs"
      />
    </>
  );
}

function RuntimeView({
  scene,
  run,
}: {
  scene: BenchmarkScene;
  run: RuntimeWorkflowRun | null;
}) {
  const [taskPage, setTaskPage] = useState(0);
  const [eventPage, setEventPage] = useState(0);
  const result = run?.result ?? null;
  const sceneTaskMap = new Map(scene.tasks.map((task) => [task.id, task]));
  const runtimeTasks = result?.task_results ?? [];
  const resolvedTaskPage = safePage(taskPage, runtimeTasks.length, TASK_PAGE_SIZE);
  const visibleRuntimeTasks = pageItems(runtimeTasks, resolvedTaskPage, TASK_PAGE_SIZE);
  const runtimeEvents = result?.events ?? [];
  const resolvedEventPage = safePage(eventPage, runtimeEvents.length, EVENT_PAGE_SIZE);
  const visibleRuntimeEvents = pageItems(runtimeEvents, resolvedEventPage, EVENT_PAGE_SIZE);
  return (
    <div className="runtime-view">
      <div className="runtime-topology">
        <div className="topology-node scheduler-node">
          <span>Control plane</span>
          <strong>MARS Central Scheduler</strong>
          <small>{result?.workflow.state ?? run?.status ?? 'scene declared'}</small>
        </div>
        <div className="topology-arrow">-&gt;</div>
        <div className="topology-agents">
          {result
            ? result.agents.map((agent) => (
              <div className={`topology-node ${agent.kind}`} key={agent.agent_id}>
                <span>{agent.kind === 'robot' ? 'Robot Agent' : 'Edge Agent'}</span>
                <strong>{agent.agent_id}</strong>
                <small>{agent.registered && agent.online ? 'registered | online' : 'offline'} | heartbeat {agent.heartbeat_sequence}</small>
              </div>
            ))
            : scene.nodes
              .filter((node) => node.kind === 'robot' || node.kind === 'edge')
              .map((node) => (
                <div className={`topology-node ${node.kind}`} key={node.id}>
                  <span>{node.kind === 'robot' ? 'Robot Agent' : 'Edge Agent'}</span>
                  <strong>{node.id}</strong>
                  <small>{node.architecture} | scene declared | max concurrency {node.max_concurrency}</small>
                </div>
              ))}
        </div>
      </div>

      {!run && <p className="runtime-hint">Submit a workflow to inspect assignments, data transfers, execution attempts, Artifacts, and runtime results.</p>}
      {run && !result && <p className="runtime-hint" role="status" aria-live="polite">{run.workflow_id} | {run.status}</p>}

      {result && (
        <>
          <h3>Runtime Result | {result.workflow.state}</h3>
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
                <p>{agent.architecture} | max concurrency {agent.max_concurrency}</p>
                <small>completed {agent.completed_attempts} | failed attempts {agent.failed_attempts} | utilization {pct(agent.utilization)}</small>
              </div>
            ))}
          </div>

          <h3>Assignments and attempts</h3>
          <div className="table-wrap runtime-table">
            <table>
              <thead>
                <tr><th>Task Type</th><th>Placement Constraints</th><th>Final Placement</th><th>Attempts</th><th>Outputs</th><th>State</th></tr>
              </thead>
              <tbody>
                {visibleRuntimeTasks.map((task) => {
                  const taskSpec = sceneTaskMap.get(task.task_id);
                  return (
                    <tr key={task.task_id}>
                      <td>
                        <strong>{task.task_type}</strong>
                        <span>{task.task_id}</span>
                      </td>
                      <td><PlacementBadges placement={taskSpec?.placement_constraints} /></td>
                      <td>{task.target_node_id || '-'}<span>{task.mode || 'not assigned'}</span></td>
                      <td>
                        {task.attempts.length === 0 ? '-' : task.attempts.map((attempt) => (
                          <span className={`attempt-line ${attempt.state}`} key={attempt.attempt_id}>
                            #{attempt.attempt_no} {attempt.target_node_id} | {attempt.state}
                            {attempt.error_code ? ` | ${attempt.error_code}` : ''}
                          </span>
                        ))}
                      </td>
                      <td>{task.outputs.map((output) => `${output.producer_port} (${output.size_mb.toFixed(3)} MB)`).join(', ') || '-'}</td>
                      <td><span className={task.state === 'succeeded' ? 'ok' : 'bad'}>{task.state}</span></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <Pagination
            page={resolvedTaskPage}
            itemCount={runtimeTasks.length}
            pageSize={TASK_PAGE_SIZE}
            onPageChange={setTaskPage}
            label="Runtime assignments"
          />

          <h3>Control-plane events</h3>
          <div className="event-list">
            {visibleRuntimeEvents.map((event) => (
              <div className={`runtime-event ${event.event_type}`} key={event.sequence}>
                <code>#{event.sequence} | {event.time_ms.toFixed(1)} ms</code>
                <strong>{event.event_type}</strong>
                <span>{event.message}</span>
              </div>
            ))}
          </div>
          <Pagination
            page={resolvedEventPage}
            itemCount={runtimeEvents.length}
            pageSize={EVENT_PAGE_SIZE}
            onPageChange={setEventPage}
            label="Control-plane events"
          />
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
