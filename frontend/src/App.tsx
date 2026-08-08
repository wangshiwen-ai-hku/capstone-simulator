import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react';
import {
  Box,
  ChevronLeft,
  ChevronRight,
  CircleStop,
  Cpu,
  Gauge,
  Pause,
  Play,
  RefreshCcw,
  RotateCcw,
  Server,
  SlidersHorizontal,
  Workflow,
  Zap,
} from 'lucide-react';
import {
  generateScene,
  getRuntimeWorkflow,
  health,
  submitRuntimeWorkflow,
} from './api';
import { canonicalDag } from './dag';
import type {
  Algorithm,
  BenchmarkScene,
  Difficulty,
  GenerateSceneRequest,
  RuntimeTaskResult,
  RuntimeWorkflowRun,
  ScenarioType,
  TaskCategory,
  Workload,
} from './types';
import { TASK_CATEGORIES } from './types';

const SCENARIOS: Array<{ value: ScenarioType; label: string }> = [
  { value: 'warehouse', label: 'Warehouse' },
  { value: 'hospital', label: 'Hospital' },
  { value: 'campus', label: 'Campus' },
  { value: 'factory', label: 'Factory' },
  { value: 'disaster', label: 'Disaster response' },
  { value: 'custom', label: 'Custom scene' },
];

const DIFFICULTIES: Difficulty[] = ['easy', 'medium', 'hard', 'stress'];
const ALGORITHMS: Array<{ value: Algorithm; label: string }> = [
  { value: 'dag_deadline', label: 'DAG deadline' },
  { value: 'rule_based', label: 'Rule based' },
  { value: 'local_first', label: 'Local first' },
  { value: 'edge_first', label: 'Edge first' },
  { value: 'greedy_cost', label: 'Greedy cost' },
];

const HARDWARE = {
  orin_nano: {
    label: 'Orin Nano',
    architecture: 'jetson-orin-nano',
    cpu: 1.2,
    gpu: 1.1,
    memory: 8,
  },
  orin_nx: {
    label: 'Orin NX',
    architecture: 'jetson-orin-nx',
    cpu: 2.2,
    gpu: 2.6,
    memory: 16,
  },
  orin_agx: {
    label: 'AGX Orin',
    architecture: 'jetson-agx-orin',
    cpu: 3.4,
    gpu: 4.2,
    memory: 32,
  },
} as const;

type HardwareId = keyof typeof HARDWARE;
type PlaybackState =
  | 'idle'
  | 'submitting'
  | 'running'
  | 'paused'
  | 'complete'
  | 'failed'
  | 'error';
type FlowKind = 'central' | 'task' | 'agent';

interface LlmStatus {
  configured: boolean;
  provider: string;
  model: string;
}

interface TaskPlayback {
  id: string;
  label: string;
  target: string;
  state: string;
  progress: number;
  start: number;
  finish: number;
}

interface FlowData extends Record<string, unknown> {
  kind: FlowKind;
  label: string;
  subtitle?: string;
  task?: Workload;
  queue?: TaskPlayback[];
  tasks?: TaskPlayback[];
  progress?: number;
  state?: string;
  hardwareKind?: 'robot' | 'edge' | 'cloud';
  architecture?: string;
  online?: boolean;
  utilization?: number;
  activeSlots?: number;
  maxConcurrency?: number;
}

type FlowNode = Node<FlowData>;

const NODE_TYPES = {
  central: CentralNode,
  task: TaskNode,
  agent: AgentNode,
};

const POLL_TIMEOUT_MS = 60_000;
const POLL_INTERVAL_MS = 450;
const PLAYBACK_DURATION_MS = 9_000;
const EMPTY_RUNTIME_TASKS: RuntimeTaskResult[] = [];
const TERMINAL_TASK_STATES = new Set([
  'succeeded',
  'failed',
  'timeout',
  'skipped',
  'dropped',
]);

function clamp(value: number, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function boundedInteger(value: string, min: number, max: number, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(max, Math.max(min, Math.trunc(parsed))) : fallback;
}

function taskLabel(category: string) {
  return category.replace(/_/g, ' ');
}

function taskTone(state?: string) {
  if (state === 'running') return 'running';
  if (state === 'succeeded') return 'succeeded';
  if (state && TERMINAL_TASK_STATES.has(state)) return 'failed';
  return 'queued';
}

export function slotUtilization(activeSlots: number, maxConcurrency: number) {
  return clamp(activeSlots / Math.max(1, maxConcurrency));
}

function agentPlaybackLoad(tasks: TaskPlayback[], maxConcurrency: number) {
  const activeSlots = tasks.filter((task) => task.state === 'running').length;
  return {
    activeSlots,
    utilization: slotUtilization(activeSlots, maxConcurrency),
  };
}

export function taskPlayback(
  task: Workload,
  runtimeTask: RuntimeTaskResult | undefined,
  playhead: number,
  makespan: number,
): TaskPlayback {
  if (!runtimeTask) {
    return {
      id: task.id,
      label: task.name || taskLabel(task.task_type),
      target: '',
      state: 'queued',
      progress: 0,
      start: task.arrival_time_ms,
      finish: task.arrival_time_ms,
    };
  }
  const attempts = runtimeTask?.attempts ?? [];
  if (attempts.length === 0 && TERMINAL_TASK_STATES.has(runtimeTask.state)) {
    const finish = Math.max(0, makespan);
    const finished = playhead >= finish;
    return {
      id: task.id,
      label: task.name || taskLabel(task.task_type),
      target: runtimeTask.target_node_id ?? '',
      state: finished ? runtimeTask.state : 'queued',
      progress: finished ? 1 : 0,
      start: finish,
      finish,
    };
  }
  const start = attempts.length
    ? Math.min(...attempts.map((attempt) => attempt.start_time_ms))
    : task.arrival_time_ms;
  const finish = attempts.length
    ? Math.max(...attempts.map((attempt) => attempt.finish_time_ms))
    : start;
  const hasStarted = playhead >= start;
  const hasFinished = playhead >= finish && finish >= start;
  const progress = !hasStarted
    ? 0
    : finish <= start
      ? 1
      : clamp((playhead - start) / (finish - start));
  let state = 'queued';
  if (hasStarted && !hasFinished) state = 'running';
  if (hasFinished) state = runtimeTask?.state ?? 'succeeded';
  return {
    id: task.id,
    label: task.name || taskLabel(task.task_type),
    target: runtimeTask?.target_node_id ?? '',
    state,
    progress,
    start,
    finish,
  };
}

function CentralNode({ data }: NodeProps<FlowNode>) {
  const queue = data.queue ?? [];
  const completed = queue.filter((task) => TERMINAL_TASK_STATES.has(task.state)).length;
  return (
    <section className="flow-node central-node">
      <Handle type="source" position={Position.Right} className="flow-handle" />
      <div className="node-titlebar central-titlebar">
        <span className="node-icon"><Workflow size={16} /></span>
        <div>
          <strong>{data.label}</strong>
          <small>{data.subtitle}</small>
        </div>
        <span className={`state-dot ${data.state ?? 'idle'}`} />
      </div>
      <div className="central-body">
        <div className="queue-meta">
          <span>Dispatch queue</span>
          <strong>{completed}/{queue.length}</strong>
        </div>
        <div className="queue-track" aria-label="Central task queue">
          {queue.map((task) => (
            <span
              key={task.id}
              className={`queue-segment ${taskTone(task.state)}`}
              style={{ flexGrow: Math.max(1, task.finish - task.start) }}
              title={`${task.label}: ${task.state}`}
            />
          ))}
        </div>
        <div className="queue-list">
          {queue.slice(0, 8).map((task, index) => (
            <div className="queue-row" key={task.id}>
              <span>{String(index + 1).padStart(2, '0')}</span>
              <p>{task.label}</p>
              <small>{task.target || 'waiting'}</small>
              <i className={taskTone(task.state)} />
            </div>
          ))}
          {queue.length > 8 && <div className="queue-more">+{queue.length - 8} queued tasks</div>}
        </div>
      </div>
    </section>
  );
}

function TaskNode({ data, selected }: NodeProps<FlowNode>) {
  const task = data.task;
  const tone = taskTone(data.state);
  return (
    <section className={`flow-node task-node ${tone} ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Left} className="flow-handle" />
      <Handle type="source" position={Position.Right} className="flow-handle" />
      <div className="node-titlebar">
        <span className="node-icon"><Box size={15} /></span>
        <strong className="node-name">{data.label}</strong>
        <span className={`task-status ${tone}`}>{tone}</span>
      </div>
      <div className="task-node-body">
        <div className="task-type">{taskLabel(task?.task_type ?? '')}</div>
        <div className="task-meta">
          <span>P{task?.priority ?? 0}</span>
          <span>{task?.compute_demand.toFixed(1)} CPU</span>
          <span>{task?.gpu_demand.toFixed(1)} GPU</span>
        </div>
        <div className="progress-label">
          <span>{data.subtitle || 'Unassigned'}</span>
          <strong>{Math.round((data.progress ?? 0) * 100)}%</strong>
        </div>
        <div className="task-progress">
          <span style={{ width: `${(data.progress ?? 0) * 100}%` }} />
        </div>
      </div>
    </section>
  );
}

function AgentNode({ data }: NodeProps<FlowNode>) {
  const tasks = data.tasks ?? [];
  const activeSlots = data.activeSlots ?? 0;
  const maxConcurrency = data.maxConcurrency ?? 1;
  const icon = data.hardwareKind === 'edge' ? <Server size={16} /> : <Cpu size={16} />;
  return (
    <section className={`flow-node agent-node ${data.hardwareKind ?? 'robot'}`}>
      <Handle type="target" position={Position.Left} className="flow-handle" />
      <div className="node-titlebar">
        <span className="node-icon">{icon}</span>
        <div>
          <strong>{data.label}</strong>
          <small>{data.architecture}</small>
        </div>
        <span className={data.online ? 'online-mark' : 'offline-mark'}>
          {data.online ? 'online' : 'offline'}
        </span>
      </div>
      <div className="agent-body">
        <div className="agent-util">
          <span>Active slots</span>
          <strong>{activeSlots}/{maxConcurrency}</strong>
        </div>
        <div className="agent-util-track">
          <span style={{ width: `${(data.utilization ?? 0) * 100}%` }} />
        </div>
        <div className="agent-tasks">
          {tasks.length === 0 && <p className="agent-empty">No assigned tasks</p>}
          {tasks.slice(0, 7).map((task) => (
            <div className="agent-task" key={task.id}>
              <div><span>{task.label}</span><small>{Math.round(task.progress * 100)}%</small></div>
              <div className="agent-task-track"><i className={taskTone(task.state)} style={{ width: `${task.progress * 100}%` }} /></div>
            </div>
          ))}
          {tasks.length > 7 && <p className="agent-empty">+{tasks.length - 7} more</p>}
        </div>
      </div>
    </section>
  );
}

function initialGraph(
  scene: BenchmarkScene,
  playback: TaskPlayback[],
): { nodes: FlowNode[]; edges: Edge[] } {
  const dag = canonicalDag(scene);
  const levelCounts = new Map<number, number>();
  const maxLevel = Math.max(0, ...Object.values(dag.levels));
  const nodes: FlowNode[] = [
    {
      id: 'central-scheduler',
      type: 'central',
      position: { x: 40, y: 140 },
      draggable: true,
      data: {
        kind: 'central',
        label: 'Central Scheduler',
        subtitle: 'MARS control plane',
        queue: playback,
        state: 'idle',
      },
    },
  ];

  scene.tasks.forEach((task) => {
    const level = dag.levels[task.id] ?? 0;
    const row = levelCounts.get(level) ?? 0;
    levelCounts.set(level, row + 1);
    const runtime = playback.find((item) => item.id === task.id);
    nodes.push({
      id: `task:${task.id}`,
      type: 'task',
      position: {
        x: 430 + level * 285,
        y: 54 + row * 168,
      },
      data: {
        kind: 'task',
        label: task.name || taskLabel(task.task_type),
        subtitle: runtime?.target || task.source_robot_id,
        task,
        progress: runtime?.progress ?? 0,
        state: runtime?.state ?? 'queued',
      },
    });
  });

  const agentX = 430 + (maxLevel + 1) * 285 + 140;
  const resourceByNode = new Map(
    scene.initial_resources.map((snapshot) => [snapshot.node_id, snapshot]),
  );
  scene.nodes.forEach((node, index) => {
    const assigned = playback.filter((task) => task.target === node.id);
    const load = agentPlaybackLoad(assigned, node.max_concurrency);
    nodes.push({
      id: `agent:${node.id}`,
      type: 'agent',
      position: { x: agentX, y: 54 + index * 248 },
      data: {
        kind: 'agent',
        label: node.display_name || node.id,
        subtitle: node.id,
        hardwareKind: node.kind,
        architecture: node.architecture,
        online: resourceByNode.get(node.id)?.online ?? false,
        tasks: assigned,
        activeSlots: load.activeSlots,
        maxConcurrency: node.max_concurrency,
        utilization: load.utilization,
      },
    });
  });

  const edges: Edge[] = [];
  scene.tasks.forEach((task) => {
    if ((dag.parents[task.id] ?? []).length === 0) {
      edges.push({
        id: `central-${task.id}`,
        source: 'central-scheduler',
        target: `task:${task.id}`,
        type: 'smoothstep',
        animated: true,
        deletable: false,
        style: { stroke: '#7dd3a8', strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#7dd3a8' },
      });
    }
  });
  dag.graphEdges.forEach((edge) => {
    const typed = edge.kind === 'data';
    const color = typed ? '#65b9c8' : '#8a929c';
    edges.push({
      id: edge.id,
      source: `task:${edge.from}`,
      target: `task:${edge.to}`,
      type: 'smoothstep',
      label: edge.label,
      deletable: false,
      style: { stroke: color, strokeWidth: typed ? 1.7 : 1.35 },
      labelStyle: { fill: '#9dcbd4', fontSize: 8 },
      labelBgStyle: { fill: '#171a1f', fillOpacity: 0.94 },
      markerEnd: { type: MarkerType.ArrowClosed, color },
    });
  });
  playback.forEach((task) => {
    if (!task.target) return;
    edges.push({
      id: `placement:${task.id}:${task.target}`,
      source: `task:${task.id}`,
      target: `agent:${task.target}`,
      type: 'smoothstep',
      deletable: false,
      style: { stroke: '#d7a94a', strokeWidth: 1, strokeDasharray: '5 5' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#d7a94a' },
    });
  });
  return { nodes, edges };
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [scenarioType, setScenarioType] = useState<ScenarioType>('warehouse');
  const [customScene, setCustomScene] = useState('');
  const [robotCount, setRobotCount] = useState(2);
  const [edgeCount, setEdgeCount] = useState(1);
  const [difficulty, setDifficulty] = useState<Difficulty>('medium');
  const [hardware, setHardware] = useState<HardwareId>('orin_nx');
  const [algorithm, setAlgorithm] = useState<Algorithm>('dag_deadline');
  const [seed, setSeed] = useState(7);
  const [useLlm, setUseLlm] = useState(false);
  const [llmStatus, setLlmStatus] = useState<LlmStatus | null>(null);
  const [taskCategories, setTaskCategories] = useState<TaskCategory[]>([
    'localization',
    'environment_understanding',
    'object_detection',
    'local_planning',
    'obstacle_avoidance',
    'local_control',
  ]);
  const [scene, setScene] = useState<BenchmarkScene | null>(null);
  const [runtimeRun, setRuntimeRun] = useState<RuntimeWorkflowRun | null>(null);
  const [playbackState, setPlaybackState] = useState<PlaybackState>('idle');
  const [playhead, setPlayhead] = useState(0);
  const [apiStatus, setApiStatus] = useState('Connecting');
  const [error, setError] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);
  const [layoutRevision, setLayoutRevision] = useState(0);
  const [nodes, setNodes, onNodesChange] = useNodesState<FlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const playStartedAt = useRef(0);
  const playStartedFrom = useRef(0);
  const initialBuildDone = useRef(false);

  const runtimeTasks = runtimeRun?.result?.task_results ?? EMPTY_RUNTIME_TASKS;
  const makespan = runtimeRun?.result?.metrics.makespan_ms ?? 0;
  const playback = useMemo(
    () => (scene?.tasks ?? []).map((task) => taskPlayback(
      task,
      runtimeTasks.find((runtimeTask) => runtimeTask.task_id === task.id),
      playhead,
      makespan,
    )),
    [scene, runtimeTasks, playhead, makespan],
  );

  const requestPayload: GenerateSceneRequest = useMemo(() => ({
    scenario_type: scenarioType,
    custom_scene: customScene || undefined,
    robot_count: robotCount,
    edge_count: edgeCount,
    task_categories: taskCategories,
    difficulty,
    seed,
    use_llm: useLlm,
  }), [
    scenarioType,
    customScene,
    robotCount,
    edgeCount,
    taskCategories,
    difficulty,
    seed,
    useLlm,
  ]);

  useEffect(() => {
    health()
      .then((response) => {
        setApiStatus(`MARS ${response.mars_version}`);
        setLlmStatus({
          configured: response.llm_configured,
          provider: response.provider,
          model: response.model,
        });
      })
      .catch(() => {
        setApiStatus('API offline');
        setLlmStatus(null);
      });
  }, []);

  const applyHardware = useCallback((generated: BenchmarkScene) => {
    const profile = HARDWARE[hardware];
    return {
      ...generated,
      nodes: generated.nodes.map((node) => (
        node.kind !== 'robot'
          ? node
          : {
            ...node,
            architecture: profile.architecture,
            cpu_capacity: profile.cpu,
            gpu_capacity: profile.gpu,
            memory_gb: profile.memory,
          }
      )),
    };
  }, [hardware]);

  const buildScene = useCallback(async (payload: GenerateSceneRequest) => {
    setBuilding(true);
    setError(null);
    setPlaybackState('idle');
    setRuntimeRun(null);
    setPlayhead(0);
    try {
      if (payload.task_categories.length === 0) throw new Error('Select at least one atomic task.');
      const generated = applyHardware(await generateScene(payload));
      setScene(generated);
      setLayoutRevision((value) => value + 1);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBuilding(false);
    }
  }, [applyHardware]);

  useEffect(() => {
    if (initialBuildDone.current) return;
    initialBuildDone.current = true;
    void buildScene(requestPayload);
  }, [buildScene, requestPayload]);

  useEffect(() => {
    if (!scene) return;
    const graph = initialGraph(scene, playback);
    setNodes(graph.nodes);
    setEdges(graph.edges);
  }, [scene, layoutRevision, setEdges, setNodes]);

  useEffect(() => {
    if (!scene) return;
    const playbackById = new Map(playback.map((task) => [task.id, task]));
    setNodes((current) => current.map((node) => {
      if (node.data.kind === 'central') {
        return {
          ...node,
          data: {
            ...node.data,
            queue: playback,
            state: playbackState,
          },
        };
      }
      if (node.data.kind === 'task' && node.data.task) {
        const task = playbackById.get(node.data.task.id);
        return {
          ...node,
          data: {
            ...node.data,
            subtitle: task?.target || node.data.task.source_robot_id,
            progress: task?.progress ?? 0,
            state: task?.state ?? 'queued',
          },
        };
      }
      if (node.data.kind === 'agent') {
        const nodeId = node.id.replace('agent:', '');
        const assigned = playback.filter((task) => task.target === nodeId);
        const maxConcurrency = node.data.maxConcurrency ?? 1;
        const load = agentPlaybackLoad(assigned, maxConcurrency);
        return {
          ...node,
          data: {
            ...node.data,
            tasks: assigned,
            activeSlots: load.activeSlots,
            utilization: load.utilization,
          },
        };
      }
      return node;
    }));
  }, [playback, playbackState, scene, setNodes]);

  useEffect(() => {
    if (playbackState !== 'running' || makespan <= 0) return undefined;
    let frame = 0;
    const tick = (now: number) => {
      const elapsed = now - playStartedAt.current;
      const next = Math.min(
        makespan,
        playStartedFrom.current + (elapsed / PLAYBACK_DURATION_MS) * makespan,
      );
      setPlayhead(next);
      if (next >= makespan) {
        setPlaybackState(
          runtimeRun?.result?.workflow.state === 'succeeded'
            ? 'complete'
            : 'failed',
        );
        return;
      }
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [makespan, playbackState, runtimeRun]);

  const toggleTask = (category: TaskCategory) => {
    setTaskCategories((current) => (
      current.includes(category)
        ? current.filter((item) => item !== category)
        : [...current, category]
    ));
  };

  async function run() {
    if (!scene || playbackState === 'submitting') return;
    setError(null);
    setRuntimeRun(null);
    setPlayhead(0);
    setPlaybackState('submitting');
    try {
      const accepted = await submitRuntimeWorkflow(scene, algorithm, seed);
      const start = Date.now();
      let current: RuntimeWorkflowRun | null = null;
      while (Date.now() - start < POLL_TIMEOUT_MS) {
        current = await getRuntimeWorkflow(accepted.run_id);
        setRuntimeRun(current);
        if (current.status === 'succeeded' || current.status === 'failed') break;
        await new Promise((resolve) => window.setTimeout(resolve, POLL_INTERVAL_MS));
      }
      if (!current || current.status === 'accepted' || current.status === 'running') {
        throw new Error('Runtime polling timed out after 60 seconds.');
      }
      if (!current.result) {
        throw new Error(current.error || 'Runtime execution failed without a report.');
      }
      playStartedFrom.current = 0;
      playStartedAt.current = performance.now();
      setPlaybackState(
        current.result.metrics.makespan_ms > 0
          ? 'running'
          : current.result.workflow.state === 'succeeded'
            ? 'complete'
            : 'failed',
      );
      setLayoutRevision((value) => value + 1);
    } catch (reason) {
      setPlaybackState('error');
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function pausePlayback() {
    if (playbackState !== 'running') return;
    setPlaybackState('paused');
  }

  function continuePlayback() {
    if (playbackState !== 'paused' || !runtimeRun?.result) return;
    playStartedFrom.current = playhead;
    playStartedAt.current = performance.now();
    setPlaybackState('running');
  }

  function reset() {
    setRuntimeRun(null);
    setPlayhead(0);
    setPlaybackState('idle');
    setError(null);
    setLayoutRevision((value) => value + 1);
  }

  const progress = makespan > 0 ? clamp(playhead / makespan) : 0;
  const succeeded = playback.filter((task) => task.state === 'succeeded').length;
  const runningCount = playback.filter((task) => task.state === 'running').length;

  return (
    <div className={`studio-shell ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
      <aside className="settings-sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark"><Workflow size={19} /></div>
          {sidebarOpen && (
            <div>
              <strong>MARS Studio</strong>
              <small>Scheduler graph</small>
            </div>
          )}
          <button
            type="button"
            className="icon-button sidebar-toggle"
            aria-label={sidebarOpen ? 'Collapse settings' : 'Expand settings'}
            title={sidebarOpen ? 'Collapse settings' : 'Expand settings'}
            onClick={() => setSidebarOpen((value) => !value)}
          >
            {sidebarOpen ? <ChevronLeft size={17} /> : <ChevronRight size={17} />}
          </button>
        </div>

        {sidebarOpen && (
          <div className="settings-scroll">
            <SettingsSection icon={<SlidersHorizontal size={15} />} title="Scene">
              <label htmlFor="scenario">Scene</label>
              <select id="scenario" value={scenarioType} onChange={(event) => setScenarioType(event.target.value as ScenarioType)}>
                {SCENARIOS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
              </select>
              {scenarioType === 'custom' && (
                <textarea
                  value={customScene}
                  onChange={(event) => setCustomScene(event.target.value)}
                  placeholder="Describe the operating scene"
                  aria-label="Custom scene description"
                />
              )}
              <label className="llm-option">
                <input
                  type="checkbox"
                  checked={useLlm}
                  disabled={!llmStatus?.configured}
                  onChange={(event) => setUseLlm(event.target.checked)}
                />
                <span>Use LLM scene generation</span>
              </label>
              <small className="settings-note">
                {llmStatus?.configured
                  ? `${llmStatus.provider} / ${llmStatus.model}`
                  : 'Configure an LLM provider in the backend to enable this option.'}
              </small>
              <label htmlFor="difficulty">Task difficulty</label>
              <div className="segmented" id="difficulty">
                {DIFFICULTIES.map((item) => (
                  <button
                    type="button"
                    key={item}
                    className={difficulty === item ? 'active' : ''}
                    onClick={() => setDifficulty(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </SettingsSection>

            <SettingsSection icon={<Cpu size={15} />} title="Hardware">
              <div className="stepper-row">
                <label htmlFor="robot-count">Robot nodes</label>
                <div className="stepper">
                  <button type="button" onClick={() => setRobotCount((value) => Math.max(1, value - 1))}>-</button>
                  <input id="robot-count" type="number" min={1} max={50} value={robotCount} onChange={(event) => setRobotCount(boundedInteger(event.target.value, 1, 50, robotCount))} />
                  <button type="button" onClick={() => setRobotCount((value) => Math.min(50, value + 1))}>+</button>
                </div>
              </div>
              <div className="stepper-row">
                <label htmlFor="edge-count">Edge nodes</label>
                <div className="stepper">
                  <button type="button" onClick={() => setEdgeCount((value) => Math.max(0, value - 1))}>-</button>
                  <input id="edge-count" type="number" min={0} max={8} value={edgeCount} onChange={(event) => setEdgeCount(boundedInteger(event.target.value, 0, 8, edgeCount))} />
                  <button type="button" onClick={() => setEdgeCount((value) => Math.min(8, value + 1))}>+</button>
                </div>
              </div>
              <label>Robot hardware</label>
              <div className="hardware-options">
                {(Object.entries(HARDWARE) as Array<[HardwareId, typeof HARDWARE[HardwareId]]>).map(([id, profile]) => (
                  <button
                    type="button"
                    key={id}
                    className={hardware === id ? 'active' : ''}
                    onClick={() => setHardware(id)}
                  >
                    <Cpu size={14} />
                    <span>{profile.label}</span>
                    <small>{profile.memory} GB</small>
                  </button>
                ))}
              </div>
            </SettingsSection>

            <SettingsSection icon={<Box size={15} />} title="Atomic tasks">
              <div className="task-options">
                {TASK_CATEGORIES.map((category) => (
                  <label key={category}>
                    <input
                      type="checkbox"
                      checked={taskCategories.includes(category)}
                      onChange={() => toggleTask(category)}
                    />
                    <span>{taskLabel(category)}</span>
                  </label>
                ))}
              </div>
            </SettingsSection>

            <SettingsSection icon={<Gauge size={15} />} title="Scheduler">
              <label htmlFor="algorithm">Policy</label>
              <select id="algorithm" value={algorithm} onChange={(event) => setAlgorithm(event.target.value as Algorithm)}>
                {ALGORITHMS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
              </select>
              <label htmlFor="seed">Deterministic seed</label>
              <input id="seed" type="number" min={0} max={2_147_483_647} value={seed} onChange={(event) => setSeed(boundedInteger(event.target.value, 0, 2_147_483_647, seed))} />
            </SettingsSection>

            <button
              type="button"
              className="apply-button"
              onClick={() => void buildScene(requestPayload)}
              disabled={building || playbackState === 'submitting'}
            >
              <RefreshCcw size={15} className={building ? 'spin' : ''} />
              {building ? 'Building graph' : 'Apply settings'}
            </button>
          </div>
        )}
      </aside>

      <main className="graph-workspace">
        <header className="command-bar">
          <div className="workspace-identity">
            <span className={`api-indicator ${apiStatus === 'API offline' ? 'offline' : ''}`} />
            <div>
              <strong>{scene?.title ?? 'Loading scene'}</strong>
              <small title={scene?.generation_note || undefined}>
                {apiStatus} | {scene?.workflow_id ?? 'No workflow'}
                {scene ? ` | ${scene.generation_source}` : ''}
              </small>
            </div>
          </div>
          <div className="run-controls" aria-label="Runtime controls">
            <button type="button" className="run" onClick={() => void run()} disabled={!scene || playbackState === 'submitting' || playbackState === 'running' || playbackState === 'paused'} title="Run workflow">
              <Play size={15} fill="currentColor" /> Run
            </button>
            <button type="button" onClick={pausePlayback} disabled={playbackState !== 'running'} title="Pause playback">
              <Pause size={15} /> Pause
            </button>
            <button type="button" onClick={reset} disabled={!scene || playbackState === 'submitting'} title="Reset workflow">
              <RotateCcw size={15} /> Reset
            </button>
            <button type="button" onClick={continuePlayback} disabled={playbackState !== 'paused' || !runtimeRun?.result} title="Continue playback">
              <Play size={15} /> Continue
            </button>
          </div>
          <div className="runtime-summary">
            <span className={`runtime-state ${playbackState}`}>{playbackState}</span>
            <strong>{Math.round(progress * 100)}%</strong>
          </div>
        </header>

        <section className="canvas-shell">
          {error && (
            <div className="workspace-error" role="alert">
              <CircleStop size={16} />
              <span>{error}</span>
              <button type="button" onClick={() => setError(null)}>Dismiss</button>
            </div>
          )}
          <ReactFlow<FlowNode, Edge>
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodesConnectable={false}
            edgesReconnectable={false}
            fitView
            fitViewOptions={{ padding: 0.16, maxZoom: 1 }}
            minZoom={0.2}
            maxZoom={1.8}
            deleteKeyCode={null}
            selectionOnDrag
            panOnScroll
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#40464f" />
            <MiniMap
              pannable
              zoomable
              nodeColor={(node) => {
                if (node.data?.kind === 'central') return '#69c493';
                if (node.data?.kind === 'agent') return '#d7a94a';
                return '#747d89';
              }}
              maskColor="rgba(15, 17, 20, 0.72)"
            />
            <Controls showInteractive={false} />
            <Panel position="bottom-center">
              <div className="timeline-panel">
                <div className="timeline-top">
                  <span><Zap size={13} /> Virtual timeline</span>
                  <strong>{playhead.toFixed(1)} / {makespan.toFixed(1)} ms</strong>
                </div>
                <div className="timeline-track">
                  <span style={{ width: `${progress * 100}%` }} />
                  {playback.map((task) => (
                    <i
                      key={task.id}
                      style={{ left: `${makespan ? clamp(task.start / makespan) * 100 : 0}%` }}
                      title={`${task.label}: ${task.start.toFixed(1)} ms`}
                    />
                  ))}
                </div>
                <div className="timeline-stats">
                  <span>{playback.length} tasks</span>
                  <span>{runningCount} running</span>
                  <span>{succeeded} completed</span>
                </div>
              </div>
            </Panel>
          </ReactFlow>
          {!scene && !building && (
            <div className="canvas-empty">
              <Workflow size={34} />
              <strong>No workflow graph</strong>
              <span>Apply scene settings to build one.</span>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

function SettingsSection({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="settings-section">
      <h2>{icon}<span>{title}</span></h2>
      {children}
    </section>
  );
}
