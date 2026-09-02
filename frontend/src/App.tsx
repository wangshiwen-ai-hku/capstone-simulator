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
  Bot,
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
  SquareStack,
  Workflow,
  Zap,
} from 'lucide-react';
import MarsModePanel, { type MarsMode } from './MarsModePanel';
import {
  generateScene,
  getArchitecture,
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
  SchedulingAlgorithmCapability,
  ScenarioType,
  TaskCategory,
  Workload,
} from './types';
import { DEFAULT_BINARY_FORMULATION, TASK_CATEGORIES } from './types';

const SCENARIOS: Array<{ value: ScenarioType; label: string }> = [
  { value: 'warehouse', label: 'Warehouse' },
  { value: 'hospital', label: 'Hospital' },
  { value: 'campus', label: 'Campus' },
  { value: 'factory', label: 'Factory' },
  { value: 'disaster', label: 'Disaster response' },
  { value: 'custom', label: 'Custom scene' },
];

const DIFFICULTIES: Difficulty[] = ['easy', 'medium', 'hard', 'stress'];
const STABLE_ALGORITHMS: Array<{ value: Algorithm; label: string }> = [
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
    cpu: 6,
    gpu: 67,
    memory: 8,
  },
  orin_nx: {
    label: 'Orin NX',
    architecture: 'jetson-orin-nx',
    cpu: 8,
    gpu: 157,
    memory: 16,
  },
  orin_agx: {
    label: 'AGX Orin',
    architecture: 'jetson-agx-orin',
    cpu: 12,
    gpu: 275,
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
  traceEnabled: boolean;
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

interface RuntimeMetricDescriptor {
  key: string;
  label: string;
  format: 'count' | 'milliseconds' | 'number' | 'percent';
}

const RUNTIME_METRIC_GROUPS: Array<{
  label: string;
  metrics: RuntimeMetricDescriptor[];
}> = [
  {
    label: 'Outcome',
    metrics: [
      { key: 'success_rate', label: 'Success', format: 'percent' },
      { key: 'required_task_on_time_rate', label: 'Required tasks on time', format: 'percent' },
      { key: 'executed_deadline_miss_rate', label: 'Executed deadline misses', format: 'percent' },
      { key: 'skipped_task_count', label: 'Skipped tasks', format: 'count' },
      { key: 'makespan_ms', label: 'Makespan', format: 'milliseconds' },
      { key: 'edge_offload_ratio', label: 'Edge offload', format: 'percent' },
    ],
  },
  {
    label: 'Solver',
    metrics: [
      { key: 'total_solver_time_ms', label: 'Total solve', format: 'milliseconds' },
      { key: 'max_solver_time_ms', label: 'Longest solve', format: 'milliseconds' },
      { key: 'scheduling_epoch_count', label: 'Epochs', format: 'count' },
    ],
  },
  {
    label: 'Evaluation',
    metrics: [
      { key: 'expected_success_reward', label: 'Expected reward', format: 'number' },
      { key: 'communication_time_ms', label: 'Communication', format: 'milliseconds' },
      { key: 'peak_cpu_utilization', label: 'Peak CPU', format: 'percent' },
      { key: 'peak_gpu_utilization', label: 'Peak GPU', format: 'percent' },
      { key: 'peak_memory_utilization', label: 'Peak memory', format: 'percent' },
      { key: 'maximum_resource_utilization', label: 'Peak utilization', format: 'percent' },
      { key: 'workflow_evaluation_objective', label: 'Objective', format: 'number' },
    ],
  },
];

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

function formatRuntimeMetric(value: number, format: RuntimeMetricDescriptor['format']) {
  if (format === 'percent') return `${(value * 100).toFixed(1)}%`;
  if (format === 'milliseconds') return `${value.toFixed(value < 10 ? 2 : 1)} ms`;
  if (format === 'count') return Math.round(value).toLocaleString();
  return value.toFixed(3);
}

function maximumDagWidth(scene: BenchmarkScene) {
  const levelCounts = new Map<number, number>();
  Object.values(canonicalDag(scene).levels).forEach((level) => {
    levelCounts.set(level, (levelCounts.get(level) ?? 0) + 1);
  });
  return Math.max(0, ...levelCounts.values());
}

function sourceCandidateAvailable(scene: BenchmarkScene, task: Workload) {
  const source = scene.nodes.find((node) => node.id === task.source_robot_id);
  if (!source) return false;
  const constraints = task.placement_constraints;
  if (!constraints) return true;
  if (constraints.pinned_node_id && constraints.pinned_node_id !== source.id) return false;
  if (!constraints.allow_source_node) return false;
  if (
    constraints.allowed_node_kinds.length > 0
    && !constraints.allowed_node_kinds.includes(source.kind)
  ) return false;
  if (constraints.safety_required && !source.safety_capable) return false;
  return constraints.required_capabilities.every((item) => source.capabilities.includes(item));
}

function compatibilityIssue(
  scene: BenchmarkScene | null,
  capability: SchedulingAlgorithmCapability | undefined,
) {
  if (!scene || !capability) return null;
  const compatibility = capability.compatibility;
  if (!compatibility) return 'The backend did not declare compatibility for this method.';
  const supportedKinds = Array.isArray(compatibility.supported_node_kinds)
    ? compatibility.supported_node_kinds
    : [];
  const unsupportedKinds = [...new Set(
    scene.nodes
      .map((node) => node.kind)
      .filter((kind) => !supportedKinds.includes(kind)),
  )];
  if (unsupportedKinds.length > 0) {
    return `This method does not support ${unsupportedKinds.join(', ')} nodes in the scene.`;
  }
  if (!compatibility.supports_multiple_nodes && scene.nodes.length > 1) {
    return 'This method supports only a single compute node.';
  }
  if (
    compatibility.requires_source_candidate
    && scene.tasks.some((task) => !sourceCandidateAvailable(scene, task))
  ) {
    return 'This method requires every task to permit its source robot as a candidate.';
  }
  if (
    Number.isFinite(compatibility.max_ready_tasks)
    && maximumDagWidth(scene) > Number(compatibility.max_ready_tasks)
  ) {
    return `This workflow may expose more than ${compatibility.max_ready_tasks} ready tasks at once.`;
  }
  return null;
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
          <span>{task?.gpu_demand.toFixed(1)} TOPS</span>
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

function RuntimeMetricsPanel({ run }: { run: RuntimeWorkflowRun | null }) {
  if (!run?.result) return null;
  const scheduling = run.result.workflow.scheduling;
  const schedulerCounts = (counts?: Record<string, number>) => (
    counts
      ? Object.entries(counts)
        .map(([id, count]) => `${id.replace(/_/g, ' ')} x ${count}`)
        .join(', ')
      : ''
  );
  const requestedAlgorithm = scheduling?.requested_algorithm
    ?? run.result.workflow.requested_algorithm
    ?? '';
  const effectiveOptimizers = schedulerCounts(scheduling?.effective_optimizers);
  const effectivePolicies = schedulerCounts(scheduling?.effective_policies);
  const solveStatuses = schedulerCounts(scheduling?.solve_statuses);
  const fallbackCount = scheduling?.fallback_count
    ?? run.result.metrics.fallback_count;
  const showScheduling = Boolean(
    requestedAlgorithm
    || effectiveOptimizers
    || effectivePolicies
    || solveStatuses
    || typeof fallbackCount === 'number',
  );
  const groups = RUNTIME_METRIC_GROUPS.map((group) => ({
    ...group,
    metrics: group.metrics.flatMap((metric) => {
      const value = run.result?.metrics[metric.key];
      return typeof value === 'number' && Number.isFinite(value)
        ? [{ ...metric, value }]
        : [];
    }),
  })).filter((group) => group.metrics.length > 0);

  return (
    <aside className="runtime-metrics-panel" aria-label="Runtime metrics">
      <div className="runtime-metrics-title">
        <strong>Runtime metrics</strong>
        <span>{run.result.workflow.state}</span>
      </div>
      {showScheduling && (
        <section>
          <h3>Scheduling audit</h3>
          <dl>
            {requestedAlgorithm && (
              <div><dt>Requested</dt><dd title={requestedAlgorithm}>{requestedAlgorithm.replace(/_/g, ' ')}</dd></div>
            )}
            {effectiveOptimizers && (
              <div><dt>Optimizers used</dt><dd title={effectiveOptimizers}>{effectiveOptimizers}</dd></div>
            )}
            {effectivePolicies && (
              <div><dt>Policies used</dt><dd title={effectivePolicies}>{effectivePolicies}</dd></div>
            )}
            {solveStatuses && (
              <div><dt>Solve status</dt><dd title={solveStatuses}>{solveStatuses}</dd></div>
            )}
            {typeof fallbackCount === 'number' && (
              <div><dt>Fallbacks</dt><dd>{formatRuntimeMetric(fallbackCount, 'count')}</dd></div>
            )}
          </dl>
        </section>
      )}
      {groups.map((group) => (
        <section key={group.label}>
          <h3>{group.label}</h3>
          <dl>
            {group.metrics.map((metric) => (
              <div key={metric.key} title={metric.key}>
                <dt>{metric.label}</dt>
                <dd>{formatRuntimeMetric(metric.value, metric.format)}</dd>
              </div>
            ))}
          </dl>
        </section>
      ))}
    </aside>
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
  const [sidebarMode, setSidebarMode] = useState<MarsMode>('studio');
  const [assistantExpanded, setAssistantExpanded] = useState(false);
  const [scenarioType, setScenarioType] = useState<ScenarioType>('warehouse');
  const [customScene, setCustomScene] = useState('');
  const [robotCount, setRobotCount] = useState(2);
  const [edgeCount, setEdgeCount] = useState(1);
  const [difficulty, setDifficulty] = useState<Difficulty>('medium');
  const [hardware, setHardware] = useState<HardwareId>('orin_nx');
  const [algorithm, setAlgorithm] = useState<Algorithm>('dag_deadline');
  const [algorithmCapabilities, setAlgorithmCapabilities] = useState<
    SchedulingAlgorithmCapability[]
  >([]);
  const [communicationWeight, setCommunicationWeight] = useState('');
  const [formulation, setFormulation] = useState('');
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

  const runtimeCapabilities = useMemo(
    () => algorithmCapabilities.filter((capability) => (
      Array.isArray(capability.execution_paths)
      && capability.execution_paths.includes('runtime')
    )),
    [algorithmCapabilities],
  );
  const binaryCapability = useMemo(
    () => runtimeCapabilities.find((capability) => capability.id === 'binary_offload'),
    [runtimeCapabilities],
  );
  const algorithms = useMemo(
    () => binaryCapability
      ? [
        ...STABLE_ALGORITHMS,
        { value: 'binary_offload' as const, label: binaryCapability.label },
      ]
      : STABLE_ALGORITHMS,
    [binaryCapability],
  );
  const selectedCapability = runtimeCapabilities.find(
    (capability) => capability.id === algorithm,
  );
  const binaryParameter = binaryCapability?.parameters?.communication_weight;
  const selectedFormulations = useMemo(() => {
    const advertised = Array.isArray(selectedCapability?.supported_formulations)
      ? selectedCapability.supported_formulations
        .filter((item): item is string => typeof item === 'string')
        .map((item) => item.trim())
        .filter(Boolean)
      : [];
    if (advertised.length > 0) return [...new Set(advertised)];
    const advertisedDefault = typeof selectedCapability?.default_formulation === 'string'
      ? selectedCapability.default_formulation.trim()
      : '';
    if (advertisedDefault) return [advertisedDefault];
    return algorithm === 'binary_offload' ? [DEFAULT_BINARY_FORMULATION] : [];
  }, [algorithm, selectedCapability]);
  const selectedDefaultFormulation = useMemo(() => {
    const advertisedDefault = typeof selectedCapability?.default_formulation === 'string'
      ? selectedCapability.default_formulation.trim()
      : '';
    if (advertisedDefault && selectedFormulations.includes(advertisedDefault)) {
      return advertisedDefault;
    }
    return algorithm === 'binary_offload'
      ? selectedFormulations[0] ?? DEFAULT_BINARY_FORMULATION
      : '';
  }, [selectedCapability, selectedFormulations]);
  const selectedFormulation = selectedFormulations.includes(formulation)
    ? formulation
    : selectedDefaultFormulation;

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
    robot_hardware: hardware,
  }), [
    scenarioType,
    customScene,
    robotCount,
    edgeCount,
    taskCategories,
    difficulty,
    seed,
    useLlm,
    hardware,
  ]);

  useEffect(() => {
    health()
      .then((response) => {
        setApiStatus(`MARS ${response.mars_version}`);
        setLlmStatus({
          configured: response.llm_configured,
          provider: response.provider,
          model: response.model,
          traceEnabled: response.trace_archive?.enabled ?? false,
        });
      })
      .catch(() => {
        setApiStatus('API offline');
        setLlmStatus(null);
      });
  }, []);

  useEffect(() => {
    let active = true;
    getArchitecture()
      .then((response) => {
        if (!active) return;
        const advertised = response.scheduling_capabilities?.algorithms;
        setAlgorithmCapabilities(
          Array.isArray(advertised)
            ? advertised.filter((capability) => (
              capability !== null
              && typeof capability === 'object'
              && typeof capability.id === 'string'
            ))
            : [],
        );
      })
      .catch(() => {
        if (active) setAlgorithmCapabilities([]);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (binaryParameter && Number.isFinite(binaryParameter.default)) {
      setCommunicationWeight(String(binaryParameter.default));
    }
  }, [binaryParameter]);

  useEffect(() => {
    setFormulation(selectedDefaultFormulation);
  }, [algorithm, selectedDefaultFormulation]);

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

  const importScene = useCallback((imported: BenchmarkScene, source: string) => {
    setError(null);
    setRuntimeRun(null);
    setPlaybackState('idle');
    setPlayhead(0);
    setScene(imported);
    setSidebarMode('studio');
    setAssistantExpanded(false);
    setLayoutRevision((value) => value + 1);
    setApiStatus((current) => current.startsWith('MARS') ? current : `Imported from ${source}`);
  }, []);

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

  const selectedCompatibilityIssue = useMemo(
    () => compatibilityIssue(scene, selectedCapability),
    [scene, selectedCapability],
  );
  const communicationWeightValue = Number(communicationWeight);
  const communicationWeightIssue = useMemo(() => {
    if (algorithm !== 'binary_offload' || !binaryParameter) return null;
    if (communicationWeight.trim() === '' || !Number.isFinite(communicationWeightValue)) {
      return `${binaryParameter.label} must be a number.`;
    }
    if (communicationWeightValue < binaryParameter.minimum) {
      return `${binaryParameter.label} must be at least ${binaryParameter.minimum}.`;
    }
    if (
      binaryParameter.maximum !== undefined
      && communicationWeightValue > binaryParameter.maximum
    ) {
      return `${binaryParameter.label} must be at most ${binaryParameter.maximum}.`;
    }
    return null;
  }, [algorithm, binaryParameter, communicationWeight, communicationWeightValue]);
  const schedulerIssue = selectedCompatibilityIssue ?? communicationWeightIssue;

  async function run() {
    if (!scene || playbackState === 'submitting' || schedulerIssue) return;
    setError(null);
    setRuntimeRun(null);
    setPlayhead(0);
    setPlaybackState('submitting');
    try {
      const accepted = await submitRuntimeWorkflow(
        scene,
        algorithm,
        seed,
        {
          ...(selectedFormulation ? { formulation: selectedFormulation } : {}),
          ...(algorithm === 'binary_offload' && binaryParameter
            ? { communicationWeight: communicationWeightValue }
            : {}),
        },
      );
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
    <div className={`studio-shell ${sidebarOpen ? '' : 'sidebar-collapsed'} ${assistantExpanded && sidebarMode === 'assistant' ? 'assistant-expanded' : ''}`}>
      <aside className="settings-sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark"><Workflow size={19} /></div>
          {sidebarOpen && (
            <div>
              <strong>{sidebarMode === 'studio' ? 'MARS Studio' : sidebarMode === 'assistant' ? 'Authoring Assistant' : 'MARS Templates'}</strong>
              <small>{sidebarMode === 'studio' ? 'Scheduler graph' : sidebarMode === 'assistant' ? 'Workflow authoring' : 'Benchmark library'}</small>
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
          <>
            <nav className="mars-mode-switcher" aria-label="MARS workspace mode">
              <button type="button" className={sidebarMode === 'studio' ? 'active' : ''} onClick={() => { setSidebarMode('studio'); setAssistantExpanded(false); }}><SlidersHorizontal size={13} />Studio</button>
              <button
                type="button"
                className={sidebarMode === 'assistant' ? 'active' : ''}
                onClick={() => setSidebarMode('assistant')}
                aria-label="Authoring Assistant"
                title="Authoring Assistant"
              ><Bot size={13} />Authoring</button>
              <button type="button" className={sidebarMode === 'templates' ? 'active' : ''} onClick={() => { setSidebarMode('templates'); setAssistantExpanded(false); }}><SquareStack size={13} />Templates</button>
            </nav>
            <MarsModePanel
              mode={sidebarMode}
              scene={scene}
              expanded={assistantExpanded}
              onExpandedChange={setAssistantExpanded}
              onImportScene={importScene}
              studio={<div className="settings-scroll">
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
                {llmStatus?.traceEnabled ? ' | trace archive on' : ''}
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
                    <small>{profile.memory} GB / {profile.gpu} TOPS</small>
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
              <label htmlFor="algorithm">Scheduling method</label>
              <select id="algorithm" value={algorithm} onChange={(event) => setAlgorithm(event.target.value as Algorithm)}>
                {algorithms.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
              </select>
              {algorithm === 'binary_offload' && binaryCapability && (
                <small className="capability-note">
                  <span>{binaryCapability.stability}</span>
                  Backend-advertised {binaryCapability.kind}
                </small>
              )}
              {(selectedFormulations.length > 1
                || (selectedFormulations.length === 1 && !selectedDefaultFormulation)) && (
                <div className="scheduler-parameter">
                  <label htmlFor="formulation">Formulation</label>
                  <select
                    id="formulation"
                    value={selectedFormulation}
                    onChange={(event) => setFormulation(event.target.value)}
                  >
                    <option value="">Default (unformulated)</option>
                    {selectedFormulations.map((item) => (
                      <option key={item} value={item}>{item}</option>
                    ))}
                  </select>
                </div>
              )}
              {algorithm === 'binary_offload' && binaryParameter && (
                <div className="scheduler-parameter">
                  <label htmlFor="communication-weight">{binaryParameter.label}</label>
                  <input
                    id="communication-weight"
                    type="number"
                    min={binaryParameter.minimum}
                    max={binaryParameter.maximum}
                    step={binaryParameter.step}
                    value={communicationWeight}
                    aria-invalid={Boolean(communicationWeightIssue)}
                    aria-describedby="communication-weight-description"
                    onChange={(event) => setCommunicationWeight(event.target.value)}
                  />
                  <small id="communication-weight-description" className="settings-note">
                    {binaryParameter.description}
                  </small>
                </div>
              )}
              {schedulerIssue && (
                <small className="scheduler-warning" role="status">{schedulerIssue}</small>
              )}
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
          </div>}
            />
          </>
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
            <button type="button" className="run" onClick={() => void run()} disabled={!scene || Boolean(schedulerIssue) || playbackState === 'submitting' || playbackState === 'running' || playbackState === 'paused'} title={schedulerIssue ?? 'Run workflow'}>
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
          {scene?.generation_source === 'deterministic_fallback' && (
            <div className="workspace-notice" role="status">
              <CircleStop size={16} />
              <span>
                {scene.generation_note
                  || 'LLM generation failed; deterministic fallback used.'}
              </span>
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
            <Panel position="top-right">
              <RuntimeMetricsPanel run={runtimeRun} />
            </Panel>
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
