export type Difficulty = 'easy' | 'medium' | 'hard' | 'stress';
export type ScenarioType = 'warehouse' | 'hospital' | 'campus' | 'factory' | 'disaster' | 'custom';
export type Algorithm = 'dag_deadline' | 'rule_based' | 'local_first' | 'edge_first' | 'greedy_cost' | 'binary_offload';
export type TaskClass = 'local_safety' | 'realtime_offloadable' | 'edge_heavy';

export const TASK_CATEGORIES = [
  'obstacle_avoidance',
  'emergency_stop',
  'local_control',
  'localization',
  'environment_understanding',
  'object_detection',
  'semantic_segmentation',
  'local_planning',
  'data_compression',
  'local_llm_7b',
  'local_llm_10b',
  'result_verification',
  'map_fusion',
] as const;

export type TaskCategory = typeof TASK_CATEGORIES[number];

export interface GenerateSceneRequest {
  scenario_type: ScenarioType;
  custom_scene?: string;
  robot_count: number;
  edge_count: number;
  task_categories: TaskCategory[];
  difficulty: Difficulty;
  seed: number;
  use_llm: boolean;
}

export interface NodeSpec {
  id: string;
  kind: 'robot' | 'edge' | 'cloud';
  display_name: string;
  architecture: string;
  cpu_capacity: number;
  gpu_capacity: number;
  memory_gb: number;
  bandwidth_mbps: number;
  base_latency_ms: number;
  battery_wh?: number | null;
  safety_capable: boolean;
  capabilities: string[];
  supported_models: string[];
  max_concurrency: number;
}

export interface ResourceSnapshot {
  node_id: string;
  cpu_util: number;
  gpu_util: number;
  memory_util: number;
  temperature_c: number;
  power_w: number;
  network_latency_ms: number;
  online: boolean;
}

export interface PlacementConstraintsSpec {
  pinned_node_id: string;
  pin_to_source: boolean;
  allowed_node_kinds: Array<'robot' | 'edge' | 'cloud'>;
  preferred_node_kinds: Array<'robot' | 'edge' | 'cloud'>;
  required_capabilities: string[];
  allow_source_node: boolean;
  allow_other_robots: boolean;
  safety_required: boolean;
  allow_fallback: boolean;
  stateful: boolean;
  idempotent: boolean;
  splittable: boolean;
  replicable: boolean;
}

export interface LinkSpec {
  id: string;
  source_node_id: string;
  target_node_id: string;
  bandwidth_mbps: number;
  base_latency_ms: number;
}

export interface LinkSnapshot {
  link_id: string;
  available_bandwidth_mbps: number;
  latency_ms: number;
  jitter_ms: number;
  packet_loss_rate: number;
  online: boolean;
}

export interface Workload {
  id: string;
  name: string;
  source_robot_id: string;
  task_type: string;
  task_class?: TaskClass | null;
  priority: number;
  compute_demand: number;
  gpu_demand: number;
  latency_budget_ms: number;
  safety_level: number;
  model_requirement: string;
  data_size_mb: number;
  output_size_mb: number;
  bandwidth_requirement_mbps: number;
  energy_budget_j: number;
  allow_local_fallback: boolean;
  placement_constraints?: PlacementConstraintsSpec | null;
  result_verification: string;
  arrival_time_ms: number;
  deadline_ms: number;
  dependencies: string[];
  stage_index: number;
  expected_accuracy: number;
  input_ports: PortSpec[];
  output_ports: PortSpec[];
}

export interface PortSpec {
  name: string;
  message_type: string;
}

export interface DataEdgeSpec {
  producer_task: string;
  producer_port: string;
  consumer_task: string;
  consumer_port: string;
  message_type: string;
}

export interface BenchmarkScene {
  id: string;
  title: string;
  natural_language_description: string;
  scenario_type: string;
  difficulty: Difficulty;
  nodes: NodeSpec[];
  initial_resources: ResourceSnapshot[];
  links?: LinkSpec[] | null;
  link_snapshots?: LinkSnapshot[] | null;
  tasks: Workload[];
  data_edges: DataEdgeSpec[];
  workflow_id: string;
  workflow_deadline_ms: number;
  failure_policy: 'skip_descendants' | 'fail_fast';
  stressors: string[];
  success_criteria: string[];
  generation_source: 'deterministic' | 'llm' | 'deterministic_fallback';
  generation_note: string;
}

export interface SimulationMetrics {
  task_count: number;
  success_rate: number;
  deadline_miss_rate: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  avg_energy_j: number;
  total_energy_j: number;
  bandwidth_mb: number;
  makespan_ms: number;
  edge_offload_ratio: number;
  safety_violation_count: number;
  skipped_task_count: number;
  workflow_success_rate: number;
  critical_path_ms: number;
  dag_depth: number;
  total_solver_time_ms: number;
  max_solver_time_ms: number;
  scheduling_epoch_count: number;
  expected_success_reward: number;
  communication_time_ms: number;
  peak_cpu_utilization: number;
  peak_gpu_utilization: number;
  peak_memory_utilization: number;
  maximum_resource_utilization: number;
  workflow_evaluation_objective: number;
}

export interface TaskRunResult {
  task_id: string;
  workflow_id: string;
  task_name: string;
  task_class?: TaskClass | null;
  stage_index: number;
  dependencies: string[];
  source_robot_id: string;
  target_node_id: string;
  mode: string;
  priority: number;
  start_time_ms: number;
  finish_time_ms: number;
  queue_delay_ms: number;
  compute_time_ms: number;
  communication_time_ms: number;
  total_latency_ms: number;
  energy_j: number;
  deadline_missed: boolean;
  success: boolean;
  state: string;
  reason: string;
  input_locations: string[];
  output_ref: string;
}

export interface SimulationResponse {
  algorithm: string;
  metrics: SimulationMetrics;
  task_results: TaskRunResult[];
  node_utilization: Record<string, number>;
  logs: string[];
  workflow: {
    workflow_id: string;
    state: string;
    failure_policy: string;
    deadline_time_ms: number;
    deadline_missed: boolean;
    state_counts: Record<string, number>;
    critical_path: string[];
  };
  task_class_summary: Partial<Record<TaskClass, {
    task_count: number;
    success_rate: number;
    avg_latency_ms: number;
    edge_offload_ratio: number;
  }>>;
  dag: {
    valid: boolean;
    topological_order: string[];
    levels: Record<string, number>;
    edges: Array<{ from: string; to: string }>;
  };
  transport: Record<string, unknown>;
}

export interface RuntimeAgent {
  agent_id: string;
  kind: 'robot' | 'edge';
  architecture: string;
  registered: boolean;
  online: boolean;
  heartbeat_sequence: number;
  last_heartbeat_ms: number;
  active_reservations: number;
  max_concurrency: number;
  completed_attempts: number;
  failed_attempts: number;
  busy_time_ms: number;
  utilization: number;
  capabilities: string[];
  supported_models: string[];
  resources: Record<string, number>;
}

export interface RuntimeStatus {
  scheduler_id: string;
  status: string;
  agent_count: number;
  agents: RuntimeAgent[];
  runtime: string;
  topology: {
    central_schedulers: number;
    orin_agents: number;
    edge_agents: number;
  };
  run_count: number;
}

export interface RuntimeAttempt {
  attempt_id: string;
  attempt_no: number;
  state: string;
  target_node_id: string;
  mode: string;
  start_time_ms: number;
  finish_time_ms: number;
  compute_time_ms: number;
  communication_time_ms: number;
  transferred_mb: number;
  energy_j: number;
  input_artifact_ids: string[];
  error_code: string;
}

export interface RuntimeTaskResult {
  task_id: string;
  task_name: string;
  task_type: string;
  task_class?: TaskClass | null;
  state: string;
  source_node_id: string;
  target_node_id: string;
  mode: string;
  dependencies: string[];
  attempt_count: number;
  attempts: RuntimeAttempt[];
  outputs: Array<{
    artifact_id: string;
    producer_task_id: string;
    node_id: string;
    size_mb: number;
    uri: string;
    checksum: string;
    producer_port: string;
    message_type: string;
  }>;
}

export interface RuntimeEvent {
  sequence: number;
  time_ms: number;
  event_type: string;
  message: string;
  workflow_id: string;
  task_id: string;
  attempt_id: string;
  agent_id: string;
}

export interface RuntimeReport {
  workflow: {
    workflow_id: string;
    state: string;
    failure_policy: string;
    state_counts: Record<string, number>;
    critical_path: string[];
    topological_order: string[];
    levels: Record<string, number>;
  };
  metrics: Record<string, number>;
  task_results: RuntimeTaskResult[];
  agents: RuntimeAgent[];
  data_edges: DataEdgeSpec[];
  events: RuntimeEvent[];
  logs: string[];
}

export interface RuntimeWorkflowRun {
  run_id: string;
  workflow_id: string;
  status: 'accepted' | 'running' | 'succeeded' | 'failed';
  result: RuntimeReport | null;
  error: string;
}
