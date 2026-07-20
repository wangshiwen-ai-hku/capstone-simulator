export type Difficulty = 'easy' | 'medium' | 'hard' | 'stress';
export type ScenarioType = 'warehouse' | 'hospital' | 'campus' | 'factory' | 'disaster' | 'custom';
export type Algorithm = 'dag_deadline' | 'rule_based' | 'local_first' | 'edge_first' | 'greedy_cost' | 'external';
export type TaskClass = 'local_safety' | 'realtime_offloadable' | 'edge_heavy';

export const TASK_CATEGORIES = [
  'obstacle_avoidance',
  'object_detection',
  'segmentation',
  'path_planning',
  'data_compression',
  'vla_inference',
  'llm_planning',
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
  cpu_capacity: number;
  gpu_capacity: number;
  memory_gb: number;
  bandwidth_mbps: number;
  base_latency_ms: number;
  battery_wh?: number | null;
  safety_capable: boolean;
}

export interface ResourceSnapshot {
  node_id: string;
  cpu_util: number;
  gpu_util: number;
  memory_util: number;
  temperature_c: number;
  power_w: number;
  network_latency_ms: number;
}

export interface Workload {
  id: string;
  name: string;
  source_robot_id: string;
  task_type: string;
  task_class: TaskClass;
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
  fallback_policy: string;
  result_verification: string;
  arrival_time_ms: number;
  deadline_ms: number;
  dependencies: string[];
  stage_index: number;
  expected_accuracy: number;
}

export interface BenchmarkScene {
  id: string;
  title: string;
  natural_language_description: string;
  scenario_type: string;
  difficulty: Difficulty;
  nodes: NodeSpec[];
  initial_resources: ResourceSnapshot[];
  tasks: Workload[];
  workflow_id: string;
  workflow_deadline_ms: number;
  failure_policy: 'skip_descendants' | 'fail_fast';
  stressors: string[];
  success_criteria: string[];
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
}

export interface TaskRunResult {
  task_id: string;
  workflow_id: string;
  task_name: string;
  task_class: TaskClass;
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
  task_class_summary: Record<TaskClass, {
    task_count: number;
    success_rate: number;
    avg_latency_ms: number;
    edge_offload_ratio: number;
  }>;
  dag: {
    valid: boolean;
    topological_order: string[];
    levels: Record<string, number>;
    edges: Array<{ from: string; to: string }>;
  };
  transport: Record<string, unknown>;
}
