import { describe, expect, it } from 'vitest';

import { canonicalDag } from './dag';
import type { DataEdgeSpec, Workload } from './types';

function task(id: string, dependencies: string[] = []): Workload {
  return {
    id,
    name: id,
    source_robot_id: 'robot_1',
    task_type: id,
    priority: 1,
    compute_demand: 1,
    gpu_demand: 0,
    latency_budget_ms: 100,
    safety_level: 1,
    model_requirement: '',
    data_size_mb: 0,
    output_size_mb: 0,
    bandwidth_requirement_mbps: 0,
    energy_budget_j: 1,
    allow_local_fallback: true,
    result_verification: '',
    arrival_time_ms: 0,
    deadline_ms: 100,
    dependencies,
    stage_index: 0,
    expected_accuracy: 1,
    input_ports: [],
    output_ports: [],
  };
}

function dataEdge(from: string, to: string, suffix = ''): DataEdgeSpec {
  return {
    producer_task: from,
    producer_port: `out${suffix}`,
    consumer_task: to,
    consumer_port: `in${suffix}`,
    message_type: `example.Message${suffix}`,
  };
}

function scene(tasks: Workload[], dataEdges: DataEdgeSpec[] = []) {
  return {
    tasks,
    data_edges: dataEdges,
  };
}

describe('canonicalDag', () => {
  it('treats a DataEdge-only producer as a canonical parent', () => {
    const dag = canonicalDag(scene([task('producer'), task('consumer')], [
      dataEdge('producer', 'consumer'),
    ]));

    expect(dag.valid).toBe(true);
    expect(dag.parents.consumer).toEqual(['producer']);
    expect(dag.levels).toEqual({ producer: 0, consumer: 1 });
    expect(dag.graphEdges).toMatchObject([
      { from: 'producer', to: 'consumer', kind: 'data' },
    ]);
  });

  it('deduplicates parent relations while preserving multiple typed edges', () => {
    const dag = canonicalDag(scene(
      [task('producer'), task('consumer', ['producer'])],
      [dataEdge('producer', 'consumer', 'A'), dataEdge('producer', 'consumer', 'B')],
    ));

    expect(dag.parents.consumer).toEqual(['producer']);
    expect(dag.levels.consumer).toBe(1);
    expect(dag.graphEdges.filter((edge) => edge.kind === 'dependency')).toHaveLength(0);
    expect(dag.graphEdges.filter((edge) => edge.kind === 'data')).toHaveLength(2);
  });

  it('computes longest-path levels across dependency and data fan-out', () => {
    const dag = canonicalDag(scene(
      [task('root'), task('left'), task('right'), task('join', ['left'])],
      [
        dataEdge('root', 'left'),
        dataEdge('root', 'right'),
        dataEdge('right', 'join'),
      ],
    ));

    expect(dag.valid).toBe(true);
    expect(dag.parents.join).toEqual(['left', 'right']);
    expect(dag.levels).toEqual({ root: 0, left: 1, right: 1, join: 2 });
    expect(dag.topologicalOrder).toEqual(['root', 'left', 'right', 'join']);
  });

  it('marks mixed dependency and DataEdge cycles invalid without dropping task keys', () => {
    const dag = canonicalDag(
      scene([task('first', ['second']), task('second')], [dataEdge('first', 'second')]),
    );

    expect(dag.valid).toBe(false);
    expect(dag.topologicalOrder).toEqual([]);
    expect(dag.parents).toEqual({ first: ['second'], second: ['first'] });
  });
});
