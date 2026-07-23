import type { DataEdgeSpec, Workload } from './types';

export interface DagSource {
  tasks: ReadonlyArray<Pick<Workload, 'id' | 'dependencies'>>;
  data_edges: readonly DataEdgeSpec[];
}

export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  kind: 'dependency' | 'data';
  label?: string;
}

export interface CanonicalDag {
  valid: boolean;
  levels: Record<string, number>;
  parents: Record<string, string[]>;
  topologicalOrder: string[];
  dependencyEdges: GraphEdge[];
  graphEdges: GraphEdge[];
}

export function canonicalDag(scene: DagSource): CanonicalDag {
  const taskIds = new Set(scene.tasks.map((task) => task.id));
  const taskOrder = new Map(scene.tasks.map((task, index) => [task.id, index]));
  const adjacency = new Map(scene.tasks.map((task) => [task.id, new Set<string>()]));
  const parentSets = new Map(scene.tasks.map((task) => [task.id, new Set<string>()]));
  const indegree = new Map(scene.tasks.map((task) => [task.id, 0]));
  const dependencyEdges: GraphEdge[] = [];
  const seenRelations = new Set<string>();

  const addRelation = (from: string, to: string) => {
    if (!taskIds.has(from) || !taskIds.has(to)) return;
    const relation = `${from}\u0000${to}`;
    if (seenRelations.has(relation)) return;
    seenRelations.add(relation);
    adjacency.get(from)?.add(to);
    parentSets.get(to)?.add(from);
    indegree.set(to, (indegree.get(to) ?? 0) + 1);
  };

  scene.tasks.forEach((task) => {
    task.dependencies.forEach((dependency) => {
      dependencyEdges.push({
        id: `dependency:${dependency}:${task.id}`,
        from: dependency,
        to: task.id,
        kind: 'dependency',
      });
      addRelation(dependency, task.id);
    });
  });
  scene.data_edges.forEach((edge) => addRelation(edge.producer_task, edge.consumer_task));

  const queue = scene.tasks
    .filter((task) => (indegree.get(task.id) ?? 0) === 0)
    .map((task) => task.id);
  const levels: Record<string, number> = Object.fromEntries(scene.tasks.map((task) => [task.id, 0]));
  const topologicalOrder: string[] = [];

  while (queue.length) {
    queue.sort((left, right) => (taskOrder.get(left) ?? 0) - (taskOrder.get(right) ?? 0));
    const current = queue.shift();
    if (!current) break;
    topologicalOrder.push(current);
    adjacency.get(current)?.forEach((next) => {
      levels[next] = Math.max(levels[next] ?? 0, (levels[current] ?? 0) + 1);
      const remaining = (indegree.get(next) ?? 1) - 1;
      indegree.set(next, remaining);
      if (remaining === 0) queue.push(next);
    });
  }

  const typedRelations = new Set(
    scene.data_edges.map((edge) => `${edge.producer_task}\u0000${edge.consumer_task}`),
  );
  const graphEdges: GraphEdge[] = [
    ...dependencyEdges.filter((edge) => !typedRelations.has(`${edge.from}\u0000${edge.to}`)),
    ...scene.data_edges.map((edge, index) => ({
      id: `data:${edge.producer_task}:${edge.producer_port}:${edge.consumer_task}:${edge.consumer_port}:${index}`,
      from: edge.producer_task,
      to: edge.consumer_task,
      kind: 'data' as const,
      label: edge.message_type,
    })),
  ];
  const parents = Object.fromEntries(
    scene.tasks.map((task) => [
      task.id,
      [...(parentSets.get(task.id) ?? [])].sort(
        (left, right) => (taskOrder.get(left) ?? 0) - (taskOrder.get(right) ?? 0),
      ),
    ]),
  );

  return {
    valid: topologicalOrder.length === scene.tasks.length,
    levels,
    parents,
    topologicalOrder,
    dependencyEdges,
    graphEdges,
  };
}
