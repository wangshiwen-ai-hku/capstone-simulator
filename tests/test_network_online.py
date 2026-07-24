from __future__ import annotations

from mars.domain import (
    LinkSnapshot,
    LinkSpec,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    PlacementConstraints,
    TaskClass,
    TaskInstance,
    TaskSpec,
)
from mars.network import NetworkTopology
from mars.optimizers import SchedulingEpoch
from mars.scheduler import build_scheduling_problem


NODE_IDS = ("source", "relay", "target")
LINK_SPECS = (
    LinkSpec("source-relay", "source", "relay", 100),
    LinkSpec("relay-target", "relay", "target", 100),
)
LINK_SNAPSHOTS = (
    LinkSnapshot("source-relay", 100),
    LinkSnapshot("relay-target", 100),
)


def test_path_cannot_traverse_an_offline_intermediate_node() -> None:
    online_topology = NetworkTopology(
        NODE_IDS,
        LINK_SPECS,
        LINK_SNAPSHOTS,
        node_online={"source": True, "relay": True, "target": True},
    )
    assert online_topology.estimate(
        transfer_id="online-path",
        source_node_id="source",
        target_node_id="target",
        size_mb=1,
    ).feasible

    offline_topology = NetworkTopology(
        NODE_IDS,
        LINK_SPECS,
        LINK_SNAPSHOTS,
        node_online={"source": True, "relay": False, "target": True},
    )
    estimate = offline_topology.estimate(
        transfer_id="offline-path",
        source_node_id="source",
        target_node_id="target",
        size_mb=1,
    )

    assert not estimate.feasible
    assert estimate.reason == "no_online_link_path"
    assert estimate.path_link_ids == ()


def test_scheduler_passes_node_online_state_to_the_topology() -> None:
    node_specs = {
        "source": NodeSpec(
            "source",
            NodeKind.ROBOT,
            4,
            1,
            8,
            100,
            1,
        ),
        "relay": NodeSpec(
            "relay",
            NodeKind.CLOUD,
            4,
            1,
            8,
            100,
            1,
        ),
        "target": NodeSpec(
            "target",
            NodeKind.EDGE,
            8,
            4,
            32,
            100,
            1,
        ),
    }
    node_snapshots = {
        "source": NodeSnapshot("source", online=True),
        "relay": NodeSnapshot("relay", online=False),
        "target": NodeSnapshot("target", online=True),
    }
    task = TaskInstance(
        task_id="task",
        workflow_id="workflow",
        name="task",
        source_node_id="source",
        spec=TaskSpec(
            task_type="test",
            task_class=TaskClass.EDGE_HEAVY,
            input_size_mb=1,
            placement_constraints=PlacementConstraints(
                pinned_node_id="target",
                allowed_node_kinds=(NodeKind.EDGE,),
                allow_source_node=False,
            ),
        ),
    )

    problem = build_scheduling_problem(
        SchedulingEpoch("offline-relay", 0, (task,)),
        node_specs=node_specs,
        node_snapshots=node_snapshots,
        parent_artifacts={},
        ready_time_ms={},
        link_specs=LINK_SPECS,
        link_snapshots=LINK_SNAPSHOTS,
    )

    candidate = problem.candidates["task"][0]
    assert not candidate.feasible
    assert candidate.reason == "no_online_link_path"
