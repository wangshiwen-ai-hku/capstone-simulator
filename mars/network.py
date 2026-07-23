"""Directed link topology and deterministic transfer estimation for MARS."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

from .models import (
    LinkSnapshot,
    LinkSpec,
    NodeSnapshot,
    NodeSpec,
    TransferEstimate,
)


class NetworkTopologyError(ValueError):
    """The declared link graph is incomplete or internally inconsistent."""


class NetworkTopology:
    """Validated directed graph used by candidate generation.

    Paths minimize estimated transfer time for the specific payload size. This
    makes a low-latency narrow link and a high-bandwidth link comparable
    without embedding network policy in an optimizer.
    """

    def __init__(
        self,
        node_ids: Iterable[str],
        link_specs: Iterable[LinkSpec],
        link_snapshots: Iterable[LinkSnapshot],
        *,
        node_online: Mapping[str, bool] | None = None,
    ) -> None:
        self.node_ids = frozenset(node_ids)
        self.node_online = (
            {node_id: True for node_id in self.node_ids}
            if node_online is None
            else dict(node_online)
        )
        if set(self.node_online) != set(self.node_ids):
            missing = sorted(set(self.node_ids) - set(self.node_online))
            unknown = sorted(set(self.node_online) - set(self.node_ids))
            raise NetworkTopologyError(
                "node online-state inventory mismatch: "
                f"missing states={missing}; unknown states={unknown}"
            )
        specs = tuple(link_specs)
        snapshots = tuple(link_snapshots)
        self.spec_by_id = {item.link_id: item for item in specs}
        self.snapshot_by_id = {item.link_id: item for item in snapshots}
        if len(self.spec_by_id) != len(specs):
            raise NetworkTopologyError("link ids must be unique")
        if len(self.snapshot_by_id) != len(snapshots):
            raise NetworkTopologyError("link snapshot ids must be unique")
        if set(self.spec_by_id) != set(self.snapshot_by_id):
            missing = sorted(set(self.spec_by_id) - set(self.snapshot_by_id))
            unknown = sorted(set(self.snapshot_by_id) - set(self.spec_by_id))
            raise NetworkTopologyError(
                "link inventory mismatch: "
                f"missing snapshots={missing}; unknown snapshots={unknown}"
            )

        endpoints: set[tuple[str, str]] = set()
        adjacency: dict[str, list[LinkSpec]] = defaultdict(list)
        for spec in specs:
            if (
                spec.source_node_id not in self.node_ids
                or spec.target_node_id not in self.node_ids
            ):
                raise NetworkTopologyError(
                    f"link {spec.link_id} references an unknown node"
                )
            endpoint = (spec.source_node_id, spec.target_node_id)
            if endpoint in endpoints:
                raise NetworkTopologyError(
                    "parallel links for the same directed endpoints are not "
                    f"supported: {endpoint[0]}->{endpoint[1]}"
                )
            endpoints.add(endpoint)
            adjacency[spec.source_node_id].append(spec)
        self._adjacency = {
            node_id: tuple(sorted(items, key=lambda item: item.link_id))
            for node_id, items in adjacency.items()
        }

    def estimate(
        self,
        *,
        transfer_id: str,
        source_node_id: str,
        target_node_id: str,
        size_mb: float,
        minimum_bandwidth_mbps: float = 0.0,
    ) -> TransferEstimate:
        if source_node_id not in self.node_ids:
            return _infeasible_transfer(
                transfer_id,
                source_node_id,
                target_node_id,
                size_mb,
                "unknown_source_node",
            )
        if target_node_id not in self.node_ids:
            return _infeasible_transfer(
                transfer_id,
                source_node_id,
                target_node_id,
                size_mb,
                "unknown_target_node",
            )
        if size_mb < 0:
            raise ValueError("transfer size_mb must be non-negative")
        if minimum_bandwidth_mbps < 0:
            raise ValueError("minimum_bandwidth_mbps must be non-negative")
        if (
            not self.node_online[source_node_id]
            or not self.node_online[target_node_id]
        ):
            return _infeasible_transfer(
                transfer_id,
                source_node_id,
                target_node_id,
                size_mb,
                "no_online_link_path",
            )
        if source_node_id == target_node_id or size_mb == 0:
            return TransferEstimate(
                transfer_id=transfer_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                size_mb=size_mb,
                path_link_ids=(),
                bottleneck_bandwidth_mbps=math.inf,
                transfer_time_ms=0.0,
            )

        # (total milliseconds, path ids, node id, bottleneck bandwidth)
        frontier: list[tuple[float, tuple[str, ...], str, float]] = [
            (0.0, (), source_node_id, math.inf)
        ]
        best: dict[str, tuple[float, tuple[str, ...]]] = {
            source_node_id: (0.0, ())
        }
        while frontier:
            total_ms, path, node_id, bottleneck = heapq.heappop(frontier)
            if best.get(node_id) != (total_ms, path):
                continue
            if not self.node_online[node_id]:
                continue
            if node_id == target_node_id:
                return TransferEstimate(
                    transfer_id=transfer_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    size_mb=size_mb,
                    path_link_ids=path,
                    bottleneck_bandwidth_mbps=bottleneck,
                    transfer_time_ms=total_ms,
                )
            for spec in self._adjacency.get(node_id, ()):
                snapshot = self.snapshot_by_id[spec.link_id]
                if not self.node_online[spec.target_node_id]:
                    continue
                bandwidth = min(
                    spec.bandwidth_mbps,
                    snapshot.available_bandwidth_mbps,
                )
                if not snapshot.online or bandwidth <= 0:
                    continue
                next_bottleneck = min(bottleneck, bandwidth)
                if next_bottleneck + 1e-9 < minimum_bandwidth_mbps:
                    continue
                payload_ms = (
                    size_mb
                    * 8.0
                    / bandwidth
                    * 1000.0
                    / max(1e-9, 1.0 - snapshot.packet_loss_rate)
                )
                edge_ms = (
                    spec.base_latency_ms
                    + snapshot.latency_ms
                    + snapshot.jitter_ms
                    + payload_ms
                )
                next_total = total_ms + edge_ms
                next_path = (*path, spec.link_id)
                current = best.get(spec.target_node_id)
                if current is None or (next_total, next_path) < current:
                    best[spec.target_node_id] = (next_total, next_path)
                    heapq.heappush(
                        frontier,
                        (
                            next_total,
                            next_path,
                            spec.target_node_id,
                            next_bottleneck,
                        ),
                    )

        reason = (
            "bandwidth_below_requirement"
            if minimum_bandwidth_mbps > 0
            and self._has_online_path(
                source_node_id,
                target_node_id,
            )
            else "no_online_link_path"
        )
        return _infeasible_transfer(
            transfer_id,
            source_node_id,
            target_node_id,
            size_mb,
            reason,
        )

    def _has_online_path(
        self,
        source_node_id: str,
        target_node_id: str,
    ) -> bool:
        if (
            not self.node_online[source_node_id]
            or not self.node_online[target_node_id]
        ):
            return False
        frontier = [source_node_id]
        visited = {source_node_id}
        while frontier:
            node_id = frontier.pop()
            if not self.node_online[node_id]:
                continue
            for spec in self._adjacency.get(node_id, ()):
                snapshot = self.snapshot_by_id[spec.link_id]
                if not self.node_online[spec.target_node_id]:
                    continue
                bandwidth = min(
                    spec.bandwidth_mbps,
                    snapshot.available_bandwidth_mbps,
                )
                if not snapshot.online or bandwidth <= 0:
                    continue
                if spec.target_node_id == target_node_id:
                    return True
                if spec.target_node_id not in visited:
                    visited.add(spec.target_node_id)
                    frontier.append(spec.target_node_id)
        return False


def synthesize_legacy_full_mesh(
    node_specs: Iterable[NodeSpec],
    node_snapshots: Iterable[NodeSnapshot],
) -> tuple[tuple[LinkSpec, ...], tuple[LinkSnapshot, ...]]:
    """Convert deprecated node-level network fields into directed links.

    The generated direct-link estimate exactly matches the previous endpoint
    formula, allowing older API payloads and tests to migrate without changing
    their expected communication time.
    """

    specs = tuple(node_specs)
    snapshots = tuple(node_snapshots)
    snapshot_by_id = {item.node_id: item for item in snapshots}
    if len(snapshot_by_id) != len(snapshots):
        raise NetworkTopologyError("node snapshot ids must be unique")
    if {item.node_id for item in specs} != set(snapshot_by_id):
        raise NetworkTopologyError(
            "legacy topology requires one snapshot for every node"
        )

    links: list[LinkSpec] = []
    link_states: list[LinkSnapshot] = []
    for source in specs:
        source_state = snapshot_by_id[source.node_id]
        for target in specs:
            if source.node_id == target.node_id:
                continue
            target_state = snapshot_by_id[target.node_id]
            link_id = f"legacy:{source.node_id}->{target.node_id}"
            bandwidth = min(source.bandwidth_mbps, target.bandwidth_mbps)
            links.append(
                LinkSpec(
                    link_id=link_id,
                    source_node_id=source.node_id,
                    target_node_id=target.node_id,
                    bandwidth_mbps=bandwidth,
                    base_latency_ms=(
                        source.base_latency_ms + target.base_latency_ms
                    ),
                )
            )
            link_states.append(
                LinkSnapshot(
                    link_id=link_id,
                    available_bandwidth_mbps=bandwidth,
                    latency_ms=(
                        source_state.network_latency_ms
                        + target_state.network_latency_ms
                    ),
                    online=source_state.online and target_state.online,
                )
            )
    return tuple(links), tuple(link_states)


def _infeasible_transfer(
    transfer_id: str,
    source_node_id: str,
    target_node_id: str,
    size_mb: float,
    reason: str,
) -> TransferEstimate:
    return TransferEstimate(
        transfer_id=transfer_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        size_mb=size_mb,
        path_link_ids=(),
        bottleneck_bandwidth_mbps=0.0,
        transfer_time_ms=0.0,
        feasible=False,
        reason=reason,
    )
