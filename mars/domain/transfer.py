"""Estimated and reserved transfers across the declared topology."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransferEstimate:
    """Estimated movement of one input artifact across a directed path."""

    transfer_id: str
    source_node_id: str
    target_node_id: str
    size_mb: float
    path_link_ids: tuple[str, ...]
    bottleneck_bandwidth_mbps: float
    transfer_time_ms: float
    feasible: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path_link_ids",
            tuple(self.path_link_ids),
        )
        if not all(
            math.isfinite(value)
            for value in (
                self.size_mb,
                self.transfer_time_ms,
            )
        ):
            raise ValueError("transfer estimates must be finite")
        if self.size_mb < 0:
            raise ValueError("transfer size_mb must be non-negative")
        if self.transfer_time_ms < 0:
            raise ValueError("transfer_time_ms must be non-negative")
        if (
            self.feasible
            and self.size_mb > 0
            and self.source_node_id != self.target_node_id
        ):
            if not self.path_link_ids:
                raise ValueError("remote feasible transfers require a link path")
            if self.bottleneck_bandwidth_mbps <= 0:
                raise ValueError("remote feasible transfers require positive bandwidth")


@dataclass(frozen=True)
class TransferReservation:
    """A planned interval during which a task owns its transfer path."""

    reservation_id: str
    epoch_id: str
    task_id: str
    transfer_id: str
    path_link_ids: tuple[str, ...]
    start_ms: float
    finish_ms: float
    size_mb: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path_link_ids",
            tuple(self.path_link_ids),
        )
        if not all(
            math.isfinite(value)
            for value in (
                self.start_ms,
                self.finish_ms,
                self.size_mb,
            )
        ):
            raise ValueError("transfer reservation values must be finite")
        if self.finish_ms < self.start_ms:
            raise ValueError("transfer reservation cannot finish before it starts")
        if self.size_mb < 0:
            raise ValueError("transfer reservation size_mb must be non-negative")


__all__ = [
    "TransferEstimate",
    "TransferReservation",
]
