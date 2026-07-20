"""Execution-profile catalog for measured or synthetic MARS workloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import NodeKind, TaskClass


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "mars" / "profiles.synthetic.json"


@dataclass(frozen=True)
class ExecutionProfile:
    task_type: str
    task_class: TaskClass
    node_kind: NodeKind
    model_variant: str
    input_shape: str
    precision: str
    batch_size: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_per_s: float
    peak_memory_mb: float
    energy_j: float
    output_size_mb: float
    supported: bool = True
    provenance: str = "synthetic_placeholder"


class ProfileCatalog:
    def __init__(self, profiles: list[ExecutionProfile]) -> None:
        self.profiles = tuple(profiles)
        self._items = {(item.task_type, item.node_kind): item for item in profiles}

    def lookup(self, task_type: str, node_kind: NodeKind) -> ExecutionProfile | None:
        return self._items.get((task_type, node_kind))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PROFILE_PATH) -> "ProfileCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            [
                ExecutionProfile(
                    task_type=item["task_type"],
                    task_class=TaskClass(item["task_class"]),
                    node_kind=NodeKind(item["node_kind"]),
                    model_variant=item["model_variant"],
                    input_shape=item["input_shape"],
                    precision=item["precision"],
                    batch_size=item["batch_size"],
                    p50_ms=item["p50_ms"],
                    p95_ms=item["p95_ms"],
                    p99_ms=item["p99_ms"],
                    throughput_per_s=item["throughput_per_s"],
                    peak_memory_mb=item["peak_memory_mb"],
                    energy_j=item["energy_j"],
                    output_size_mb=item["output_size_mb"],
                    supported=item.get("supported", True),
                    provenance=item.get("provenance", raw.get("provenance", "unknown")),
                )
                for item in raw["profiles"]
            ]
        )


def load_default_catalog() -> ProfileCatalog | None:
    try:
        return ProfileCatalog.load()
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
