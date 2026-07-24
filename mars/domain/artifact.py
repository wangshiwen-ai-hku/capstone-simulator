"""Artifact references and task-input bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    producer_task_id: str
    node_id: str
    size_mb: float
    uri: str = ""
    checksum: str = ""
    producer_port: str = "result"
    message_type: str = ""


@dataclass(frozen=True)
class InputArtifactBinding:
    """Bind one reusable artifact to an exact consumer task input port."""

    consumer_task_id: str
    consumer_port: str
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        if not self.consumer_task_id.strip():
            raise ValueError("consumer_task_id must be non-blank")
        if not self.consumer_port.strip():
            raise ValueError("consumer_port must be non-blank")
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("artifact must be an ArtifactRef")


def artifacts_from_bindings(
    bindings: Iterable[InputArtifactBinding],
) -> tuple[ArtifactRef, ...]:
    """Project port bindings to unique payload transfers in stable order."""

    artifacts_by_id: dict[str, ArtifactRef] = {}
    ordered: list[ArtifactRef] = []
    for binding in bindings:
        artifact = binding.artifact
        existing = artifacts_by_id.get(artifact.artifact_id)
        if existing is not None:
            if existing != artifact:
                raise ValueError("one artifact_id cannot describe different artifacts")
            continue
        artifacts_by_id[artifact.artifact_id] = artifact
        ordered.append(artifact)
    return tuple(ordered)


__all__ = [
    "ArtifactRef",
    "InputArtifactBinding",
    "artifacts_from_bindings",
]
