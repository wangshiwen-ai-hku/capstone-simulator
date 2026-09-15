"""Content-addressed JSON artifacts, transferred only from configured peers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

import grpc

from interfaces.proto.mars.v1 import artifact_service_pb2 as pb
from interfaces.proto.mars.v1 import artifact_service_pb2_grpc as rpc


MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtifactFiles:
    """Immutable per-agent cache. Only explicit SHA-256 object names are served."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        if not _DIGEST.fullmatch(digest):
            raise ValueError("artifact key must be a lowercase SHA-256 digest")
        return self.directory / f"{digest}.json"

    def put(self, data: bytes) -> str:
        if not data or len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact is empty or exceeds the 2 MiB limit")
        digest = digest_bytes(data)
        path = self._path(digest)
        if path.exists():
            if self.read(digest) != data:
                raise ValueError("artifact content collision")
            return digest
        # A completed object is never visible until the atomic rename.
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as temp:
            temporary = Path(temp.name)
            try:
                temp.write(data)
                temp.flush()
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def read(self, digest: str) -> bytes:
        path = self._path(digest)
        if path.is_symlink():
            raise ValueError("artifact symlinks are not allowed")
        with path.open("rb") as source:
            data = source.read(MAX_ARTIFACT_BYTES + 1)
        if not data or len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact is empty or too large")
        if digest_bytes(data) != digest:
            raise ValueError("artifact checksum mismatch")
        return data


class ArtifactService(rpc.ArtifactStoreServicer):
    def __init__(self, files: ArtifactFiles) -> None:
        self.files = files

    async def ReadArtifact(self, request, context):
        try:
            data = self.files.read(request.sha256)
        except FileNotFoundError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "artifact not found")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return pb.ReadArtifactResponse(data=data, sha256=request.sha256)


async def fetch_artifact(
    artifact,
    *,
    agent_id: str,
    files: ArtifactFiles,
    peers: dict[str, str],
    timeout_seconds: float = 10.0,
) -> tuple[dict, int]:
    """Resolve an ArtifactRef; URI host is a node ID, never an arbitrary URL."""
    digest = artifact.checksum
    if not _DIGEST.fullmatch(digest):
        raise ValueError("missing or invalid artifact checksum")
    uri = urlsplit(artifact.uri)
    if (
        uri.scheme != "mars-artifact"
        or uri.netloc != artifact.node_id
        or uri.path != f"/{digest}"
        or uri.query
        or uri.fragment
        or not math.isfinite(artifact.size_mb)
        or artifact.size_mb <= 0
        or artifact.size_mb * 1_000_000 > MAX_ARTIFACT_BYTES
    ):
        raise ValueError("invalid artifact reference")
    transferred = 0
    if artifact.node_id == agent_id:
        data = files.read(digest)
    else:
        endpoint = peers.get(artifact.node_id)
        if endpoint is None:
            raise ValueError(
                f"artifact producer is not a configured peer: {artifact.node_id}"
            )
        # Remote reads are deliberate even if the digest exists locally: the
        # validation report must attest to an actual cross-node byte transfer.
        async with grpc.aio.insecure_channel(endpoint) as channel:
            response = await rpc.ArtifactStoreStub(channel).ReadArtifact(
                pb.ReadArtifactRequest(sha256=digest),
                timeout=timeout_seconds,
            )
        data = response.data
        if response.sha256 != digest or digest_bytes(data) != digest:
            raise ValueError("remote artifact checksum mismatch")
        if not data or len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError("remote artifact is empty or too large")
        files.put(data)
        transferred = len(data)
    if abs(len(data) - artifact.size_mb * 1_000_000) > 1:
        raise ValueError("artifact byte length differs from its declaration")
    envelope = json.loads(data)
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema") != "mars.hil.artifact.v1"
        or envelope.get("producer_task_id") != artifact.producer_task_id
        or envelope.get("producer_port") != artifact.producer_port
        or envelope.get("agent_id") != artifact.node_id
        or envelope.get("message_type") != artifact.message_type
        or not isinstance(envelope.get("payload"), dict)
    ):
        raise ValueError("artifact envelope does not match its binding")
    return envelope, transferred
