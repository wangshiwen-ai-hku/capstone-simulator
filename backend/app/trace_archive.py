"""Scene-centric on-disk traces for generation and scheduler calls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TRACE_DIR = _REPO_ROOT / "tmp" / "mars-traces"
_SAFE_COMPONENT = re.compile(r"[^a-zA-Z0-9_.-]+")
_API_KEY = re.compile(r"\bsk-[a-zA-Z0-9_-]{8,}\b")
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+\S+")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;&]+"
)
_URL_USERINFO = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)
_SENSITIVE_KEY = re.compile(
    r"(?i)(api[_-]?key|authorization|credential|password|secret|token)"
)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_trace_root(settings: Settings) -> Path:
    raw = (settings.mars_trace_dir or "").strip()
    path = Path(raw) if raw else _DEFAULT_TRACE_DIR
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _safe_component(value: str) -> str:
    normalized = _SAFE_COMPONENT.sub("-", value.strip().replace("/", "-"))
    return normalized.strip("-.") or "unknown"


def _safe_relative_path(filename: str) -> Path:
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"invalid trace path: {filename!r}")
    return Path(*(_safe_component(part) for part in path.parts))


def _redact_text(value: object) -> str:
    text = str(value)
    try:
        structured = json.loads(text)
    except (TypeError, ValueError):
        structured = None
    if isinstance(structured, (dict, list)):
        return json.dumps(
            _redact_data(structured),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    text = _API_KEY.sub("<redacted-api-key>", text)
    text = _BEARER_TOKEN.sub("Bearer <redacted>", text)
    text = _CREDENTIAL_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}=<redacted>",
        text,
    )
    text = _URL_USERINFO.sub(r"\1<redacted>@", text)
    return text


def _safe_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if parsed.port is not None:
            hostname = f"{hostname}:{parsed.port}"
        return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))
    except ValueError:
        return "<invalid-url>"


def _json_data(payload: Mapping[str, Any] | Any) -> Any:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return _redact_data(payload)


def _redact_data(value: Any, *, key: str = "") -> Any:
    if key and _SENSITIVE_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_data(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_data(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _ensure_private_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)


def _write_json_atomic(target: Path, payload: Mapping[str, Any] | Any) -> Path:
    _ensure_private_directory(target.parent)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    encoded = (
        json.dumps(
            _json_data(payload),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        temporary.replace(target)
        target.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    logger.info("MARS trace archived: %s", target)
    return target


def _now() -> datetime:
    return datetime.now().astimezone()


def _timestamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S.%f")


@dataclass(frozen=True)
class TraceSession:
    trace_id: str
    scene_trace_id: str
    endpoint: str
    directory: Path
    root_directory: Path
    created_at: datetime

    def write_json(
        self,
        filename: str,
        payload: Mapping[str, Any] | Any,
    ) -> Path | None:
        """Best-effort atomic write below this session."""

        target = self.directory / _safe_relative_path(filename)
        try:
            return _write_json_atomic(target, payload)
        except OSError as exc:
            logger.warning("MARS trace write failed for %s: %s", target, exc)
            return None

    def write_request(self, payload: Mapping[str, Any] | Any) -> Path | None:
        return self.write_json("request.json", payload)

    def write_response(self, payload: Mapping[str, Any] | Any) -> Path | None:
        return self.write_json("response.json", payload)


def trace_status(settings: Settings | None = None) -> dict[str, object]:
    active = settings or get_settings()
    return {
        "enabled": _parse_bool(active.mars_trace_archive),
        "layout": "scene/calls-by-solver",
        "schema_version": "mars.trace.v3",
    }


def trace_enabled(settings: Settings | None = None) -> bool:
    return bool(trace_status(settings)["enabled"])


def _new_scene_root(
    root: Path,
    *,
    created_at: datetime,
    imported_scene_id: str = "",
) -> tuple[str, Path]:
    suffix = (
        f"_imported-{_safe_component(imported_scene_id)}"
        if imported_scene_id
        else ""
    )
    scene_trace_id = f"{_timestamp(created_at)}{suffix}_{uuid4().hex[:8]}"
    directory = root / scene_trace_id
    _ensure_private_directory(directory)
    return scene_trace_id, directory


def _existing_scene_root(root: Path, scene_trace_id: str) -> Path | None:
    safe_id = _safe_component(scene_trace_id)
    if safe_id != scene_trace_id:
        return None
    candidate = root / safe_id
    if not candidate.is_dir():
        return None
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        return None
    return resolved


def _begin_session(
    endpoint: str,
    settings: Settings | None = None,
    *,
    scene_trace_id: str | None = None,
    algorithm: str = "",
    scene_id: str = "",
) -> TraceSession | None:
    """Create a scene root or attach a scheduler call to an existing root."""

    active = settings or get_settings()
    if not _parse_bool(active.mars_trace_archive):
        return None
    created_at = _now()
    root = _resolve_trace_root(active)
    _ensure_private_directory(root)

    if endpoint == "generate-scene":
        resolved_scene_id, scene_root = _new_scene_root(
            root,
            created_at=created_at,
        )
        session = TraceSession(
            trace_id=resolved_scene_id,
            scene_trace_id=resolved_scene_id,
            endpoint=endpoint,
            directory=scene_root,
            root_directory=scene_root,
            created_at=created_at,
        )
        session.write_json(
            "scene/meta.json",
            {
                "schema_version": "mars.trace.v3",
                "scene_trace_id": resolved_scene_id,
                "created_at": created_at.isoformat(),
                "status": "generating",
            },
        )
        logger.warning(
            "MARS trace scene %s → %s",
            resolved_scene_id,
            scene_root,
        )
        return session

    scene_root = (
        _existing_scene_root(root, scene_trace_id)
        if scene_trace_id
        else None
    )
    imported = scene_root is None
    if scene_root is None:
        resolved_scene_id, scene_root = _new_scene_root(
            root,
            created_at=created_at,
            imported_scene_id=scene_id or "scene",
        )
        scene_meta = {
            "schema_version": "mars.trace.v3",
            "scene_trace_id": resolved_scene_id,
            "created_at": created_at.isoformat(),
            "status": "imported",
            "scene_id": scene_id,
            "note": "Scheduler call arrived without a known generation trace.",
        }
        _write_json_atomic(scene_root / "scene" / "meta.json", scene_meta)
    else:
        resolved_scene_id = scene_root.name

    safe_endpoint = _safe_component(endpoint)
    safe_algorithm = _safe_component(algorithm or "default")
    call_id = f"{_timestamp(created_at)}_{safe_algorithm}_{uuid4().hex[:8]}"
    call_directory = scene_root / "calls" / safe_endpoint / call_id
    _ensure_private_directory(call_directory)
    session = TraceSession(
        trace_id=call_id,
        scene_trace_id=resolved_scene_id,
        endpoint=endpoint,
        directory=call_directory,
        root_directory=scene_root,
        created_at=created_at,
    )
    session.write_json(
        "meta.json",
        {
            "schema_version": "mars.trace.v3",
            "call_id": call_id,
            "scene_trace_id": resolved_scene_id,
            "endpoint": endpoint,
            "algorithm": algorithm,
            "created_at": created_at.isoformat(),
            "imported_scene_root": imported,
        },
    )
    logger.warning(
        "MARS trace call %s attached to scene %s → %s",
        call_id,
        resolved_scene_id,
        call_directory,
    )
    return session


def begin_session(
    endpoint: str,
    settings: Settings | None = None,
    *,
    scene_trace_id: str | None = None,
    algorithm: str = "",
    scene_id: str = "",
) -> TraceSession | None:
    """Best-effort scene trace creation that never breaks API execution."""

    try:
        return _begin_session(
            endpoint,
            settings,
            scene_trace_id=scene_trace_id,
            algorithm=algorithm,
            scene_id=scene_id,
        )
    except OSError as exc:
        logger.warning("MARS trace session unavailable for %s: %s", endpoint, exc)
        return None


def summarize_llm_response(
    content: str,
    *,
    preview_chars: int = 800,
) -> dict[str, object]:
    stripped = content.strip() if isinstance(content, str) else str(content or "")
    preview = stripped[:preview_chars]
    summary: dict[str, object] = {
        "response_bytes": len(stripped.encode("utf-8")),
        "response_chars": len(stripped),
        "preview": preview,
        "preview_truncated": len(stripped) > preview_chars,
    }
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            tasks = data.get("tasks")
            edges = data.get("data_edges")
            nodes = data.get("nodes")
            workflow_id = data.get("workflow_id")
            title = data.get("title")
            summary["parsed_json"] = {
                "workflow_id": (
                    workflow_id if isinstance(workflow_id, str) else None
                ),
                "title": title if isinstance(title, str) else None,
                "task_count": len(tasks) if isinstance(tasks, list) else None,
                "data_edge_count": (
                    len(edges) if isinstance(edges, list) else None
                ),
                "node_count": len(nodes) if isinstance(nodes, list) else None,
                "top_level_keys": sorted(data.keys()),
            }
    except (TypeError, ValueError):
        summary["parsed_json"] = None
    return summary


def exception_chain(exc: BaseException) -> list[dict[str, str]]:
    """Return a bounded and redacted exception chain for diagnostics."""

    chain: list[dict[str, str]] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": _redact_text(current),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


def archive_llm_request(
    session: TraceSession | None,
    *,
    provider: str,
    model: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
    max_retries: int,
    stream: bool,
) -> None:
    if session is None:
        return
    session.write_json(
        "llm/request.json",
        {
            "started_at": _now().isoformat(),
            "provider": provider,
            "model": model,
            "base_url": _safe_base_url(base_url),
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "stream": stream,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    )


def archive_llm_result(
    session: TraceSession | None,
    *,
    provider: str,
    model: str,
    response_content: str,
    success: bool,
    elapsed_ms: float,
    error: BaseException | None = None,
) -> None:
    if session is None:
        return
    session.write_json(
        "llm/meta.json",
        {
            "finished_at": _now().isoformat(),
            "provider": provider,
            "model": model,
            "success": success,
            "elapsed_ms": round(elapsed_ms, 3),
            "error_chain": exception_chain(error) if error else [],
            "response_summary": summarize_llm_response(response_content),
        },
    )
    if response_content:
        session.write_json("llm/response.json", {"content": response_content})


def log_startup_banner(settings: Settings | None = None) -> None:
    active = settings or get_settings()
    status = trace_status(active)
    if status["enabled"]:
        logger.warning(
            "MARS_TRACE_ARCHIVE=1 — scene traces will be written under %s",
            _resolve_trace_root(active),
        )
    else:
        logger.info(
            "MARS trace archive disabled (set MARS_TRACE_ARCHIVE=1 to enable)"
        )


__all__ = [
    "TraceSession",
    "archive_llm_request",
    "archive_llm_result",
    "begin_session",
    "exception_chain",
    "log_startup_banner",
    "summarize_llm_response",
    "trace_enabled",
    "trace_status",
]
