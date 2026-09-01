"""Persistent, versioned benchmark templates backed by validated scene JSON."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from threading import RLock
from uuid import uuid4

from .schemas import BenchmarkTemplate, BenchmarkTemplateCreate


WORKSPACE_TOKEN_PATTERN = re.compile(r"[a-f0-9]{64}")
TEMPLATE_ID_PATTERN = re.compile(r"template_[a-f0-9]{12}")


def validate_workspace_token(workspace_token: str) -> str:
    """Validate the browser-held capability before it influences a path."""

    if not WORKSPACE_TOKEN_PATTERN.fullmatch(workspace_token):
        raise ValueError("invalid template workspace token")
    return workspace_token


class TemplateStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._lock = RLock()

    def _workspace_directory(self, workspace_token: str) -> Path:
        validated_token = validate_workspace_token(workspace_token)
        workspace_digest = sha256(validated_token.encode("ascii")).hexdigest()
        return self.directory / "workspaces" / f"workspace_{workspace_digest}"

    def _path(self, workspace_token: str, template_id: str) -> Path:
        if not TEMPLATE_ID_PATTERN.fullmatch(template_id):
            raise KeyError(template_id)
        return self._workspace_directory(workspace_token) / f"{template_id}.json"

    def list(self, workspace_token: str) -> list[BenchmarkTemplate]:
        workspace_directory = self._workspace_directory(workspace_token)
        if not workspace_directory.exists():
            return []
        templates: list[BenchmarkTemplate] = []
        with self._lock:
            for path in workspace_directory.glob("template_*.json"):
                try:
                    templates.append(
                        BenchmarkTemplate.model_validate_json(
                            path.read_text(encoding="utf-8")
                        )
                    )
                except (OSError, ValueError):
                    continue
        return sorted(templates, key=lambda item: item.updated_at, reverse=True)

    def get(self, workspace_token: str, template_id: str) -> BenchmarkTemplate:
        path = self._path(workspace_token, template_id)
        with self._lock:
            if not path.exists():
                raise KeyError(template_id)
            return BenchmarkTemplate.model_validate_json(
                path.read_text(encoding="utf-8")
            )

    def create(
        self,
        workspace_token: str,
        request: BenchmarkTemplateCreate,
    ) -> BenchmarkTemplate:
        now = datetime.now(timezone.utc)
        template = BenchmarkTemplate(
            id=f"template_{uuid4().hex[:12]}",
            name=request.name.strip(),
            description=request.description.strip(),
            tags=request.tags,
            created_at=now,
            updated_at=now,
            scene=request.scene,
        )
        path = self._path(workspace_token, template.id)
        temporary = path.with_suffix(".tmp")
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(
                    template.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.replace(path)
        return template

    def delete(self, workspace_token: str, template_id: str) -> None:
        path = self._path(workspace_token, template_id)
        with self._lock:
            if not path.exists():
                raise KeyError(template_id)
            path.unlink()
