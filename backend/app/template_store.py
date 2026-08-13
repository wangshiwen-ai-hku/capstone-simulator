"""Persistent, versioned benchmark templates backed by validated scene JSON."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import RLock
from uuid import uuid4

from .schemas import BenchmarkTemplate, BenchmarkTemplateCreate


class TemplateStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._lock = RLock()

    def _path(self, template_id: str) -> Path:
        if not re.fullmatch(r"template_[a-f0-9]{12}", template_id):
            raise KeyError(template_id)
        return self.directory / f"{template_id}.json"

    def list(self) -> list[BenchmarkTemplate]:
        if not self.directory.exists():
            return []
        templates: list[BenchmarkTemplate] = []
        with self._lock:
            for path in self.directory.glob("template_*.json"):
                try:
                    templates.append(
                        BenchmarkTemplate.model_validate_json(
                            path.read_text(encoding="utf-8")
                        )
                    )
                except (OSError, ValueError):
                    continue
        return sorted(templates, key=lambda item: item.updated_at, reverse=True)

    def get(self, template_id: str) -> BenchmarkTemplate:
        path = self._path(template_id)
        if not path.exists():
            raise KeyError(template_id)
        return BenchmarkTemplate.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def create(self, request: BenchmarkTemplateCreate) -> BenchmarkTemplate:
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
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(template.id)
        temporary = path.with_suffix(".tmp")
        with self._lock:
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

    def delete(self, template_id: str) -> None:
        path = self._path(template_id)
        if not path.exists():
            raise KeyError(template_id)
        with self._lock:
            path.unlink()
