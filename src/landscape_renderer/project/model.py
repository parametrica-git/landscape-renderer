"""Проект .lscapeproj (EXP-04): источник, лист, профиль, стиль, seed, переопределения."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = "1.0"


@dataclass
class RenderSettings:
    style: str = "pero_akvarel"
    dpi: int = 220
    seed: int = 21
    light_azimuth_deg: float = 315.0      # откуда свет: 315° = сверху слева
    light_elevation_deg: float = 35.0
    shadow_k: float = 0.55
    season: str = "summer"
    annotations: bool = True
    statuses: bool = True


@dataclass
class Project:
    source: str
    layout: str | None = None
    viewport: str | None = None
    profile: str | None = None
    manual: dict[str, str] = field(default_factory=dict)       # "layer:ИМЯ" / "block:ИМЯ" / "handle:H" → класс
    overrides: dict[str, dict] = field(default_factory=dict)   # handle → {...}
    render: RenderSettings = field(default_factory=RenderSettings)
    units_to_m: float | None = None
    source_sha256: str | None = None
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    path: str | None = None

    def to_dict(self) -> dict:
        base = Path(self.path).parent if self.path else None

        def rel(p):
            if not p:
                return p
            if base:
                try:
                    return os.path.relpath(p, base)
                except ValueError:
                    return p
            return p

        return {"schema_version": SCHEMA, "project_id": self.project_id,
                "source": {"path": rel(self.source), "abs_path": os.path.abspath(self.source),
                           "sha256": self.source_sha256},
                "target": {"layout": self.layout, "viewport_handle": self.viewport},
                "profile": rel(self.profile), "units_to_m": self.units_to_m,
                "manual": self.manual, "overrides": self.overrides, "render": asdict(self.render)}

    def save(self, path: str | None = None) -> str:
        self.path = path or self.path
        Path(self.path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return self.path

    @classmethod
    def load(cls, path: str) -> "Project":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(d.get("schema_version", "1.0")).split(".")[0] != SCHEMA.split(".")[0]:
            raise ValueError(f"Несовместимая версия проекта: {d.get('schema_version')}")
        base = Path(path).parent

        def absp(p, fallback=None):
            if not p:
                return p
            q = (base / p).resolve()
            if q.exists():
                return str(q)
            return fallback or str(q)

        src = d["source"]
        return cls(source=absp(src["path"], src.get("abs_path")), layout=d["target"].get("layout"),
                   viewport=d["target"].get("viewport_handle"), profile=absp(d.get("profile")),
                   manual=d.get("manual", {}), overrides=d.get("overrides", {}),
                   render=RenderSettings(**d.get("render", {})), units_to_m=d.get("units_to_m"),
                   source_sha256=src.get("sha256"), project_id=d.get("project_id", str(uuid.uuid4())),
                   path=path)
