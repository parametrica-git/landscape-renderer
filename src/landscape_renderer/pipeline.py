"""Сквозной конвейер: проект → CAD → сцена → рендер → экспорт. Его используют CLI и UI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cad import CadDocument, ViewportInfo, open_any, pick_viewport
from .cad.reader import sha256
from .mapping import Profile, build_report, summary
from .project import Project
from .render import get_backend
from .render.procedural import RenderResult
from .scene import Scene, build_scene, match_overrides

DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "packs" / "profiles" / "ru-landscape-starter.json"


@dataclass
class Session:
    project: Project
    cad: CadDocument
    viewport: ViewportInfo
    profile: Profile | None
    scene: Scene
    source_changed: bool = False
    override_warnings: list[str] | None = None

    @classmethod
    def open(cls, project: Project, allow_oda: bool = False) -> "Session":
        cad = open_any(project.source, allow_oda=allow_oda)
        changed = bool(project.source_sha256 and project.source_sha256 != sha256(project.source))
        project.source_sha256 = sha256(project.source)
        vp = pick_viewport(cad, project.layout, project.viewport)
        project.layout, project.viewport = vp.layout, vp.handle
        prof = None
        ppath = project.profile or (str(DEFAULT_PROFILE) if DEFAULT_PROFILE.exists() else None)
        if ppath:
            prof = Profile.load(ppath)
            project.profile = ppath
        s = cls(project, cad, vp, prof, None, changed)  # type: ignore[arg-type]
        s.rebuild()
        return s

    def rebuild(self):
        p = self.project
        self.scene = build_scene(self.cad, self.viewport, self.profile, p.manual, p.overrides, p.units_to_m)
        if p.overrides:
            p.overrides, self.override_warnings = match_overrides(p.overrides, self.scene)
            self.scene = build_scene(self.cad, self.viewport, self.profile, p.manual, p.overrides, p.units_to_m)
        return self.scene

    def select_viewport(self, layout: str, handle: str | None = None):
        self.viewport = pick_viewport(self.cad, layout, handle)
        self.project.layout, self.project.viewport = self.viewport.layout, self.viewport.handle
        self.rebuild()

    def report(self):
        groups = build_report(self.scene.entities, self.scene.assignments)
        return groups, summary(groups)

    def render(self, preview_scale: float = 1.0) -> RenderResult:
        return get_backend().render(self.scene, self.project.render, preview_scale)

    def set_override(self, handle: str, **kw):
        ov = self.project.overrides.setdefault(handle, {})
        ov.update(kw)
        o = next((o for o in self.scene.objects if o.id == handle), None)
        if o:
            ov["_fp"] = o.fingerprint
        self.rebuild()
