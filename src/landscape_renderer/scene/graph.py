"""Граф сцены (REN-01): объекты в метрах окна видового экрана (ось Y вниз, начало — левый верхний угол),
с происхождением, семантическим классом, проектным статусом и переопределениями."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..cad.model import CadDocument, CadEntity, ViewportInfo
from ..mapping.profile import Assignment, Profile, assign

Pt = tuple[float, float]


@dataclass
class SceneObject:
    id: str                      # handle источника
    cls: str
    state: str                   # статус сопоставления
    status: str = "existing"     # проектный статус
    polys: list[list[Pt]] = field(default_factory=list)
    center: Pt | None = None
    diameter_m: float | None = None
    height_m: float | None = None
    width_m: float | None = None
    plant: str | None = None
    text: str | None = None
    text_height_m: float | None = None
    rotation: float = 0.0
    layer: str = ""
    block: str | None = None
    attribs: dict = field(default_factory=dict)
    hidden: bool = False
    variant: int = 0
    fingerprint: dict = field(default_factory=dict)


@dataclass
class Scene:
    viewport: ViewportInfo
    units_to_m: float
    size_m: Pt
    objects: list[SceneObject]
    assignments: list[Assignment]
    entities: list[CadEntity]

    def of(self, *prefixes: str) -> list[SceneObject]:
        return [o for o in self.objects if not o.hidden and any(o.cls.startswith(p) for p in prefixes)]


def _float(s, default=None):
    try:
        return float(str(s).replace(",", "."))
    except (TypeError, ValueError):
        return default


def fingerprint(e: CadEntity) -> dict:
    """Составной отпечаток для переноса переопределений (OBJ-01): handle ненадёжен после конвертаций."""
    pts = e.points or ([e.insert] if e.insert else [])
    if pts:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        c = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        ext = (max(xs) - min(xs), max(ys) - min(ys))
    else:
        c, ext = (0.0, 0.0), (0.0, 0.0)
    return {"handle": e.handle, "type": e.dxftype, "layer": e.layer, "block": e.block,
            "attribs": dict(e.attribs), "center": [round(c[0], 3), round(c[1], 3)],
            "extent": [round(ext[0], 3), round(ext[1], 3)]}


def build_scene(cad: CadDocument, vp: ViewportInfo, profile: Profile | None,
                manual: dict[str, str] | None = None, overrides: dict | None = None,
                units_to_m: float | None = None) -> Scene:
    u = units_to_m or cad.units_to_m or 1.0
    x0, y0, x1, y1 = vp.window
    frozen = set(vp.frozen_layers)

    def W(pts):
        return [((x - x0) * u, (y1 - y) * u) for x, y in pts]

    overrides = overrides or {}
    objs, assigns, ents = [], [], []
    for e in cad.entities:
        lay = cad.layers.get(e.layer)
        if lay and (not lay.on or lay.frozen) or e.layer in frozen:
            continue
        a = assign(e, profile, manual)
        assigns.append(a)
        ents.append(e)
        if a.cls in (None, "ignore"):
            continue
        p = a.params
        o = SceneObject(id=e.handle, cls=a.cls, state=a.state, status=a.status, layer=e.layer,
                        block=e.block, attribs=dict(e.attribs), fingerprint=fingerprint(e),
                        height_m=p.get("height_m"))
        if e.dxftype == "INSERT":
            o.center = W([e.insert])[0]
            o.rotation = e.rotation
            dattr = p.get("diameter_from_attribute")
            D = _float(e.attribs.get(dattr)) if dattr else None
            if D is not None and p.get("diameter_unit", "m") == "mm":
                D /= 1000.0
            if D is None:
                D = 2.0 * abs(e.xscale) * u
            o.diameter_m = D
            pattr = p.get("plant_from_attribute")
            o.plant = e.attribs.get(pattr) if pattr else None
            o.polys = [W(part) for part in e.parts]
        elif e.dxftype in ("TEXT", "MTEXT"):
            o.center = W([e.insert])[0]
            o.text = e.text
            o.text_height_m = (e.text_height or 0.2) * u
            o.rotation = e.rotation
        else:
            parts = e.parts or ([e.points] if e.points else [])
            o.polys = [W(part) for part in parts if len(part) >= 2]
            if p.get("width_from") == "const_width":
                o.width_m = (e.width or 0.8) * u
            if p.get("plant_from_layer_suffix") and "_" in e.layer:
                o.plant = e.layer.split("_", 1)[1]
        ov = overrides.get(e.handle)
        if ov:
            o.hidden = bool(ov.get("hidden", False))
            o.variant = int(ov.get("variant", 0))
            if "status" in ov:
                o.status = ov["status"]
            if "diameter_m" in ov:
                o.diameter_m = float(ov["diameter_m"])
            if "cls" in ov:
                o.cls = ov["cls"]
        objs.append(o)
    size = ((x1 - x0) * u, (y1 - y0) * u)
    return Scene(viewport=vp, units_to_m=u, size_m=size, objects=objs, assignments=assigns, entities=ents)


def match_overrides(old: dict[str, dict], scene: Scene, tol_m: float = 0.3) -> tuple[dict, list[str]]:
    """Перенос переопределений на новую версию чертежа. Возвращает (новые overrides, предупреждения)."""
    by_handle = {o.id: o for o in scene.objects}
    out, warn = {}, []
    for h, ov in old.items():
        if h in by_handle:
            out[h] = ov
            continue
        fp = ov.get("_fp")
        if not fp:
            warn.append(f"Объект {h} исчез из чертежа; переопределение не перенесено.")
            continue
        cands = [o for o in scene.objects if o.fingerprint.get("type") == fp.get("type")
                 and o.fingerprint.get("layer") == fp.get("layer") and o.fingerprint.get("block") == fp.get("block")
                 and math.dist(o.fingerprint["center"], fp["center"]) <= tol_m / scene.units_to_m]
        if len(cands) == 1:
            out[cands[0].id] = ov
            warn.append(f"Переопределение {h} перенесено на {cands[0].id} по отпечатку.")
        else:
            warn.append(f"Объект {h}: {'несколько кандидатов' if cands else 'нет кандидатов'} — нужна перепривязка.")
    return out, warn
