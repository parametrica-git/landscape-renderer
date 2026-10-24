"""Чтение DXF (ezdxf) в нейтральную модель. DWG — через конвертер (см. dwg.py)."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import ezdxf
from ezdxf import path as ezpath

from .model import CadDocument, CadEntity, LayerInfo, LayoutInfo, ViewportInfo

# $INSUNITS → метры
UNITS_TO_M = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 14: 0.1}
UNIT_NAMES = {0: "не заданы", 1: "дюймы", 2: "футы", 4: "мм", 5: "см", 6: "м", 14: "дм"}
FLATTEN = 0.02   # допуск аппроксимации дуг, единицы модели


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flat(entity, tol=FLATTEN) -> list[tuple[float, float]]:
    try:
        p = ezpath.make_path(entity)
        return [(v.x, v.y) for v in p.flattening(tol)]
    except Exception:
        return []


def _entity(e) -> CadEntity | None:
    t = e.dxftype()
    base = dict(handle=e.dxf.handle, dxftype=t, layer=e.dxf.layer)
    if t == "LWPOLYLINE":
        pts = _flat(e)
        if e.closed and pts and pts[0] != pts[-1]:
            pass
        width = float(e.dxf.const_width) if e.dxf.hasattr("const_width") else None
        return CadEntity(**base, points=pts, closed=bool(e.closed), width=width or None)
    if t in ("POLYLINE", "SPLINE", "ELLIPSE", "ARC", "LINE"):
        pts = _flat(e)
        closed = bool(getattr(e, "is_closed", False)) if t == "POLYLINE" else False
        return CadEntity(**base, points=pts, closed=closed)
    if t == "CIRCLE":
        return CadEntity(**base, points=_flat(e), closed=True)
    if t == "HATCH":
        parts = []
        for bp in e.paths:
            try:
                p = ezpath.from_hatch_boundary_path(bp)
                parts.append([(v.x, v.y) for v in p.flattening(FLATTEN)])
            except Exception:
                continue
        parts = [p for p in parts if len(p) >= 3]
        return CadEntity(**base, points=parts[0] if parts else [], closed=True, parts=parts)
    if t == "INSERT":
        ins = e.dxf.insert
        attribs = {a.dxf.tag: a.dxf.text for a in e.attribs}
        parts = []
        try:
            for v in e.virtual_entities():
                if v.dxftype() in ("LWPOLYLINE", "POLYLINE", "CIRCLE") and getattr(v, "closed", True):
                    pts = _flat(v)
                    if len(pts) >= 3:
                        parts.append(pts)
        except Exception:
            pass
        return CadEntity(**base, block=e.dxf.name, attribs=attribs, insert=(ins.x, ins.y),
                         rotation=float(e.dxf.rotation), xscale=float(e.dxf.xscale),
                         yscale=float(e.dxf.yscale), parts=parts)
    if t in ("TEXT", "MTEXT"):
        ins = e.dxf.insert
        txt = e.plain_text() if t == "MTEXT" else e.dxf.text
        h = float(e.dxf.char_height if t == "MTEXT" else e.dxf.height)
        return CadEntity(**base, insert=(ins.x, ins.y), text=txt, text_height=h,
                         rotation=float(e.dxf.get("rotation", 0.0)))
    return None


def read_dxf(path: str | Path) -> CadDocument:
    path = str(path)
    doc = ezdxf.readfile(path)
    warnings: list[str] = []
    insunits = int(doc.header.get("$INSUNITS", 0))
    units_to_m = UNITS_TO_M.get(insunits)
    if units_to_m is None:
        warnings.append(f"Единицы чертежа ($INSUNITS={insunits}) не заданы — подтвердите их вручную.")

    layers = {}
    for lay in doc.layers:
        layers[lay.dxf.name] = LayerInfo(name=lay.dxf.name, on=lay.is_on(), frozen=lay.is_frozen(),
                                         color=abs(int(lay.dxf.color)))

    layouts = []
    for lay in doc.layouts:
        if lay.name == "Model":
            continue
        vps = []
        for vp in lay.query("VIEWPORT"):
            if vp.dxf.id <= 1:
                continue        # первый VIEWPORT — сам лист
            c = vp.dxf.view_center_point
            vh = float(vp.dxf.view_height)
            w_mm, h_mm = float(vp.dxf.width), float(vp.dxf.height)
            if h_mm <= 0 or vh <= 0:
                continue
            vw = vh * w_mm / h_mm
            u = units_to_m or 1.0
            scale_den = round(vh * u * 1000.0 / h_mm, 3)
            frozen = []
            try:
                frozen = list(vp.frozen_layer_names)
            except Exception:
                pass
            twist = float(vp.dxf.get("view_twist_angle", 0.0))
            if abs(twist) > 1e-6:
                warnings.append(f"Видовой экран {vp.dxf.handle} повёрнут на {twist:.1f}° — поворот пока не поддержан.")
            pc = vp.dxf.center
            vps.append(ViewportInfo(handle=vp.dxf.handle, layout=lay.name, center=(c.x, c.y), view_height=vh,
                                    paper_size_mm=(w_mm, h_mm), paper_center_mm=(pc.x, pc.y),
                                    scale_den=scale_den,
                                    window=(c.x - vw / 2, c.y - vh / 2, c.x + vw / 2, c.y + vh / 2),
                                    frozen_layers=frozen, twist_deg=twist))
        paper = None
        try:
            paper = (float(lay.dxf_layout.dxf.paper_width), float(lay.dxf_layout.dxf.paper_height))
        except Exception:
            pass
        layouts.append(LayoutInfo(name=lay.name, paper_size_mm=paper, viewports=vps))
    if not any(l.viewports for l in layouts):
        warnings.append("В листах нет видовых экранов — будет использована вся модель.")

    entities, skipped = [], {}
    for e in doc.modelspace():
        ce = _entity(e)
        if ce is None:
            skipped[e.dxftype()] = skipped.get(e.dxftype(), 0) + 1
            continue
        entities.append(ce)
    for t, n in sorted(skipped.items()):
        warnings.append(f"Не поддержано: {t} × {n}")
    return CadDocument(path=path, sha256=sha256(path), insunits=insunits, units_to_m=units_to_m,
                       layers=layers, layouts=layouts, entities=entities, warnings=warnings)


def model_extent_viewport(cad: CadDocument, margin: float = 0.02) -> ViewportInfo:
    """Если видовых экранов нет — окно по габаритам модели в масштабе, близком к 1:100."""
    xs, ys = [], []
    for e in cad.entities:
        for x, y in e.points or ([e.insert] if e.insert else []):
            xs.append(x); ys.append(y)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy
    u = cad.units_to_m or 1.0
    den = 100.0
    w_mm, h_mm = (x1 - x0) * u * 1000 / den, (y1 - y0) * u * 1000 / den
    return ViewportInfo(handle="MODEL", layout="Model", center=((x0 + x1) / 2, (y0 + y1) / 2),
                        view_height=y1 - y0, paper_size_mm=(w_mm, h_mm), paper_center_mm=(w_mm / 2, h_mm / 2),
                        scale_den=den, window=(x0, y0, x1, y1))


def pick_viewport(cad: CadDocument, layout: str | None = None, viewport: str | None = None) -> ViewportInfo:
    lays = [l for l in cad.layouts if l.viewports]
    if not lays:
        return model_extent_viewport(cad)
    lay = cad.layout(layout) if layout else lays[0]
    if not lay.viewports:
        raise ValueError(f"На листе «{lay.name}» нет видовых экранов")
    if viewport:
        for vp in lay.viewports:
            if vp.handle == viewport:
                return vp
        raise KeyError(f"Видовой экран {viewport} не найден на листе «{lay.name}»")
    return max(lay.viewports, key=lambda v: v.paper_size_mm[0] * v.paper_size_mm[1])


def angle_norm(a: float) -> float:
    return math.degrees(math.atan2(math.sin(math.radians(a)), math.cos(math.radians(a))))
