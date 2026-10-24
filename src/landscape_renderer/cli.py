"""Командная строка: landscape-renderer layouts | report | render."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .cad import UNIT_NAMES, open_any
from .export.writers import save_dxf_bundle, save_pdf, save_png
from .mapping import CLASSES
from .pipeline import Session
from .project import Project, RenderSettings
from .render import STYLE_KEYS, STYLE_RU


def _project(a) -> Project:
    if a.source.lower().endswith(".lscapeproj"):
        p = Project.load(a.source)
    else:
        p = Project(source=a.source, layout=a.layout, viewport=a.viewport, profile=a.profile)
    if getattr(a, "style", None):
        p.render.style = a.style
    if getattr(a, "dpi", None):
        p.render.dpi = a.dpi
    if getattr(a, "seed", None) is not None:
        p.render.seed = a.seed
    return p


def cmd_layouts(a):
    cad = open_any(a.source, allow_oda=a.allow_oda)
    print(f"Файл: {a.source}\nЕдиницы: {UNIT_NAMES.get(cad.insunits, cad.insunits)}  объектов: {len(cad.entities)}")
    for lay in cad.layouts:
        print(f"Лист «{lay.name}»")
        for vp in lay.viewports:
            print(f"  {vp.label}  окно {tuple(round(v, 2) for v in vp.window)}")
    for w in cad.warnings:
        print("! " + w)


def cmd_report(a):
    s = Session.open(_project(a), allow_oda=a.allow_oda)
    groups, summ = s.report()
    print(f"Лист «{s.viewport.layout}», {s.viewport.label}; профиль: {s.profile.name if s.profile else '—'}")
    print(f"подтверждено {summ['confirmed']} · предложено {summ['suggested']} · не сопоставлено {summ['unmapped']}")
    mark = {"confirmed": "✓", "suggested": "?", "unmapped": "✗"}
    for g in groups:
        cls = CLASSES.get(g.cls, g.cls or "—")
        st = ", ".join(f"{k}:{v}" for k, v in g.statuses.items() if k != "existing")
        print(f" {mark[g.state]} {g.kind:5s} {g.name:28s} ×{g.count:<3d} → {cls}{'  [' + st + ']' if st else ''}")
        for c in sorted(set(g.conflicts)):
            print(f"     ! конфликт: {c}")


def cmd_render(a):
    t = time.time()
    s = Session.open(_project(a), allow_oda=a.allow_oda)
    res = s.render()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(s.project.source).stem + "_" + s.project.render.style
    done = [save_png(res, out / f"{stem}.png")]
    if a.pdf:
        done.append(save_pdf(res, out / f"{stem}.pdf"))
    if a.dxf_bundle:
        src = s.cad.path
        meta = save_dxf_bundle(res, s.scene, src, out / f"{stem}_bundle", name=stem, project_id=s.project.project_id)
        done += [meta["dxf"], meta["png"]]
    if a.save_project:
        done.append(s.project.save(str(out / f"{Path(s.project.source).stem}.lscapeproj")))
    print(f"Готово за {time.time() - t:.1f} с: {s.viewport.label}, {res.stats['px'][0]}×{res.stats['px'][1]} px")
    for d in done:
        print("  " + d)
    if s.source_changed:
        print("! Источник изменился с прошлого сохранения проекта.")
    for w in s.override_warnings or []:
        print("! " + w)


def main(argv=None):
    ap = argparse.ArgumentParser("landscape-renderer", description="Ручная подача планов ландшафтного дизайна")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(required=True)

    def common(p):
        p.add_argument("source", help="DXF / DWG или проект .lscapeproj")
        p.add_argument("--layout")
        p.add_argument("--viewport")
        p.add_argument("--profile", help="профиль сопоставления (.json)")
        p.add_argument("--allow-oda", action="store_true",
                       help="разрешить ODA File Converter (без членства ODA — только некоммерческое использование)")

    p = sub.add_parser("layouts", help="листы и видовые экраны")
    p.add_argument("source")
    p.add_argument("--allow-oda", action="store_true")
    p.set_defaults(fn=cmd_layouts)
    p = sub.add_parser("report", help="отчёт сопоставления")
    common(p)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("render", help="построить подачу")
    common(p)
    p.add_argument("--style", choices=STYLE_KEYS, help=", ".join(f"{k} — {v}" for k, v in STYLE_RU.items()))
    p.add_argument("--dpi", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--out", default="export")
    p.add_argument("--pdf", action="store_true")
    p.add_argument("--dxf-bundle", action="store_true", help="копия DXF с IMAGE в листе + PNG")
    p.add_argument("--save-project", action="store_true")
    p.set_defaults(fn=cmd_render)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except Exception as e:  # понятная ошибка вместо трассировки
        if "--debug" in sys.argv:
            raise
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
