"""Экспорт (EXP-01…06): PNG/PDF с физическим размером и DPI, DXF-комплект с IMAGE в листе.
Исходный чертёж никогда не изменяется."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import ezdxf

from ..render.procedural import RenderResult
from ..scene.graph import Scene

APPID = "LANDSCAPE_RENDERER"
LAYER = "LANDSCAPE_RENDER"


def save_png(res: RenderResult, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    res.image.save(path, dpi=(res.dpi, res.dpi))
    return str(path)


def save_pdf(res: RenderResult, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    res.image.convert("RGB").save(path, "PDF", resolution=float(res.dpi))
    return str(path)


def save_dxf_bundle(res: RenderResult, scene: Scene, source_dxf: str, out_dir: str | Path,
                    name: str = "presentation", project_id: str = "") -> dict:
    """Копия DXF + PNG рядом; IMAGE в paper space на границах видового экрана (с полями отрисовки)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    save_png(res, png)
    dst = out / f"{name}.dxf"
    shutil.copy2(source_dxf, dst)
    doc = ezdxf.readfile(dst)
    if APPID not in doc.appids:
        doc.appids.new(APPID)
    if LAYER not in doc.layers:
        doc.layers.add(LAYER, color=8)
    vp = scene.viewport
    if vp.layout == "Model":
        layout = doc.modelspace()
        k = vp.scale_den / 1000.0 / scene.units_to_m       # мм листа → единицы модели
        x0, y0, x1, y1 = vp.window
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    else:
        layout = doc.layouts.get(vp.layout)
        k = 1.0
        cx, cy = vp.paper_center_mm
    # удалить ранее вставленные нами изображения (обновление без дублей)
    for e in list(layout.query("IMAGE")):
        if e.has_xdata(APPID):
            layout.delete_entity(e)
    w_mm, h_mm = res.sheet_mm
    vw, vh = res.viewport_mm
    ox, oy = res.origin_mm
    # левый нижний угол изображения на листе
    ins = (cx - vw / 2 * k - ox * k, cy - vh / 2 * k - (h_mm - vh - oy) * k)
    idef = doc.add_image_def(filename=png.name, size_in_pixel=res.image.size)
    img = layout.add_image(idef, insert=ins, size_in_units=(w_mm * k, h_mm * k), rotation=0,
                           dxfattribs={"layer": LAYER})
    img.set_xdata(APPID, [(1000, "presentation"), (1000, project_id or ""), (1000, png.name)])
    doc.saveas(dst)
    readme = out / "README.txt"
    readme.write_text(
        "Комплект подачи Landscape Renderer\n"
        f"- {dst.name}: копия исходного чертежа; подача вставлена на слой {LAYER} листа «{vp.layout}».\n"
        f"- {png.name}: изображение подачи. IMAGE в DXF ссылается на этот файл — передавайте их вместе.\n"
        "Исходный чертёж не изменялся.\n", encoding="utf-8")
    meta = {"dxf": str(dst), "png": str(png), "layout": vp.layout, "insert": ins,
            "size": (w_mm * k, h_mm * k), "dpi": res.dpi}
    (out / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
