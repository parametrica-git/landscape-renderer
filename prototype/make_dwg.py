"""Создаёт исходный CAD-файл сада (DXF R2018; в AutoCAD открывается и сохраняется как DWG).

Как в реальном проекте ландшафтного дизайна:
- слои по элементам (ДОМ, ТЕРРАСА, НАСТИЛ, ГАЗОН, ГРАВИЙ, ВОДОЁМ, ЦВ_<растение>, ...)
- растения — блоки с атрибутами ВИД и D (диаметр кроны), масштаб блока = радиус
- лист «Генплан 1-100» (A3) с видовым экраном в масштабе 1:100
"""
import math
import numpy as np
import cv2
import ezdxf
from ezdxf.enums import TextEntityAlignment
import garden as G
from engine import Sheet, F
from objects import grid

TAU = 2 * math.pi
H = G.H_M


def cad_xy(p):          # мир программы (y вниз) -> CAD (y вверх)
    p = np.asarray(p, float)
    return [(float(x), float(H - y)) for x, y in p.reshape(-1, 2)]


def drift_polys(sh):
    """Контуры посадочных пятен (в метрах) — то, что дизайнер рисует полилиниями."""
    g = sh.geo
    bed = g["beds"]
    m = g["m_bed"]
    ys, xs = np.nonzero(m > 0.5)
    bb = sh.bbox(xs.min() - 20, ys.min() - 20, xs.max() + 20, ys.max() + 20)
    s_ = sh.sl(bb)
    X, Y = grid(bb)
    seeds = np.array([sh.px(p) for p, _ in bed.seeds_types], F)
    D = np.stack([np.hypot(X - x, Y - y) for x, y in seeds])
    lab = np.argmin(D, 0)
    org = np.array([sh.mm(10), sh.mm(10)], F)
    out = []
    for i, (_, typ) in enumerate(bed.seeds_types):
        dm = ((lab == i) * (m[s_] > 0.5)).astype(np.uint8)
        k = np.ones((int(sh.mm(3)) | 1,) * 2, np.uint8)
        dm = cv2.morphologyEx(cv2.morphologyEx(dm, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        cnt, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for ct in cnt:
            if cv2.contourArea(ct) > sh.ppm ** 2 * 0.3:
                ct = cv2.approxPolyDP(ct, sh.mm(0.5), True)[:, 0, :].astype(F)
                pts = (ct + np.array([bb[0], bb[1]], F) - org) / sh.ppm
                out.append((typ, pts))
    return out


PLANT_RU = {"lavender": "лаванда", "salvia": "шалфей", "echinacea": "эхинацея", "rudbeckia": "рудбекия",
            "gaura": "гаура", "hosta": "хоста", "grass": "злаки"}
TREE_RU = {"leaf": "клён остролистный", "leaf2": "липа мелколистная", "copper": "ирга канадская",
           "leaf3": "спирея", "hedge": "туя западная"}


def make(path="../tests/fixtures/dxf/simple_site.dxf"):
    sh = Sheet(G.W_M * 10 + 20, G.H_M * 10 + 28, dpi=150, scale=100, seed=21)
    rs = np.random.default_rng(21)
    G.build(sh, rs)
    g = sh.geo

    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.M
    doc.header["$INSUNITS"] = 6
    for name, col in [("ДОМ", 7), ("ТЕРРАСА", 8), ("НАСТИЛ", 30), ("ГАЗОН", 3), ("ГРАВИЙ", 9), ("ВОДОЁМ", 5),
                      ("ДОРОЖКА_ПЛИТЫ", 8), ("КАМНИ", 8), ("ДЕРЕВЬЯ", 94), ("ХВОЙНЫЕ", 96), ("КУСТАРНИКИ", 82),
                      ("ЖИВАЯ_ИЗГОРОДЬ", 104), ("МАФ", 40), ("МЕБЕЛЬ", 42), ("ГРАНИЦА", 1), ("ТЕКСТ", 7)] + \
            [("ЦВ_" + v, c) for v, c in zip(PLANT_RU.values(), (200, 210, 230, 50, 254, 112, 42))]:
        doc.layers.add(name, color=col)
    doc.styles.add("ГОСТ", font="isocpeur.ttf")

    # ---------- блоки растений (единичный радиус; масштаб вставки = радиус кроны, м)
    b = doc.blocks.new("ДЕРЕВО")
    b.add_circle((0, 0), 1.0)
    b.add_line((-0.25, 0), (0.25, 0)); b.add_line((0, -0.25), (0, 0.25))
    b.add_circle((0, 0), 0.06)
    for k in range(12):
        t = k * TAU / 12
        b.add_line((math.cos(t) * 0.86, math.sin(t) * 0.86), (math.cos(t), math.sin(t)))
    for tag, y in (("ВИД", 1.15), ("D", -1.3)):
        b.add_attdef(tag, (0.8, y), dxfattribs={"height": 0.18, "style": "ГОСТ"})
    b = doc.blocks.new("ХВОЙНОЕ")
    pts = []
    for k in range(36):
        t = k * TAU / 36
        r = 1.0 if k % 2 == 0 else 0.8
        pts.append((math.cos(t) * r, math.sin(t) * r))
    b.add_lwpolyline(pts, close=True)
    for k in range(18):
        t = k * TAU / 18
        b.add_line((0, 0), (math.cos(t) * 0.95, math.sin(t) * 0.95))
    for tag, y in (("ВИД", 1.15), ("D", -1.3)):
        b.add_attdef(tag, (0.8, y), dxfattribs={"height": 0.18, "style": "ГОСТ"})
    for name in ("КУСТ", "МОЖЖЕВЕЛЬНИК"):
        b = doc.blocks.new(name)
        b.add_circle((0, 0), 1.0)
        b.add_line((-0.3, 0), (0.3, 0)); b.add_line((0, -0.3), (0, 0.3))
        for tag, y in (("ВИД", 1.15), ("D", -1.3)):
            b.add_attdef(tag, (0.8, y), dxfattribs={"height": 0.18, "style": "ГОСТ"})
    b = doc.blocks.new("ПЛИТА_ШАГ")
    b.add_lwpolyline([(math.cos(t) * 0.3 * (1 + 0.05 * math.sin(5 * t)), math.sin(t) * 0.3 * (1 + 0.05 * math.sin(5 * t)))
                      for t in np.linspace(0, TAU, 24, endpoint=False)], close=True)

    msp = doc.modelspace()

    def poly(pts, layer, closed=True, **kw):
        return msp.add_lwpolyline(cad_xy(pts), close=closed, dxfattribs={"layer": layer, **kw})

    def plant(block, layer, c, D, vid):
        x, y = cad_xy([c])[0]
        ins = msp.add_blockref(block, (x, y), dxfattribs={"layer": layer, "xscale": D / 2, "yscale": D / 2})
        ins.add_auto_attribs({"ВИД": vid, "D": f"{D:.1f}"})
        for a in ins.attribs:          # подписи фиксированной высоты, рядом с кроной
            a.dxf.height = 0.16
            dy = D / 2 * 0.75 + (0.22 if a.dxf.tag == "ВИД" else 0.0)
            a.dxf.insert = (x + D / 2 * 0.75, y + dy)
            a.dxf.layer = "ТЕКСТ"

    poly(g["house"], "ДОМ")
    hatch = msp.add_hatch(color=8, dxfattribs={"layer": "ДОМ"})
    hatch.set_pattern_fill("ANSI31", scale=0.05)
    hatch.paths.add_polyline_path(cad_xy(g["house"]), is_closed=True)
    poly(g["terrace"], "ТЕРРАСА")
    poly(g["deck"], "НАСТИЛ")
    poly(g["lawn"], "ГАЗОН")
    poly(g["gravel"], "ГРАВИЙ")
    poly(g["pond"], "ВОДОЁМ")
    for s in g["stones"]:
        c = s.mean(0)
        x, y = cad_xy([c])[0]
        msp.add_blockref("ПЛИТА_ШАГ", (x, y), dxfattribs={"layer": "ДОРОЖКА_ПЛИТЫ",
                                                           "rotation": float(rs.uniform(0, 360))})
    org = np.array([sh.mm(10), sh.mm(10)], F)
    for p in g["boulders"].polys:
        poly((p - org) / sh.ppm, "КАМНИ")
    for typ, pts in drift_polys(sh):
        lay = "ЦВ_" + PLANT_RU[typ]
        poly(pts, lay)
        c = pts.mean(0)
        x, y = cad_xy([c])[0]
        msp.add_text(PLANT_RU[typ], height=0.22, dxfattribs={"layer": "ТЕКСТ", "style": "ГОСТ"}).set_placement(
            (x, y), align=TextEntityAlignment.MIDDLE_CENTER)
    # растения
    world = lambda c: ((c.cx - org[0]) / sh.ppm, (c.cy - org[1]) / sh.ppm)
    for c in g["trees"]:
        palname = [k for k, v in G.PAL.items() if v is c.pal][0]
        plant("ДЕРЕВО", "ДЕРЕВЬЯ", world(c), c.R * 2 / sh.ppm, TREE_RU.get(palname, "дерево"))
    plant("ХВОЙНОЕ", "ХВОЙНЫЕ", world(g["spruce"]), g["spruce"].R * 2 / sh.ppm, "ель колючая")
    for c in g["junip"]:
        plant("МОЖЖЕВЕЛЬНИК", "ХВОЙНЫЕ", world(c), c.R * 2 / sh.ppm, "можжевельник скальный")
    for c in g["shrubs"]:
        palname = [k for k, v in G.PAL.items() if v is c.pal][0]
        plant("КУСТ", "КУСТАРНИКИ", world(c), c.R * 2 / sh.ppm,
              {"leaf3": "спирея", "leaf": "барбарис", "leaf2": "чубушник"}[palname])
    for h in g["hedge"]:
        pts = (h.line[:: max(1, len(h.line) // 40)] - org) / sh.ppm
        pl = msp.add_lwpolyline(cad_xy(pts), dxfattribs={"layer": "ЖИВАЯ_ИЗГОРОДЬ", "const_width": h.wpx / sh.ppm})
    bench = g["bench"]
    poly((bench.poly - org) / sh.ppm, "МАФ")
    fu = g["furn"]
    poly((fu.table - org) / sh.ppm, "МЕБЕЛЬ")
    for ch in fu.chairs:
        poly((ch - org) / sh.ppm, "МЕБЕЛЬ")
    poly([(0, 0), (G.W_M, 0), (G.W_M, G.H_M), (0, G.H_M)], "ГРАНИЦА")

    # ---------- лист A3 с видовым экраном 1:100
    lay = doc.layouts.get("Layout1")
    doc.layouts.rename("Layout1", "Генплан 1-100")
    lay.page_setup(size=(420, 297), margins=(10, 10, 10, 10), units="mm")
    vw, vh = 240.0, 170.0                    # размер видового экрана на листе, мм
    vp = lay.add_viewport(center=(150, 160), size=(vw, vh), view_center_point=(G.W_M / 2, G.H_M / 2),
                          view_height=vh / 10.0)   # 1:100: 1 мм листа = 0,1 м
    vp.dxf.status = 2
    lay.add_text("Сад. Генеральный план. М 1:100", height=5,
                 dxfattribs={"style": "ГОСТ"}).set_placement((30, 260))
    lay.add_lwpolyline([(20, 20), (400, 20), (400, 287), (20, 287)], close=True)
    doc.saveas(path)
    return path


if __name__ == "__main__":
    print(make())
