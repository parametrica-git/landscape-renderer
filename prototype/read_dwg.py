"""Наша программа: открывает DWG/DXF, берёт выбранный лист и видовой экран,
по правилам «слой/блок -> элемент» строит сцену и рендерит ручную подачу.

DWG читается через ODA File Converter (DWG->DXF) на машине пользователя; здесь — DXF."""
import sys, math, time
import numpy as np
import ezdxf
from PIL import Image, ImageDraw, ImageFont
import garden as G
from garden import (LeafCrown, Hedge, Bed, Pond, Flat, PAL, ruled, stipple, hand_line, Watercolor)
from engine import Sheet, STYLES, F, Pen
from objects import Crown, Solid, grid
from render import FONT, STYLE_RU

# --------------------------------------------------------------- правила (как шаблон стилей Impression)
RULES_LAYER = {"ДОМ": "house", "ТЕРРАСА": "terrace", "НАСТИЛ": "deck", "ГАЗОН": "lawn", "ГРАВИЙ": "gravel",
               "ВОДОЁМ": "pond", "КАМНИ": "boulder", "МАФ": "maf", "МЕБЕЛЬ": "furniture",
               "ЖИВАЯ_ИЗГОРОДЬ": "hedge", "ГРАНИЦА": "boundary"}
RULES_BLOCK = {"ДЕРЕВО": "decid", "ХВОЙНОЕ": "spruce", "МОЖЖЕВЕЛЬНИК": "column", "КУСТ": "shrub",
               "ПЛИТА_ШАГ": "stone"}
PLANT_BY_RU = {"лаванда": "lavender", "шалфей": "salvia", "эхинацея": "echinacea", "рудбекия": "rudbeckia",
               "гаура": "gaura", "хоста": "hosta", "злаки": "grass"}
SPECIES_PAL = {"клён": "leaf", "липа": "leaf2", "ирга": "copper", "спирея": "leaf3", "барбарис": "leaf",
               "чубушник": "leaf2"}


def pal_for(vid, default):
    for k, v in SPECIES_PAL.items():
        if k in (vid or "").lower():
            return v
    return default


class PolySolid(Solid):
    """Мебель/МАФ: заливка + контур; низкие — тени на них падают."""

    def __init__(self, sh, polys_px, height, color, seed, low=True):
        self.sh, self.rs = sh, np.random.default_rng(seed)
        self.polys, self.height, self.color, self.low = polys_px, height, color, low

    def prepare(self):
        self.bb_, _ = self.prep_masks(self.polys)
        if self.low:
            s = self.sh.sl(self.bb_)
            m = self.sh.poly_mask(self.polys, self.bb_)
            self.sh.top[s] = np.clip(self.sh.top[s] - m, 0, 1)

    def draw(self, st):
        sh, bb = self.sh, self.bb_
        m = sh.poly_mask(self.polys, bb)
        if not isinstance(st, Pen):
            st.fill(sh, m, bb, self.color, 0, self.rs, angle=0.3)
        if st.line:
            for p in self.polys:
                hand_line(sh, np.vstack([p, p[:2]]), st.line_w * 0.9, self.rs)


def load(path, layout_name=None):
    doc = ezdxf.readfile(path)
    names = [l.name for l in doc.layouts if l.name != "Model"]
    layout = doc.layouts.get(layout_name or names[0])
    vps = [v for v in layout.query("VIEWPORT") if v.dxf.id > 1]
    vp = vps[0]
    cx, cy = vp.dxf.view_center_point.x, vp.dxf.view_center_point.y
    vh_m = vp.dxf.view_height
    w_mm, h_mm = vp.dxf.width, vp.dxf.height
    vw_m = vh_m * w_mm / h_mm
    scale_den = round(vh_m * 1000 / h_mm)           # 1:N
    win = (cx - vw_m / 2, cy - vh_m / 2, cx + vw_m / 2, cy + vh_m / 2)
    return doc, layout.name, win, (w_mm, h_mm), scale_den


def scene_from_dxf(doc, win, sh, rs):
    x0, y0, x1, y1 = win
    tw = lambda pts: [(float(x) - x0, y1 - float(y)) for x, y in pts]          # CAD -> мир программы
    px = sh.px
    msp = doc.modelspace()
    geo = {k: [] for k in ("house", "terrace", "deck", "lawn", "gravel", "pond", "boulder", "maf", "furniture",
                           "hedge", "boundary", "stone", "drift", "tree")}
    stats = {}
    for e in msp:
        lay, t = e.dxf.layer, e.dxftype()
        stats[(t, lay)] = stats.get((t, lay), 0) + 1
        if t == "LWPOLYLINE":
            pts = tw(e.get_points("xy"))
            if lay.startswith("ЦВ_"):
                geo["drift"].append((PLANT_BY_RU.get(lay[3:], "salvia"), pts))
            elif lay in RULES_LAYER:
                kind = RULES_LAYER[lay]
                if kind == "hedge":
                    geo["hedge"].append((pts, float(e.dxf.const_width or 0.8)))
                else:
                    geo[kind].append(pts)
        elif t == "INSERT" and e.dxf.name in RULES_BLOCK:
            kind = RULES_BLOCK[e.dxf.name]
            ins = e.dxf.insert
            if kind == "stone":
                for v in e.virtual_entities():
                    if v.dxftype() == "LWPOLYLINE":
                        geo["stone"].append(tw(v.get_points("xy")))
                continue
            attrs = {a.dxf.tag: a.dxf.text for a in e.attribs}
            D = float(attrs.get("D", e.dxf.xscale * 2))
            geo["tree"].append((kind, tw([(ins.x, ins.y)])[0], D, attrs.get("ВИД", "")))
    return geo, stats


def build_from_geo(sh, geo, rs):
    px = sh.px
    M = lambda polys: sh.poly_mask([px(p) for p in polys]) if polys else np.zeros((sh.H, sh.W), F)
    bnd = geo["boundary"][0] if geo["boundary"] else None
    sh.site = M([bnd]) if bnd else np.ones((sh.H, sh.W), F)
    m_lawn, m_grav, m_pond = M(geo["lawn"]), M(geo["gravel"]), M(geo["pond"])
    m_st = M(geo["stone"])
    m_grav = np.clip(m_grav - m_pond - m_st, 0, 1)
    m_lawn = np.clip(m_lawn - m_st, 0, 1)
    plants = []
    k_tree = 0
    for kind, xy, D, vid in geo["tree"]:
        k_tree += 1
        if kind == "decid":
            plants.append(LeafCrown(sh, "decid", xy, D, 500 + k_tree, pal=pal_for(vid, "leaf"),
                                    dab_m=0.18 if D < 3.5 else None))
        elif kind == "shrub":
            plants.append(LeafCrown(sh, "shrub", xy, D, 500 + k_tree, pal=pal_for(vid, "leaf3"), dab_m=0.14,
                                    branches=False))
        else:
            plants.append(Crown(sh, kind, xy, D, 500 + k_tree))
    excl = np.zeros((sh.H, sh.W), F)
    for c in plants:
        bbx = c.bb()
        X, Y = grid(bbx)
        s = sh.sl(bbx)
        excl[s] = np.maximum(excl[s], np.clip(c.sd(X, Y) / 1.5 + 0.5, 0, 1))
    drifts = [(M([p]), typ) for typ, p in geo["drift"]]
    m_bed = np.clip(sum(d for d, _ in drifts), 0, 1) if drifts else np.zeros((sh.H, sh.W), F)
    beds = Bed(sh, m_bed, [(None, typ) for _, typ in drifts], 340, exclude=excl)
    beds.drifts = drifts
    hedges = [Hedge(sh, np.array(p, F), w, 330 + i) for i, (p, w) in enumerate(geo["hedge"])]
    flats = [Flat(sh, geo["terrace"], "terrace", 370)] if geo["terrace"] else []
    if geo["deck"]:
        d = np.array(geo["deck"][0], F)
        wide = (d[:, 0].ptp() if hasattr(d[:, 0], "ptp") else np.ptp(d[:, 0])) > np.ptp(d[:, 1])
        flats.append(Flat(sh, geo["deck"], "deck", 371, board_angle=math.radians(0 if wide else 90)))
    if geo["stone"]:
        flats.append(Flat(sh, [np.array(s, F) for s in geo["stone"]], "stone", 372))
    if geo["boulder"]:
        flats.append(Flat(sh, [np.array(b, F) for b in geo["boulder"]], "stone", 360))
    flats += [Pond(sh, np.array(p, F), 373) for p in geo["pond"]]

    def ground(sh, st, rs):
        full = sh.full()
        st.fill(sh, m_lawn, full, PAL["lawn"][0], 0, rs, angle=math.radians(-38))
        st.fill(sh, m_grav, full, PAL["gravel"][0], 0, rs)
        beds.ground(st, rs)
        for f in flats:
            f.ground(st, rs)
        if st.line:
            stipple(sh, m_grav, full, rs, per_mm2=1.6, r_mm=0.09, alpha=0.55)
            stipple(sh, m_bed, full, rs, per_mm2=0.25, r_mm=0.08, alpha=0.4)
            for p in geo["lawn"]:
                q = px(p)
                hand_line(sh, np.vstack([q, q[:3]]), st.line_w, rs)
            for p in geo["gravel"]:
                q = px(p)
                hand_line(sh, np.vstack([q, q[:3]]), st.line_w * 0.6, rs, 0.7)
            for p in geo["house"]:
                ruled(sh, px(p), st.line_w * 3.0, rs, over_mm=1.2)
            if bnd:
                ruled(sh, px(bnd), st.line_w * 0.8, rs, 0.6)

    class House(Solid):
        height = 6.0

        def __init__(s, poly):
            s.sh, s.rs, s.poly = sh, np.random.default_rng(9), px(poly)

        def prepare(s):
            s.bb_, _ = s.prep_masks([s.poly])

        def draw(s, st):
            m = sh.poly_mask([s.poly], s.bb_)
            if isinstance(st, Pen):
                st.fill(sh, m, s.bb_, None, 1, s.rs, angle=0.0)
            else:
                st.fill(sh, m, s.bb_, (0.88, 0.87, 0.86), 0, s.rs, angle=0.0)

    objs = [House(p) for p in geo["house"]]
    if geo["furniture"]:
        objs.append(PolySolid(sh, [px(p) for p in geo["furniture"]], 0.75, (0.90, 0.90, 0.92), 380))
    for i, p in enumerate(geo["maf"]):
        objs.append(PolySolid(sh, [px(p)], 0.45, PAL["wood"][0], 381 + i))
    objs += [beds] + hedges
    order = {"column": 0, "shrub": 1, "spruce": 2, "decid": 3}
    objs += sorted(plants, key=lambda c: (order.get(c.kind, 3), c.R))
    return objs, ground


def process(path, style_key, layout_name=None, dpi=220, seed=21):
    t0 = time.time()
    doc, lname, win, (w_mm, h_mm), N = load(path, layout_name)
    sh = Sheet(w_mm + 20, h_mm + 28, dpi=dpi, scale=N, seed=seed)
    rs = np.random.default_rng(seed)
    geo, stats = scene_from_dxf(doc, win, sh, rs)
    objs, ground = build_from_geo(sh, geo, rs)
    st = STYLES[style_key]()
    for o in objs:
        o.prepare()
    ground(sh, st, rs)
    st.shadow(sh, sh.shad * (1 - sh.top) * sh.site, sh.full(), rs)
    for o in objs:
        o.draw(st)
    img = sh.finish(st)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, int(sh.mm(3.2)))
    y0 = sh.mm(10) + h_mm * sh.ppmm + sh.mm(5)
    d.text((sh.mm(10), y0), f"{path.split('/')[-1]} · лист «{lname}» · М 1:{N} · {STYLE_RU[style_key]}",
           fill=(55, 55, 58), font=f)
    n = sum(len(v) for v in geo.values())
    print(f"{style_key}: лист «{lname}», окно {[round(v, 2) for v in win]}, М 1:{N}, объектов {n}, "
          f"{time.time() - t0:.1f} c")
    return im, geo


if __name__ == "__main__":
    import os; os.makedirs("out", exist_ok=True)
    path = sys.argv[1] if len(sys.argv) > 1 else "../tests/fixtures/dxf/simple_site.dxf"
    for k in sys.argv[2:] or ["pero_akvarel", "pero_marker"]:
        im, geo = process(path, k)
        im.save(f"out/from_dwg_{k}.jpg", quality=90)
