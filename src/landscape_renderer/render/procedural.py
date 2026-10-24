"""Процедурный CPU-бэкенд (эталон корректности). Использует NPR-движок прототипа (render/npr)."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..project.model import RenderSettings
from ..scene.graph import Scene, SceneObject
from .npr import garden as G
from .npr.common import FONT
from .npr.engine import F, PAL, STYLES, Pen, Sheet, hand_line, stipple
from .npr.garden import Bed, Flat, Hedge, LeafCrown, Pond, ruled
from .npr.objects import Crown, Solid, grid

STYLE_KEYS = list(STYLES.keys())

PLANT_TYPES = {"лаванда": "lavender", "шалфей": "salvia", "эхинацея": "echinacea", "рудбекия": "rudbeckia",
               "гаура": "gaura", "хоста": "hosta", "злаки": "grass", "мискантус": "grass", "овсяница": "grass",
               "lavender": "lavender", "salvia": "salvia", "grass": "grass", "hosta": "hosta"}
SPECIES_PAL = {"клён": "leaf", "клен": "leaf", "липа": "leaf2", "ирга": "copper", "слива": "copper",
               "спирея": "leaf3", "барбарис": "copper", "чубушник": "leaf2", "берёза": "leaf2", "береза": "leaf2",
               "дуб": "leaf", "яблоня": "leaf3"}


def _stable(s: str, n: int) -> int:
    return int(hashlib.md5((s or "").encode("utf-8")).hexdigest(), 16) % n


def pal_for(name: str | None, default: str) -> str:
    low = (name or "").lower()
    for k, v in SPECIES_PAL.items():
        if k in low:
            return v
    return default


def bed_type(name: str | None) -> str:
    low = (name or "").lower()
    for k, v in PLANT_TYPES.items():
        if k in low:
            return v
    keys = ["salvia", "echinacea", "rudbeckia", "lavender", "hosta"]
    return keys[_stable(low, len(keys))]


class PolySolid(Solid):
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


class Building(Solid):
    def __init__(self, sh, poly_px, height, seed):
        self.sh, self.rs, self.poly, self.height = sh, np.random.default_rng(seed), poly_px, height

    def prepare(self):
        self.bb_, _ = self.prep_masks([self.poly])

    def draw(self, st):
        sh = self.sh
        m = sh.poly_mask([self.poly], self.bb_)
        if isinstance(st, Pen):
            st.fill(sh, m, self.bb_, None, 1, self.rs, angle=0.0)
        else:
            st.fill(sh, m, self.bb_, (0.88, 0.87, 0.86), 0, self.rs, angle=0.0)


@dataclass
class RenderResult:
    image: Image.Image
    sheet_mm: tuple[float, float]       # размер всего изображения, мм
    origin_mm: tuple[float, float]      # где на изображении левый верхний угол окна видового экрана
    viewport_mm: tuple[float, float]    # размер окна на листе
    dpi: int
    stats: dict


class ProceduralBackend:
    name = "cpu-procedural"
    MARGIN = 10.0          # = origin_mm по умолчанию в npr.engine.Sheet.px: объекты движка и сцены совпадают

    def render(self, scene: Scene, rs_: RenderSettings, preview_scale: float = 1.0) -> RenderResult:
        vp = scene.viewport
        w_mm, h_mm = vp.paper_size_mm
        dpi = max(40, int(rs_.dpi * preview_scale))
        shadow_deg = (rs_.light_azimuth_deg + 90.0) % 360.0
        sh = Sheet(w_mm + 2 * self.MARGIN, h_mm + 2 * self.MARGIN, dpi=dpi, scale=vp.scale_den, seed=rs_.seed,
                   shadow_deg=shadow_deg, shadow_k=rs_.shadow_k, light_elev=rs_.light_elevation_deg)
        rs = np.random.default_rng(rs_.seed)
        px = lambda pts: sh.px(pts, origin_mm=(self.MARGIN, self.MARGIN))  # noqa: E731
        sh.px_ = px
        objs, ground, after = self._build(sh, scene, px)
        st = STYLES[rs_.style]()
        for o in objs:
            o.prepare()
        ground(sh, st, rs)
        site = getattr(sh, "site", None)
        sm = sh.shad * (1 - sh.top) * (site if site is not None else 1)
        if sm.max() > 0.05:
            st.shadow(sh, sm, sh.full(), rs)
        for o in objs:
            o.draw(st)
        after(sh, st, rs, rs_)
        img = Image.fromarray(sh.finish(st))
        if rs_.annotations:
            self._annotations(img, sh, scene, px)
        stats = {"objects": len(scene.objects), "dpi": dpi, "px": img.size}
        return RenderResult(img, (w_mm + 2 * self.MARGIN, h_mm + 2 * self.MARGIN), (self.MARGIN, self.MARGIN),
                            (w_mm, h_mm), dpi, stats)

    # ---------------------------------------------------------------------------------------------
    def _build(self, sh, scene: Scene, px):
        o_px = lambda pts: px(np.asarray(pts, F))  # noqa: E731
        M = lambda polys: sh.poly_mask([o_px(p) for p in polys]) if polys else np.zeros((sh.H, sh.W), F)  # noqa
        polys = lambda *c: [p for o in scene.of(*c) for p in o.polys if len(p) >= 3]  # noqa: E731
        bnd = polys("boundary")
        sh.site = M(bnd[:1]) if bnd else None
        lawn, gravel, pond, stones = polys("ground.lawn"), polys("ground.gravel"), polys("water"), polys("hard.stone")
        m_lawn, m_grav, m_pond, m_st = M(lawn), M(gravel), M(pond), M(stones)
        m_grav = np.clip(m_grav - m_pond - m_st, 0, 1)
        m_lawn = np.clip(m_lawn - m_st, 0, 1)

        plants, crowns_status = [], []
        for i, o in enumerate(scene.of("vegetation.tree", "vegetation.shrub")):
            D = max(0.3, o.diameter_m or 2.0)
            seed = 500 + _stable(o.id, 10_000) + o.variant * 7919
            # перенос в координаты движка: движок работает с метрами относительно origin листа
            xy = o.center
            if o.cls == "vegetation.tree.deciduous":
                c = LeafCrown(sh, "decid", xy, D, seed, pal=pal_for(o.plant, "leaf"), dab_m=0.18 if D < 3.5 else None)
            elif o.cls == "vegetation.shrub":
                c = LeafCrown(sh, "shrub", xy, D, seed, pal=pal_for(o.plant, "leaf3"), dab_m=0.14, branches=False)
            elif o.cls == "vegetation.tree.conifer":
                c = Crown(sh, "spruce", xy, D, seed)
            else:
                c = Crown(sh, "column", xy, D, seed)
            plants.append(c)
            crowns_status.append((c, o.status))

        excl = np.zeros((sh.H, sh.W), F)
        for c in plants:
            bbx = c.bb()
            X, Y = grid(bbx)
            s = sh.sl(bbx)
            excl[s] = np.maximum(excl[s], np.clip(c.sd(X, Y) / 1.5 + 0.5, 0, 1))
        drifts = [(M([p]), bed_type(o.plant)) for o in scene.of("planting.bed") for p in o.polys if len(p) >= 3]
        m_bed = np.clip(sum(d for d, _ in drifts), 0, 1) if drifts else np.zeros((sh.H, sh.W), F)
        beds = Bed(sh, m_bed, [(None, t) for _, t in drifts], 340, exclude=excl)
        beds.drifts = drifts

        hedges = []
        for i, o in enumerate(scene.of("vegetation.hedge")):
            for p in o.polys:
                hedges.append(Hedge(sh, np.asarray(p, F), o.width_m or 0.8, 330 + i))

        flats = []
        terr = polys("hard.terrace")
        if terr:
            flats.append(self._flat(sh, px, terr, "terrace", 370))
        for o in scene.of("hard.deck"):
            for p in o.polys:
                a = np.asarray(p, F)
                wide = np.ptp(a[:, 0]) > np.ptp(a[:, 1])
                flats.append(self._flat(sh, px, [p], "deck", 371, math.radians(0 if wide else 90)))
        if stones:
            flats.append(self._flat(sh, px, stones, "stone", 372))
        bould = polys("rock")
        if bould:
            flats.append(self._flat(sh, px, bould, "stone", 360))
        for i, p in enumerate(pond):
            pd = Pond.__new__(Pond)
            pd.sh, pd.rs, pd.poly = sh, np.random.default_rng(373 + i), o_px(p)
            flats.append(pd)

        houses = [p for o in scene.of("building") for p in o.polys if len(p) >= 3]

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
                for p in lawn:
                    q = o_px(p)
                    hand_line(sh, np.vstack([q, q[:3]]), st.line_w, rs)
                for p in gravel:
                    q = o_px(p)
                    hand_line(sh, np.vstack([q, q[:3]]), st.line_w * 0.6, rs, 0.7)
                for p in houses:
                    ruled(sh, o_px(p), st.line_w * 3.0, rs, over_mm=1.2)
                for p in bnd[:1]:
                    ruled(sh, o_px(p), st.line_w * 0.8, rs, 0.6)

        objs = [Building(sh, o_px(p), o.height_m or 6.0, 9 + i)
                for i, o in enumerate(scene.of("building")) for p in o.polys if len(p) >= 3]
        furn = [o_px(p) for o in scene.of("furniture") for p in o.polys if len(p) >= 3]
        if furn:
            objs.append(PolySolid(sh, furn, 0.75, (0.90, 0.90, 0.92), 380))
        for i, o in enumerate(scene.of("maf")):
            for p in o.polys:
                if len(p) >= 3:
                    objs.append(PolySolid(sh, [o_px(p)], o.height_m or 0.45, PAL["wood"][0], 381 + i))
        objs += [beds] + hedges
        order = {"column": 0, "shrub": 1, "spruce": 2, "decid": 3}
        objs += sorted(plants, key=lambda c: (order.get(c.kind, 3), c.R))

        def after(sh, st, rs, rset):
            if not rset.statuses:
                return
            for c, status in crowns_status:
                if status == "remove":       # удаляемое: крест через крону
                    r = c.R * 0.75
                    for s in (1, -1):
                        hand_line(sh, [(c.cx - r, c.cy - s * r), (c.cx + r, c.cy + s * r)],
                                  max(st.line_w, 0.18) * 1.4, rs, 0.95)
                elif status == "new":        # новое: тонкий второй контур
                    t = np.linspace(0, 2 * math.pi, 120)
                    rr = c.R * 1.06
                    hand_line(sh, np.stack([c.cx + rr * np.cos(t), c.cy + rr * np.sin(t)], 1),
                              max(st.line_w, 0.16) * 0.7, rs, 0.6)
        return objs, ground, after

    @staticmethod
    def _flat(sh, px, polys_m, kind, seed, board_angle=0.0):
        f = Flat.__new__(Flat)
        f.sh, f.rs = sh, np.random.default_rng(seed)
        f.polys = [px(np.asarray(p, F)) for p in polys_m]
        f.kind, f.ba = kind, board_angle
        return f

    # ---------------------------------------------------------------------------------------------
    def _annotations(self, img: Image.Image, sh, scene: Scene, px):
        d = ImageDraw.Draw(img)
        for o in scene.of("annotation"):
            if not o.text or o.center is None:
                continue
            h_mm = (o.text_height_m or 0.2) * 1000.0 / scene.viewport.scale_den
            size = max(8, int(round(h_mm * sh.ppmm * 1.25)))
            try:
                f = ImageFont.truetype(FONT, size)
            except OSError:
                f = ImageFont.load_default()
            x, y = px(np.asarray([o.center], F))[0]
            d.text((float(x), float(y)), o.text, font=f, fill=(45, 45, 48), anchor="mm",
                   stroke_width=max(1, size // 7), stroke_fill=(252, 250, 244))
