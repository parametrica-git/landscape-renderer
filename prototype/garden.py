"""Ландшафтный дизайн: сад 22×15 м, М 1:100. Архитектурная подача:
тонкая иерархия линий, кроны из листовых кластеров, архитектура под линейку,
лёгкие заливки и много белой бумаги."""
import sys, time, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import engine
from engine import (Sheet, STYLES, PAL, F, smoothstep, shift, raster, stipple, hand_line,
                    resample, Pen, Pastel, Watercolor, Marker, Pencil)
from objects import Crown, Solid, Bench, grid
from render import FONT, STYLE_RU

TAU = 2 * math.pi

# ---------------------------------------------------------------- «архитектурные» настройки
engine.Style.line_w = 0.16
engine.hand_line.__defaults__ = (1.0, 0.04, False)     # почти без дрожания
Marker.alpha = None


def _marker_init(self, ink=True):
    self.line = ink
    self.w_mm = (5.0, 3.6, 2.4)
    self.alpha = (0.62, 0.72, 0.80)


Marker.__init__ = _marker_init
Marker.shadow_color = (0.70, 0.73, 0.75)
Watercolor.shadow_color = (0.82, 0.82, 0.88)

PAL.update({
    "lawn": [(0.86, 0.92, 0.68), (0.78, 0.88, 0.58), (0.84, 0.78, 0.60)],
    "leaf": [(0.80, 0.88, 0.56), (0.56, 0.71, 0.36), (0.30, 0.45, 0.24), (0.94, 0.96, 0.72)],
    "leaf2": [(0.85, 0.87, 0.52), (0.64, 0.70, 0.34), (0.38, 0.46, 0.22), (0.96, 0.95, 0.70)],
    "leaf3": [(0.74, 0.84, 0.62), (0.48, 0.64, 0.44), (0.26, 0.40, 0.30), (0.90, 0.94, 0.80)],
    "copper": [(0.92, 0.82, 0.72), (0.80, 0.62, 0.54), (0.62, 0.42, 0.40), (0.97, 0.90, 0.82)],
    "hedge": [(0.72, 0.82, 0.56), (0.46, 0.60, 0.36), (0.26, 0.38, 0.22), (0.88, 0.92, 0.72)],
    "water": [(0.72, 0.85, 0.89), (0.52, 0.70, 0.82), (0.38, 0.54, 0.68)],
    "stone": [(0.88, 0.87, 0.85), (0.74, 0.73, 0.72), (0.58, 0.57, 0.57)],
    "terrace": [(0.92, 0.89, 0.83), (0.84, 0.80, 0.73), (0.72, 0.68, 0.62)],
    "mulch": [(0.91, 0.87, 0.79), (0.84, 0.78, 0.68), (0.72, 0.64, 0.54)],
    "gravel": [(0.94, 0.92, 0.86), (0.86, 0.83, 0.76), (0.74, 0.71, 0.66)],
    "wood": [(0.88, 0.76, 0.58), (0.76, 0.62, 0.44), (0.58, 0.44, 0.30)],
})


def ruled(sh, pts, w_mm, rs, alpha=1.0, over_mm=0.7, closed=True):
    """Линии под линейку с лёгким перехлёстом в углах."""
    P = np.asarray(pts, F)
    n = len(P) if closed else len(P) - 1
    for i in range(n):
        a, b = P[i], P[(i + 1) % len(P)]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1:
            continue
        d /= L
        hand_line(sh, [a - d * sh.mm(over_mm) * rs.uniform(0.2, 1.0), b + d * sh.mm(over_mm) * rs.uniform(0.2, 1.0)],
                  w_mm, rs, alpha, tremor_mm=0.0)


def smooth_blob(c, rx, ry, rs, amp=0.10, n=160, rot=0.0):
    t = np.linspace(0, TAU, n, endpoint=False)
    r = np.ones_like(t)
    for k in (2, 3, 4, 5):
        r += rs.uniform(-1, 1) * amp / k * 2 * np.cos(k * t + rs.uniform(0, TAU))
    x, y = np.cos(t) * r * rx, np.sin(t) * r * ry
    cr, sr = math.cos(rot), math.sin(rot)
    return np.stack([c[0] + x * cr - y * sr, c[1] + x * sr + y * cr], 1).astype(F)


def organic(ctrl, step=0.1, smooth=9):
    P = np.asarray(ctrl, F)
    P = np.vstack([P, P[:1]])
    P = resample(P, step)[:-1]
    k = np.ones(smooth * 2 + 1, F) / (smooth * 2 + 1)
    return np.stack([np.convolve(np.pad(P[:, i], smooth, mode="wrap"), k, "valid") for i in (0, 1)], 1).astype(F)


def lobed(cx, cy, r, rs, n=14, amp=0.16):
    k = int(rs.integers(4, 8))
    ph = rs.uniform(0, TAU)
    t = np.linspace(0, TAU, n, endpoint=False)
    rr = r * (1 + amp * np.sin(k * t + ph) + 0.07 * rs.normal(size=n))
    return np.stack([cx + rr * np.cos(t), cy + rr * np.sin(t)], 1).astype(F)


def fill_polys(mask, polys, off):
    for p in polys:
        cv2.fillPoly(mask, [np.round((p - off) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)


def dots(sh, style, m, bb, color, dens=1.0):
    """Мелкие цветки/точки цвета в любом стиле."""
    s = sh.sl(bb)
    if isinstance(style, Pen):
        sh.ink_add(m * 0.75, bb)
    elif isinstance(style, Pastel):
        tooth = np.clip((sh.grain[s] - 0.25) * 2.5, 0, 1)
        sh.over(m * tooth * 0.95, color, bb)
    elif isinstance(style, Watercolor):
        mm = cv2.GaussianBlur(m, (0, 0), sh.mm(0.15))
        sh.beer(mm * (0.85 + 0.35 * sh.grain[s]) * dens * 1.1, color, bb)
    elif isinstance(style, Pencil):
        tooth = np.clip((sh.grain[s] - 0.15) * 2.0, 0, 1)
        sh.mul(m * tooth * 0.9, color, bb)
    else:
        sh.mul(m * 0.85 * dens, color, bb)


# ================================================================ кроны из листовых кластеров
class LeafCrown(Crown):
    def __init__(self, sh, kind, xy_m, D_m, seed, pal="leaf", dab_m=None, holes=0.04,
                 branches=True, height_m=None):
        super().__init__(sh, kind, xy_m, D_m, seed, pal=None, height_m=height_m)
        self.pal = PAL[pal]
        self.dab = dab_m or (0.24 if kind == "decid" else 0.15)
        self.holes, self.branches = holes, branches

    def leaf_layers(self, bb, shade):
        sh, rs = self.sh, self.rs
        h, w = shade.shape
        off = np.array([bb[0], bb[1]], F)
        r0 = self.dab * sh.ppm
        N = int((self.R / r0) ** 2 * 2.6) + 6
        lay = {k: np.zeros((h, w), np.uint8) for k in ("base", "mid", "dark", "hl")}
        self.dabs = []
        for _ in range(N):
            rr, th = math.sqrt(rs.random()), rs.uniform(0, TAU)
            er = float(self.r(np.array([th], F))[0])
            x, y = self.cx + math.cos(th) * rr * er, self.cy + math.sin(th) * rr * er
            rad = r0 * rs.uniform(0.6, 1.25) * (0.8 if rr > 0.88 else 1.0)
            xi, yi = int(np.clip(x - bb[0], 0, w - 1)), int(np.clip(y - bb[1], 0, h - 1))
            t = float(shade[yi, xi]) + rs.normal(0, 0.07)
            if t > 0.55 and rs.random() < self.holes and rr > 0.3:
                continue                           # просвет бумаги
            self.dabs.append((x, y, rad, t))
        self.dabs.sort(key=lambda d: -d[3])
        for x, y, rad, t in self.dabs:
            fill_polys(lay["base"], [lobed(x, y, rad, rs)], off)
            if t < 0.50:
                fill_polys(lay["mid"], [lobed(x + rad * 0.1, y + rad * 0.1, rad * 0.85, rs)], off)
            if t < 0.30:
                fill_polys(lay["dark"], [lobed(x + rad * 0.15, y + rad * 0.15, rad * 0.72, rs)], off)
            if t > 0.74:
                fill_polys(lay["hl"], [lobed(x - rad * 0.1, y - rad * 0.1, rad * 0.55, rs)], off)
        return {k: v.astype(F) / 255 for k, v in lay.items()}

    def prepare(self):
        super().prepare()

    def draw(self, style):
        if self.kind == "spruce":
            return super().draw(style)
        sh, rs = self.sh, self.rs
        bb = self.bb()
        X, Y, sd, m = self.masks(bb)
        g = self.shade(X, Y)
        shade = np.clip((g + 0.85) / 1.7 + 0.06 * (sh.nmid[sh.sl(bb)] - 0.5), 0, 1)
        L = self.leaf_layers(bb, shade)
        form = (self.cx - bb[0], self.cy - bb[1], self.R)
        p = self.pal
        wc = isinstance(style, Watercolor)
        if wc:
            en, dn = style.edge_noise, style.dens
            style.edge_noise, style.dens = 0.12, (0.95, 0.80, 0.70)
        style.fill(sh, L["base"], bb, p[0], 0, rs, form=form)
        style.fill(sh, L["mid"], bb, p[1], 1, rs, form=form)
        style.fill(sh, L["dark"], bb, p[2], 2, rs, form=form)
        if wc:
            style.edge_noise, style.dens = en, dn
        if style.opaque:
            style.fill(sh, L["hl"], bb, p[3], 2, rs, form=form)
        if style.line:
            self.leaf_ink(style, L, bb)

    def leaf_ink(self, style, L, bb):
        sh, rs = self.sh, self.rs
        w = style.line_w * 0.75
        # прерывистый контур по листовой кромке
        cnt, _ = cv2.findContours((L["base"] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for ct in cnt:
            if len(ct) < 30:
                continue
            P = ct[:, 0, :].astype(F) + np.array([bb[0], bb[1]], F)
            k = np.ones(9, F) / 9
            P = np.stack([np.convolve(np.pad(P[:, i], 4, mode="wrap"), k, "valid") for i in (0, 1)], 1)
            i = int(rs.integers(0, len(P)))
            P = np.roll(P, -i, axis=0)
            j = 0
            while j < len(P) - 4:
                seg = int(rs.integers(25, 90))
                if rs.random() < 0.93:
                    hand_line(sh, P[j:j + seg + 3], w * 1.15, rs, 0.9)
                j += seg + int(rs.integers(0, 6))
        # «гребешки» на теневой стороне кластеров
        ta = sh.shadow_theta
        for x, y, rad, t in self.dabs:
            if t < 0.48 and rs.random() < 0.35:
                ts = np.linspace(ta - 1.1, ta + 1.1, 9)
                hand_line(sh, np.stack([x + rad * 0.9 * np.cos(ts), y + rad * 0.9 * np.sin(ts)], 1),
                          w * 0.6, rs, 0.55)
        # ветви, проглядывающие в просветах
        if self.branches and self.kind == "decid":
            c = np.array([self.cx, self.cy], F)
            R = self.R

            def br(p, a, l, d, ww):
                if d == 0:
                    return
                q = p + np.array([math.cos(a), math.sin(a)], F) * l
                pts = resample([p, q], sh.mm(0.6))
                vis = np.array([L["base"][int(np.clip(y - bb[1], 0, L["base"].shape[0] - 1)),
                                          int(np.clip(x - bb[0], 0, L["base"].shape[1] - 1))] < 0.5
                                or rs.random() < 0.12 for x, y in pts])
                run = []
                for pt, v in zip(pts, vis):
                    if v:
                        run.append(pt)
                    else:
                        if len(run) > 2:
                            hand_line(sh, run, ww, rs, 0.8)
                        run = []
                if len(run) > 2:
                    hand_line(sh, run, ww, rs, 0.8)
                for s_ in (-1, 1):
                    br(q, a + s_ * rs.uniform(0.3, 0.6), l * 0.62, d - 1, ww * 0.8)

            for k in range(int(rs.integers(4, 7))):
                br(c, k * TAU / 5.5 + rs.normal(0, 0.3), R * rs.uniform(0.3, 0.4), 3, w * 0.9)
            cv2.circle(sh.ink, (int(c[0]), int(c[1])), max(2, int(sh.mm(0.35))), 0.9, -1, cv2.LINE_AA)


# ================================================================ живая изгородь
class Hedge(Solid):
    height = 1.6

    def __init__(self, sh, line_m, width_m, seed, pal="hedge"):
        self.sh, self.rs = sh, np.random.default_rng(seed)
        self.line = sh.px(resample(line_m, 0.1))
        self.wpx = width_m * sh.ppm
        self.pal = PAL[pal]
        d = np.gradient(self.line, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        self.n = np.stack([-d[:, 1], d[:, 0]], 1)
        L, R = self.line + self.n * self.wpx / 2, self.line - self.n * self.wpx / 2
        self.band = np.vstack([L, R[::-1]])

    def prepare(self):
        self.bb_, _ = self.prep_masks([self.band], pad_extra=self.wpx * 0.4)

    def draw(self, style):
        sh, rs, bb = self.sh, self.rs, self.bb_
        h, w = bb[3] - bb[1], bb[2] - bb[0]
        off = np.array([bb[0], bb[1]], F)
        lay = {k: np.zeros((h, w), np.uint8) for k in ("base", "mid", "dark", "hl")}
        dabs = []
        step = self.wpx * 0.28
        acc = 0.0
        for i in range(1, len(self.line)):
            acc += float(np.linalg.norm(self.line[i] - self.line[i - 1]))
            if acc < step:
                continue
            acc = 0
            for u in (-0.42, -0.15, 0.15, 0.42):
                u2 = u + rs.normal(0, 0.06)
                p = self.line[i] + self.n[i] * self.wpx * u2
                lit = -float(np.dot(self.n[i] * np.sign(u2), sh.sdir))   # >0 сторона к свету
                t = 0.55 + 0.30 * lit * abs(u2) * 2 + rs.normal(0, 0.08)
                dabs.append((p[0], p[1], self.wpx * rs.uniform(0.20, 0.30), t))
        dabs.sort(key=lambda d: -d[3])
        for x, y, r, t in dabs:
            fill_polys(lay["base"], [lobed(x, y, r, rs)], off)
            if t < 0.52:
                fill_polys(lay["mid"], [lobed(x + r * .1, y + r * .1, r * .85, rs)], off)
            if t < 0.34:
                fill_polys(lay["dark"], [lobed(x + r * .15, y + r * .15, r * .7, rs)], off)
            if t > 0.72:
                fill_polys(lay["hl"], [lobed(x, y, r * .5, rs)], off)
        Lf = {k: v.astype(F) / 255 for k, v in lay.items()}
        p = self.pal
        style.fill(sh, Lf["base"], bb, p[0], 0, rs)
        style.fill(sh, Lf["mid"], bb, p[1], 1, rs)
        style.fill(sh, Lf["dark"], bb, p[2], 2, rs)
        if style.opaque:
            style.fill(sh, Lf["hl"], bb, p[3], 2, rs)
        if style.line:
            cnt, _ = cv2.findContours((Lf["base"] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for ct in cnt:
                if len(ct) > 30:
                    P = ct[:, 0, :].astype(F) + off
                    for j in range(0, len(P), 60):
                        if rs.random() < 0.8:
                            hand_line(sh, P[j:j + 55], style.line_w * 0.75, rs, 0.85)


# ================================================================ цветник (миксбордер)
PLANTS = {
    #           base leaf            mid               flower             шаг, м
    "lavender": [(0.84, 0.88, 0.78), (0.72, 0.78, 0.66), (0.66, 0.56, 0.82), 0.45],
    "salvia":   [(0.82, 0.89, 0.70), (0.68, 0.78, 0.56), (0.56, 0.46, 0.78), 0.50],
    "echinacea": [(0.82, 0.89, 0.68), (0.66, 0.76, 0.52), (0.92, 0.60, 0.70), 0.50],
    "rudbeckia": [(0.84, 0.90, 0.66), (0.68, 0.78, 0.50), (0.96, 0.80, 0.36), 0.50],
    "gaura":    [(0.82, 0.89, 0.70), (0.68, 0.78, 0.56), (0.99, 0.99, 0.99), 0.55],
    "hosta":    [(0.78, 0.88, 0.70), (0.60, 0.74, 0.54), None, 0.75],
    "grass":    [(0.93, 0.90, 0.72), (0.84, 0.79, 0.58), None, 0.70],
}


class Bed:
    def __init__(self, sh, mask_full, seeds_types, seed, exclude=None):
        self.sh, self.rs = sh, np.random.default_rng(seed)
        self.m = mask_full
        self.seeds_types = seeds_types      # [(xy_m, type)]
        self.exclude = exclude

    def prepare(self):
        pass

    def ground(self, style, rs):
        sh = self.sh
        style.fill(sh, self.m, sh.full(), PAL["mulch"][0], 0, rs, angle=math.radians(15))

    def draw(self, style):
        sh, rs = self.sh, self.rs
        ys, xs = np.nonzero(self.m > 0.5)
        if len(xs) == 0:
            return
        bb = sh.bbox(xs.min() - 20, ys.min() - 20, xs.max() + 20, ys.max() + 20)
        s = sh.sl(bb)
        m = self.m[s] * (1 - (self.exclude[s] if self.exclude is not None else 0))
        X, Y = grid(bb)
        # вороной с шумом по границам -> «дрифты» посадок
        if getattr(self, "drifts", None):      # пятна заданы полилиниями из DWG
            lab = np.full(m.shape, -1, int)
            for i, (mk, _) in enumerate(self.drifts):
                lab[mk[s] > 0.5] = i
        else:
            seeds = np.array([sh.px(p) for p, _ in self.seeds_types], F)
            nz = (sh.nmid[s] - 0.5) * sh.ppm * 1.4
            D = np.stack([np.hypot(X - x, Y - y) + nz * (1 if i % 2 else -1) for i, (x, y) in enumerate(seeds)])
            lab = np.argmin(D, 0)
        off = np.array([bb[0], bb[1]], F)
        h, w = m.shape
        for i, (_, typ) in enumerate(self.seeds_types):
            dm = m * (lab == i)
            if dm.sum() < 50:
                continue
            base, mid, flower, step = PLANTS[typ]
            sp = step * sh.ppm
            lay_p = np.zeros((h, w), np.uint8)
            lay_mid = np.zeros((h, w), np.uint8)
            lay_f = np.zeros((h, w), np.uint8)
            tufts = []
            y0 = rs.uniform(0, sp)
            yy = y0
            row = 0
            while yy < h:
                xx = rs.uniform(0, sp) + (sp / 2 if row % 2 else 0)
                while xx < w:
                    px_, py_ = xx + rs.normal(0, sp * 0.15), yy + rs.normal(0, sp * 0.15)
                    if 0 <= int(py_) < h and 0 <= int(px_) < w and dm[int(py_), int(px_)] > 0.5:
                        r = sp * rs.uniform(0.50, 0.62)
                        cx, cy = px_ + off[0], py_ + off[1]
                        fill_polys(lay_p, [lobed(cx, cy, r, rs, amp=0.22)], off)
                        # теневая половина куста
                        fill_polys(lay_mid, [lobed(cx + sh.sdir[0] * r * 0.40, cy + sh.sdir[1] * r * 0.40,
                                                   r * 0.48, rs)], off)
                        if flower is not None:
                            for _ in range(int(rs.integers(3, 7))):
                                a, rr = rs.uniform(0, TAU), r * math.sqrt(rs.random()) * 0.8
                                cv2.circle(lay_f, (int((cx + rr * math.cos(a) - off[0]) * 16),
                                                   int((cy + rr * math.sin(a) - off[1]) * 16)),
                                           int(max(1.5, sh.mm(0.38)) * 16), 255, -1, cv2.LINE_AA, shift=4)
                        if typ == "grass":
                            tufts.append((cx, cy, r))
                    xx += sp
                yy += sp * 0.87
                row += 1
            P = lay_p.astype(F) / 255
            Mi = lay_mid.astype(F) / 255 * P
            Fl = lay_f.astype(F) / 255
            if flower == (0.99, 0.99, 0.99):     # белые цветы — оставляем бумагу
                P = P * (1 - Fl)
                Mi = Mi * (1 - Fl)
            style.fill(sh, P, bb, base, 0, rs)
            style.fill(sh, Mi, bb, mid, 1, rs, angle=math.radians(-35))
            if flower is not None and flower != (0.99, 0.99, 0.99):
                dots(sh, style, Fl, bb, flower)
            elif flower == (0.99, 0.99, 0.99) and style.line:
                pass
            if style.line:
                # редкие контуры кустиков + пучки злаков
                cnt, _ = cv2.findContours((P > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                for ct in cnt:
                    if len(ct) > 12 and rs.random() < 0.30:
                        Q = ct[:, 0, :].astype(F) + off
                        a0 = int(rs.integers(0, len(Q)))
                        Q = np.roll(Q, -a0, 0)[: int(len(Q) * rs.uniform(0.3, 0.7))]
                        hand_line(sh, Q, style.line_w * 0.45, rs, 0.45)
                for cx, cy, r in tufts:
                    for k in range(int(rs.integers(6, 10))):
                        a = rs.uniform(0, TAU)
                        l = r * rs.uniform(0.7, 1.15)
                        bend = rs.normal(0, 0.25)
                        pts = [(cx, cy), (cx + math.cos(a) * l * .55, cy + math.sin(a) * l * .55),
                               (cx + math.cos(a + bend * .5) * l, cy + math.sin(a + bend * .5) * l)]
                        hand_line(sh, pts, style.line_w * 0.40, rs, 0.50)


# ================================================================ водоём, настил, плиты, мебель
class Pond:
    def __init__(self, sh, poly_m, seed):
        self.sh, self.rs = sh, np.random.default_rng(seed)
        self.poly = sh.px(poly_m)

    def prepare(self):
        pass

    def ground(self, style, rs):
        sh = self.sh
        P = self.poly
        bb = sh.bbox(P[:, 0].min() - 30, P[:, 1].min() - 30, P[:, 0].max() + 30, P[:, 1].max() + 30)
        m = sh.poly_mask([P], bb)
        h, w = m.shape
        dist = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        depth = dist / (dist.max() + 1e-6)
        # блики — короткие волнистые полосы, оставляем бумагу
        hl = []
        for _ in range(9):
            x, y = rs.uniform(0.2, 0.8) * w, rs.uniform(0.2, 0.8) * h
            l = rs.uniform(0.6, 1.4) * sh.ppm
            hl.append(np.array([[x - l / 2, y], [x, y + rs.normal(0, 3)], [x + l / 2, y]], F))
        H = raster(h, w, hl, max(2, sh.mm(0.5))) * m
        p = PAL["water"]
        style.fill(sh, m * (1 - H), bb, p[0], 0, rs, angle=0.0)
        style.fill(sh, m * smoothstep(0.25, 0.6, depth) * (1 - H), bb, p[1], 1, rs, angle=0.0)
        # тень от берега на светлой (верхне-левой) кромке
        k = 0.35 * sh.ppm
        bank = m * (1 - shift(m, sh.sdir[0] * k, sh.sdir[1] * k))
        style.fill(sh, bank, bb, p[2], 2, rs, angle=0.0)
        if style.line:
            hand_line(sh, np.vstack([P, P[:3]]), style.line_w * 1.1, rs)
            for _ in range(6):   # рябь
                c = np.array([rs.uniform(0.3, 0.7) * w, rs.uniform(0.3, 0.7) * h], F) + np.array([bb[0], bb[1]], F)
                for rr in (0.25, 0.42):
                    a0 = rs.uniform(0, TAU)
                    ts = np.linspace(a0, a0 + 1.3, 10)
                    hand_line(sh, np.stack([c[0] + rr * sh.ppm * np.cos(ts), c[1] + rr * sh.ppm * .5 * np.sin(ts)], 1),
                              style.line_w * 0.45, rs, 0.5)

    def draw(self, style):
        pass


class Flat:
    """Плоские элементы на уровне земли: плиты, настил, мощение (тени на них падают)."""

    def __init__(self, sh, polys_m, kind, seed, board_angle=0.0):
        self.sh, self.rs = sh, np.random.default_rng(seed)
        self.polys = [sh.px(p) for p in polys_m]
        self.kind, self.ba = kind, board_angle

    def prepare(self):
        pass

    def ground(self, style, rs):
        sh = self.sh
        P = np.vstack(self.polys)
        bb = sh.bbox(P[:, 0].min() - 20, P[:, 1].min() - 20, P[:, 0].max() + 20, P[:, 1].max() + 20)
        m = sh.poly_mask(self.polys, bb)
        pal = PAL[{"deck": "wood", "stone": "stone", "terrace": "terrace"}[self.kind]]
        style.fill(sh, m, bb, pal[0], 0, rs, angle=self.ba)
        if self.kind == "stone":
            sx, sy = -sh.sdir * 0.06 * sh.ppm
            style.fill(sh, m * (1 - shift(m, sx, sy)), bb, pal[1], 1, rs)
        if not style.line:
            return
        w = style.line_w
        if self.kind == "stone":
            for p in self.polys:
                hand_line(sh, np.vstack([p, p[:2]]), w * 0.9, rs)
        else:
            for p in self.polys:
                ruled(sh, p, w * 1.2, rs)
            h, wd = m.shape
            lines = []
            if self.kind == "deck":     # доски 0,14 м
                sp = 0.14 * sh.ppm
                d = np.array([math.cos(self.ba), math.sin(self.ba)], F)
                n = np.array([-d[1], d[0]], F)
                diag = math.hypot(h, wd)
                for t in np.arange(-diag, diag, sp):
                    c = np.array([wd / 2, h / 2], F) + n * t
                    lines.append(np.array([c - d * diag, c + d * diag]))
                # торцевые стыки
                for t in np.arange(-diag, diag, sp):
                    c = np.array([wd / 2, h / 2], F) + n * t
                    u = rs.uniform(-diag, diag)
                    lines.append(np.array([c + d * u - n * sp / 2, c + d * u + n * sp / 2]))
                alpha = 0.45
            else:                       # плиты 0,8×0,6 вразбежку
                for j, y in enumerate(np.arange(0, h, 0.6 * sh.ppm)):
                    lines.append(np.array([[0, y], [wd, y]], F))
                    for x in np.arange((j % 2) * 0.4 * sh.ppm, wd, 0.8 * sh.ppm):
                        lines.append(np.array([[x, y], [x, y + 0.6 * sh.ppm]], F))
                alpha = 0.40
            lay = raster(h, wd, lines, max(1, sh.mm(0.08)))
            sh.ink_add(lay * np.clip(m * 1.3 - 0.15, 0, 1), bb, alpha)

    def draw(self, style):
        pass


class Furniture(Solid):
    height = 0.75

    def __init__(self, sh, center_m, seed):
        self.sh, self.rs = sh, np.random.default_rng(seed)
        c = np.asarray(center_m, F)
        t = np.linspace(0, TAU, 40, endpoint=False)
        self.table = sh.px(np.stack([c[0] + 0.55 * np.cos(t), c[1] + 0.55 * np.sin(t)], 1))
        self.chairs = []
        for a in np.radians([20, 110, 200, 290]):
            cc = c + np.array([math.cos(a), math.sin(a)], F) * 0.95
            q = np.array([[-0.22, -0.22], [0.22, -0.22], [0.22, 0.22], [-0.22, 0.22]], F)
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], F)
            self.chairs.append(sh.px(q @ R.T + cc))

    def prepare(self):
        self.bb_, _ = self.prep_masks([self.table] + self.chairs)
        s = self.sh.sl(self.bb_)
        self.sh.top[s] *= 0        # низкий объект: тени на него падают

    def draw(self, style):
        sh, rs, bb = self.sh, self.rs, self.bb_
        m = sh.poly_mask([self.table] + self.chairs, bb)
        if not isinstance(style, Pen):
            style.fill(sh, m, bb, (0.90, 0.90, 0.92), 0, rs, angle=0.4)
        if style.line:
            hand_line(sh, np.vstack([self.table, self.table[:2]]), style.line_w, rs)
            for c in self.chairs:
                ruled(sh, c, style.line_w * 0.8, rs, over_mm=0.2)


# ================================================================ сцена
W_M, H_M = 22.0, 15.0


def build(sh, rs):
    px = sh.px
    house = [(0, 0), (14, 0), (14, 3.0), (0, 3.0)]
    terrace = [(0, 3.0), (8.6, 3.0), (8.6, 6.2), (0, 6.2)]
    deck = [(8.6, 3.0), (14.0, 3.0), (14.0, 6.8), (8.6, 6.8)]
    lawn = organic([(3.6, 8.6), (6.5, 7.6), (10.4, 7.9), (13.0, 9.0), (12.9, 11.0), (10.8, 12.1),
                    (8.0, 11.5), (5.6, 12.7), (3.4, 12.2), (3.0, 10.2)])
    gravel = smooth_blob((17.6, 6.6), 3.6, 3.4, rs, amp=0.12)
    pond = smooth_blob((17.9, 6.9), 2.2, 1.6, rs, amp=0.14, rot=0.3)
    # дорожка из плит: настил -> пруд -> скамейка
    path = [(13.2, 7.3), (14.4, 8.5), (15.3, 9.9), (16.4, 11.2), (17.6, 12.0)]
    pts = resample(path, 0.75)
    stones = [lobed(x, y, 0.30, rs, n=24, amp=0.06) for x, y in pts]
    stones = [np.asarray(s_, F) for s_ in stones]

    M = lambda polys: sh.poly_mask([px(p) for p in polys])
    garden = M([[(0, 3.0), (21.1, 3.0), (21.1, 14.1), (0, 14.1)], [(14, 0), (21.1, 0), (21.1, 3.0), (14, 3.0)]])
    m_lawn, m_grav, m_pond = M([lawn]), M([gravel]), M([pond])
    m_hard = M([terrace, deck])
    m_st = M(stones)
    sh.site = M([[(0, 0), (W_M, 0), (W_M, H_M), (0, H_M)]])
    m_grav = np.clip(m_grav - m_pond - m_st, 0, 1)
    m_bed = np.clip(garden - m_lawn - m_grav - m_pond - m_hard - m_st, 0, 1)
    m_lawn = np.clip(m_lawn - m_st, 0, 1)

    # посадки
    tree1 = LeafCrown(sh, "decid", (2.6, 11.8), 5.4, 301, pal="leaf")
    tree2 = LeafCrown(sh, "decid", (15.4, 12.3), 4.6, 302, pal="leaf2")
    tree3 = LeafCrown(sh, "decid", (11.6, 8.3), 3.0, 303, pal="copper", dab_m=0.18)   # многоствольное декоративное
    spruce = Crown(sh, "spruce", (19.8, 1.8), 2.6, 304)
    junip = [Crown(sh, "column", xy, 0.9, 305 + i) for i, xy in enumerate([(0.7, 7.0), (8.0, 6.9), (14.6, 3.6)])]
    shrubs = [LeafCrown(sh, "shrub", xy, d, 320 + i, pal=p, dab_m=0.14, branches=False)
              for i, (xy, d, p) in enumerate([((5.6, 13.2), 1.6, "leaf3"), ((7.0, 13.4), 1.3, "leaf"),
                                               ((19.4, 10.6), 1.7, "leaf3"), ((20.2, 12.4), 1.4, "leaf2"),
                                               ((15.0, 3.9), 1.3, "leaf3"), ((1.2, 8.8), 1.4, "leaf")])]
    hedge = [Hedge(sh, [(21.45, 0.3), (21.45, 14.5), (0.4, 14.5)], 0.8, 330)]
    # маска кустов, чтобы цветы под ними не рисовать
    excl = np.zeros((sh.H, sh.W), F)
    for c in shrubs + junip + [tree1, tree2, tree3]:
        bbx = c.bb()
        X, Y = grid(bbx)
        s = sh.sl(bbx)
        excl[s] = np.maximum(excl[s], np.clip(c.sd(X, Y) / 1.5 + 0.5, 0, 1))
    beds = Bed(sh, m_bed, [((0.8, 4.0), "lavender"), ((1.4, 6.5), "hosta"), ((1.0, 9.6), "salvia"),
                           ((4.0, 13.4), "echinacea"), ((9.0, 13.5), "gaura"), ((11.6, 13.3), "rudbeckia"),
                           ((13.6, 12.0), "grass"), ((14.6, 6.4), "lavender"), ((16.2, 3.5), "grass"),
                           ((20.4, 4.6), "hosta"), ((20.4, 8.4), "salvia"), ((18.6, 13.2), "echinacea"),
                           ((12.6, 10.4), "grass"), ((4.2, 7.4), "echinacea")], 340, exclude=excl)
    # злаки и валуны в гравийном саду
    tuft_m = M([lobed(x, y, r, rs, n=24) for x, y, r in [(15.4, 7.6, 0.7), (16.2, 4.4, 0.6), (20.0, 7.8, 0.6),
                                                           (19.4, 4.1, 0.55), (17.0, 9.0, 0.5)]]) * m_grav
    grav_tufts = Bed(sh, tuft_m, [((17.6, 6.6), "grass")], 350)
    boulders = Flat(sh, [lobed(x, y, r, rs, n=16, amp=0.12) for x, y, r in
                         [(15.2, 5.2, 0.45), (15.6, 5.8, 0.28), (20.2, 6.0, 0.38), (16.8, 9.4, 0.32)]], "stone", 360)

    flats = [Flat(sh, [terrace], "terrace", 370), Flat(sh, [deck], "deck", 371, board_angle=math.radians(90)),
             Flat(sh, stones, "stone", 372), boulders, Pond(sh, pond, 373)]

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
            # кромка газона — стальной бордюр
            hand_line(sh, np.vstack([px(lawn), px(lawn[:3])]), st.line_w * 1.0, rs)
            hand_line(sh, np.vstack([px(gravel), px(gravel[:3])]), st.line_w * 0.6, rs, 0.7)
            ruled(sh, px(house), st.line_w * 3.0, rs, over_mm=1.2)
            # граница участка
            ruled(sh, px([(0, 0), (W_M, 0), (W_M, H_M), (0, H_M)]), st.line_w * 0.8, rs, 0.6)

    class House(Solid):
        height = 6.0

        def __init__(s):
            s.sh, s.rs, s.poly = sh, np.random.default_rng(9), px(house)

        def prepare(s):
            s.bb_, _ = s.prep_masks([s.poly])

        def draw(s, st):
            m = sh.poly_mask([s.poly], s.bb_)
            if isinstance(st, Pen):
                st.fill(sh, m, s.bb_, None, 1, s.rs, angle=0.0)
            else:
                st.fill(sh, m, s.bb_, (0.88, 0.87, 0.86), 0, s.rs, angle=0.0)

    objs = [House(), Furniture(sh, (4.4, 4.7), 380),
            Bench(sh, (17.9, 6.9), 2.75, 45, 115, 381)]
    objs += [beds, grav_tufts] + hedge + junip + shrubs + [tree3, spruce, tree1, tree2]
    sh.geo = dict(house=house, terrace=terrace, deck=deck, lawn=lawn, gravel=gravel, pond=pond, stones=stones,
                  boulders=boulders, beds=beds, m_bed=m_bed, tufts=tuft_m, trees=[tree1, tree2, tree3],
                  spruce=spruce, junip=junip, shrubs=shrubs, hedge=hedge, bench=objs[2], furn=objs[1])
    return objs, ground


def render_garden(style_key, dpi=220, seed=21):
    sh = Sheet(W_M * 10 + 20, H_M * 10 + 28, dpi=dpi, scale=100, seed=seed)
    st = STYLES[style_key]()
    rs = np.random.default_rng(seed)
    objs, ground = build(sh, rs)
    for o in objs:
        o.prepare()
    ground(sh, st, rs)
    sm = sh.shad * (1 - sh.top) * sh.site
    st.shadow(sh, sm, sh.full(), rs)
    for o in objs:
        o.draw(st)
    img = sh.finish(st)
    # подпись и масштабная линейка
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, int(sh.mm(3.2)))
    fs = ImageFont.truetype(FONT, int(sh.mm(2.4)))
    y0 = sh.px((0, H_M))[1] + sh.mm(6)
    d.text((sh.px((0, 0))[0], y0), f"Сад · {STYLE_RU[style_key]} · М 1:100", fill=(55, 55, 58), font=f)
    bx = sh.px((14, 0))[0]
    seg = sh.ppm
    for i in range(5):
        d.rectangle([bx + i * seg, y0 + sh.mm(1), bx + (i + 1) * seg, y0 + sh.mm(2)],
                    fill=(45, 45, 48) if i % 2 == 0 else (252, 250, 244), outline=(45, 45, 48), width=2)
    for i, t in ((0, "0"), (5, "5 м")):
        d.text((bx + i * seg - sh.mm(0.6), y0 + sh.mm(2.6)), t, fill=(55, 55, 58), font=fs)
    return np.asarray(im)


if __name__ == "__main__":
    import os; os.makedirs("out", exist_ok=True)
    for k in sys.argv[1:] or ["pero_akvarel", "pero_marker"]:
        t = time.time()
        img = render_garden(k)
        Image.fromarray(img).save(f"out/garden_{k}.jpg", quality=90)
        print(k, round(time.time() - t, 1), "s", flush=True)
