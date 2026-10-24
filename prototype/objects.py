"""Объекты генплана: кроны, группы кустов, машины, люди, скамейки, здания."""
import math
import numpy as np
import cv2
from engine import (F, PAL, smoothstep, shift, hand_line, poly_outline, closed_curve,
                    raster, Pen, Pastel)

TAU = 2 * math.pi


def grid(bb):
    xs = np.arange(bb[0], bb[2], dtype=F) + 0.5
    ys = np.arange(bb[1], bb[3], dtype=F) + 0.5
    return np.meshgrid(xs, ys)


# ================================================================ кроны
class Crown:
    """kind: decid | spruce | column | shrub. Размеры в метрах, координаты листа в px."""

    def __init__(self, sh, kind, xy_m, D_m, seed, pal=None, flat=False, pen=0, height_m=None):
        self.sh, self.kind, self.flat, self.pen = sh, kind, flat, pen
        self.cx, self.cy = sh.px(xy_m)
        self.R = D_m / 2 * sh.ppm
        self.pal = PAL[pal or {"decid": "decid", "spruce": "spruce", "column": "column",
                               "shrub": "shrub"}[kind]]
        rs = np.random.default_rng(seed)
        self.rs = rs
        self.harm = [(k, rs.uniform(-1, 1) * (0.045 if kind != "column" else 0.025), rs.uniform(0, TAU))
                     for k in (2, 3, 4, 6)]
        self.n = int({"decid": rs.integers(10, 16), "spruce": rs.integers(17, 24),
                      "column": rs.integers(8, 11), "shrub": rs.integers(8, 12)}[kind])
        self.sc = {"decid": 0.075, "spruce": 0.0, "column": 0.05, "shrub": 0.09}[kind]
        self.ph = rs.uniform(0, TAU)
        h = height_m if height_m is not None else {"decid": D_m * 1.3, "spruce": D_m * 2.4,
                                                   "column": D_m * 4.0, "shrub": 1.4}[kind]
        off = min(h * sh.shadow_k, D_m * 0.40) * sh.ppm   # тень ~1/3–1/2 кроны, как в подаче
        self.soff = sh.sdir * off
        self.clumps = []
        if kind in ("decid", "shrub", "column"):
            nc = {"decid": int(rs.integers(6, 10)), "shrub": int(rs.integers(3, 5)), "column": 3}[kind]
            self.clumps.append((0.0, 0.0, 0.55))
            for _ in range(nc - 1):
                rr, th = math.sqrt(rs.random()) * 0.62, rs.uniform(0, TAU)
                self.clumps.append((rr * math.cos(th), rr * math.sin(th), rs.uniform(0.30, 0.48)))

    # радиус контура
    def r(self, th):
        r = np.ones_like(th, dtype=F)
        for k, a, p in self.harm:
            r += a * np.cos(k * th + p)
        if self.kind == "spruce":
            r = r * 0.80 + 0.24 * np.abs(np.cos(self.n * th / 2 + self.ph)) ** 3 - 0.03
        else:
            r += self.sc * (np.abs(np.sin(self.n * th / 2 + self.ph)) ** 0.5 - 0.64)
        return self.R * r

    def bb(self, pad=1.2):
        R = self.R * pad
        ox, oy = self.soff
        return self.sh.bbox(self.cx - R + min(0, ox), self.cy - R + min(0, oy),
                            self.cx + R + max(0, ox), self.cy + R + max(0, oy))

    def sd(self, X, Y, cx=None, cy=None):
        cx = self.cx if cx is None else cx
        cy = self.cy if cy is None else cy
        dx, dy = X - cx, Y - cy
        return self.r(np.arctan2(dy, dx)) - np.hypot(dx, dy)

    def shade(self, X, Y):
        L = self.sh.L
        R = self.R
        dx, dy = (X - self.cx) / R, (Y - self.cy) / R
        if self.kind == "spruce":
            rho = np.hypot(dx, dy) + 1e-6
            ux, uy = dx / rho, dy / rho
            b = math.radians(52)
            g = math.sin(b) * (ux * L[0] + uy * L[1]) + math.cos(b) * L[2]
            th = np.arctan2(dy, dx)
            psi = self.n * th / 2 + self.ph
            tl = (-uy * L[0] + ux * L[1])
            g = g + 0.55 * np.sin(2 * psi) * np.sign(tl + 1e-6) * np.abs(tl) ** 0.5 * np.clip(rho * 2, 0, 1)
            g = g - 0.25 * np.clip(rho - 0.2, 0, 1) * 0  # оставлено для настройки
        else:
            nx, ny = dx * 0.92, dy * 0.92
            nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
            g0 = nx * L[0] + ny * L[1] + nz * L[2]
            best_q = np.full(X.shape, 9.0, F)
            gc = np.zeros(X.shape, F)
            for ox, oy, rad in self.clumps:
                ex, ey = (dx - ox) / rad, (dy - oy) / rad
                q = ex * ex + ey * ey
                cz = np.sqrt(np.clip(1 - q, 0, 1))
                gj = ex * L[0] + ey * L[1] + cz * L[2]
                sel = q < best_q
                best_q = np.where(sel, q, best_q)
                gc = np.where(sel, gj, gc)
            gc = np.where(best_q < 1, gc, g0)
            g = 0.55 * g0 + 0.45 * gc
        s = self.sh.sl(self.bb())
        return g

    def masks(self, bb):
        X, Y = grid(bb)
        sd = self.sd(X, Y)
        m = np.clip(sd / 1.2 + 0.5, 0, 1)
        return X, Y, sd, m

    # ---- фаза A: силуэт и тень в общие слои
    def prepare(self):
        bb = self.bb()
        X, Y, sd, m = self.masks(bb)
        s = self.sh.sl(bb)
        self.sh.top[s] = np.maximum(self.sh.top[s], m)
        sd2 = self.sd(X, Y, self.cx + self.soff[0], self.cy + self.soff[1])
        self.sh.shad[s] = np.maximum(self.sh.shad[s], np.clip(sd2 / 1.5 + 0.5, 0, 1))

    # ---- фаза C: отрисовка
    def draw(self, style):
        sh, rs = self.sh, self.rs
        bb = self.bb()
        X, Y, sd, m = self.masks(bb)
        g = self.shade(X, Y)
        shade = np.clip((g + 0.85) / 1.7 + 0.07 * (sh.nmid[sh.sl(bb)] - 0.5), 0, 1)
        if self.kind != "spruce":   # рефлекс по краю теневой стороны
            rim = np.clip(1 - sd / (0.10 * self.R), 0, 1) * (shade < 0.45)
            shade = np.clip(shade + 0.10 * rim, 0, 1)
        form = (self.cx - bb[0], self.cy - bb[1], self.R)
        pal = self.pal
        style.fill(sh, m, bb, pal[0], 0, rs, form=form)
        if self.flat:
            sx, sy = -sh.sdir * 0.14 * self.R
            rimm = m * (1 - shift(m, sx, sy))
            style.fill(sh, rimm, bb, pal[1], 1, rs, form=form)
        else:
            style.fill(sh, m * smoothstep(0.58, 0.46, shade), bb, pal[1], 1, rs, form=form)
            style.fill(sh, m * smoothstep(0.36, 0.25, shade), bb, pal[2], 2, rs, form=form)
        if style.opaque:   # пастель: светлые блики поверх
            style.fill(sh, m * smoothstep(0.74, 0.86, shade), bb, pal[3], 2, rs, form=form)
        if style.line:
            self.ink(style, shade, bb)

    # ---- перо
    def outline_pts(self, th0=None):
        th0 = self.rs.uniform(0, TAU) if th0 is None else th0
        n = max(60, int(self.R * TAU / self.sh.mm(0.5)))
        th = th0 + np.linspace(0, TAU, n, endpoint=False)
        wob = 1 + 0.012 * np.sin(3 * th + self.ph) + 0.008 * np.sin(7 * th)
        r = self.r(th) * wob
        return np.stack([self.cx + r * np.cos(th), self.cy + r * np.sin(th)], 1)

    def ink(self, style, shade, bb, hide=None):
        sh, rs = self.sh, self.rs
        lw = style.line_w
        pts = self.outline_pts()
        if hide is not None:   # для групп: прячем куски, закрытые соседями
            vis = ~hide(pts)
            runs, cur = [], []
            for p, v in zip(pts, vis):
                if v:
                    cur.append(p)
                elif len(cur) > 3:
                    runs.append(np.array(cur)); cur = []
                else:
                    cur = []
            if len(cur) > 3:
                runs.append(np.array(cur))
            if len(runs) == 1 and len(runs[0]) == len(pts):
                closed_curve(sh, pts, lw, rs)
            else:
                for rr in runs:
                    hand_line(sh, rr, lw, rs)
        else:
            closed_curve(sh, pts, lw, rs)
        thin = lw * 0.65
        c = np.array([self.cx, self.cy], F)
        R = self.R
        if self.kind == "spruce":
            tips = [2 * (k * math.pi - self.ph) / self.n for k in range(self.n)]
            for t in tips:
                rr = float(self.r(np.array([t], F))[0]) * 0.96
                d = np.array([math.cos(t), math.sin(t)], F)
                nn = np.array([-d[1], d[0]], F)
                bend = rs.normal(0, 0.03) * R
                us = np.linspace(0.06, 1.0, 8)
                line = c[None] + np.outer(us * rr, d) + np.outer(np.sin(us * math.pi) * bend, nn)
                hand_line(sh, line, thin, rs, 0.9)
                for u in np.arange(0.45, 0.95, 0.12):   # «хвоя» — короткие засечки
                    p = c + d * rr * u
                    for sgn in (-1, 1):
                        q = p + (d * 0.55 + nn * sgn * 0.8) * R * 0.07
                        hand_line(sh, [p, q], thin * 0.8, rs, 0.8, tremor_mm=0.03)
            cv2.circle(sh.ink, (int(c[0]), int(c[1])), max(2, int(sh.mm(0.45))), 1.0, -1, cv2.LINE_AA)
            return
        if self.kind == "column":
            cv2.circle(sh.ink, (int(c[0]), int(c[1])), max(2, int(sh.mm(0.35))), 1.0, -1, cv2.LINE_AA)
        variant = self.pen % 3
        if self.kind in ("decid",) and variant == 0:      # ветви
            def branch(p, ang, L, depth, w):
                if depth == 0 or L < R * 0.06:
                    return
                q = p + np.array([math.cos(ang), math.sin(ang)], F) * L
                mid = (p + q) / 2 + np.array([-math.sin(ang), math.cos(ang)], F) * rs.normal(0, L * 0.08)
                hand_line(sh, [p, mid, q], w, rs, 0.9, tremor_mm=0.05)
                for s_ in (-1, 1):
                    branch(q, ang + s_ * rs.uniform(0.25, 0.6), L * rs.uniform(0.5, 0.7), depth - 1, w * 0.8)
            for k in range(int(rs.integers(4, 7))):
                a0 = k * TAU / 6 + rs.normal(0, 0.3)
                branch(c, a0, R * rs.uniform(0.28, 0.40), 3, thin)
        if self.kind in ("decid", "shrub", "column") and variant in (1, 2) or self.kind == "shrub":
            # нижние (теневые) дуги «облачков» листвы
            ta = sh.shadow_theta
            for ox, oy, rad in self.clumps[1:] if variant == 1 else self.clumps[1:3]:
                cc = c + np.array([ox, oy], F) * R
                rr = rad * R * 0.95
                sp = rs.uniform(1.6, 2.4)
                ts = np.linspace(ta - sp / 2, ta + sp / 2, 24)
                arc = np.stack([cc[0] + rr * np.cos(ts), cc[1] + rr * np.sin(ts)], 1)
                inside = self.sd(arc[:, 0], arc[:, 1]) > sh.mm(0.8)
                if inside.sum() > 5:
                    hand_line(sh, arc[inside], thin, rs, 0.85)
            # мелкие «гребешки» на теневой стороне
            n = int((R / sh.ppmm) * (1.2 if variant == 1 else 0.6))
            for _ in range(n):
                rr = math.sqrt(rs.uniform(0.15, 0.85)) * R
                t = sh.shadow_theta + rs.normal(0, 0.9)
                p = c + np.array([math.cos(t), math.sin(t)], F) * rr
                rad = R * rs.uniform(0.04, 0.07)
                ts = np.linspace(ta - 1.3, ta + 1.3, 10)
                hand_line(sh, np.stack([p[0] + rad * np.cos(ts), p[1] + rad * np.sin(ts)], 1),
                          thin * 0.85, rs, 0.8, tremor_mm=0.03)


class Group:
    """Группа кустов: несколько слившихся крон, контуры перекрываются честно."""

    def __init__(self, sh, centers_m, diams_m, seed, kind="shrub"):
        self.sh = sh
        self.items = [Crown(sh, kind, c, d, seed + 17 * i, pen=1) for i, (c, d) in
                      enumerate(zip(centers_m, diams_m))]
        self.rs = np.random.default_rng(seed)

    def bb(self):
        bbs = [c.bb() for c in self.items]
        return (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))

    def prepare(self):
        for c in self.items:
            c.prepare()

    def draw(self, style):
        sh = self.sh
        bb = self.bb()
        from objects import grid as _g
        X, Y = _g(bb)
        m = np.zeros(X.shape, F)
        shade = np.zeros(X.shape, F)
        for c in self.items:   # верхний (последний) перекрывает
            mi = np.clip(c.sd(X, Y) / 1.2 + 0.5, 0, 1)
            g = c.shade(X, Y)
            si = np.clip((g + 0.85) / 1.7 + 0.07 * (sh.nmid[sh.sl(bb)] - 0.5), 0, 1)
            shade = np.where(mi > 0.5, si, shade)
            m = np.maximum(m, mi)
        c0 = self.items[0]
        cx = np.mean([c.cx for c in self.items]) - bb[0]
        cy = np.mean([c.cy for c in self.items]) - bb[1]
        R = max(np.hypot(c.cx - bb[0] - cx, c.cy - bb[1] - cy) + c.R for c in self.items)
        form = (cx, cy, R)
        rs, pal = self.rs, c0.pal
        style.fill(sh, m, bb, pal[0], 0, rs, form=form)
        style.fill(sh, m * smoothstep(0.58, 0.46, shade), bb, pal[1], 1, rs, form=form)
        style.fill(sh, m * smoothstep(0.36, 0.25, shade), bb, pal[2], 2, rs, form=form)
        if style.opaque:
            style.fill(sh, m * smoothstep(0.74, 0.86, shade), bb, pal[3], 2, rs, form=form)
        if style.line:
            for i, c in enumerate(self.items):
                later = self.items[i + 1:]
                hide = (lambda P, later=later: np.any([o.sd(P[:, 0], P[:, 1]) > 1.5 for o in later], axis=0)
                        if later else np.zeros(len(P), bool))
                c.ink(style, None, bb, hide=hide)


# ================================================================ полигональные объекты
def rot(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.asarray(pts, F) @ np.array([[c, s], [-s, c]], F)


def rounded_rect(hx, hy, r, n=6):
    pts = []
    for cx, cy, a0 in ((hx - r, hy - r, 0), (-hx + r, hy - r, 90), (-hx + r, -hy + r, 180), (hx - r, -hy + r, 270)):
        for t in np.linspace(math.radians(a0), math.radians(a0 + 90), n):
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return np.array(pts, F)


class Solid:
    """Базовый класс: полигоны в метрах (локально) -> маски, падающая тень с высотой."""
    height = 1.0

    def mask(self, polys_px, bb):
        return self.sh.poly_mask(polys_px, bb)

    def own_rim(self, m, k_px):
        sx, sy = -self.sh.sdir * k_px
        return m * (1 - shift(m, sx, sy))

    def prep_masks(self, body_px, pad_extra=0):
        sh = self.sh
        off = sh.sdir * self.height * sh.shadow_k * sh.ppm
        P = np.vstack(body_px)
        pad = sh.mm(2) + pad_extra
        bb = sh.bbox(P[:, 0].min() - pad + min(0, off[0]), P[:, 1].min() - pad + min(0, off[1]),
                     P[:, 0].max() + pad + max(0, off[0]), P[:, 1].max() + pad + max(0, off[1]))
        m = self.mask(body_px, bb)
        # тень = «протягивание» силуэта по направлению света (сумма Минковского)
        polys = [p + off for p in body_px]
        for p in body_px:
            for i in range(len(p)):
                a, b = p[i], p[(i + 1) % len(p)]
                polys.append(np.array([a, b, b + off, a + off], F))
        sm = self.mask(polys, bb)
        s = sh.sl(bb)
        sh.top[s] = np.maximum(sh.top[s], m)
        sh.shad[s] = np.maximum(sh.shad[s], sm)
        return bb, m


class Car(Solid):
    height = 1.45
    COLORS = [(0.80, 0.38, 0.33), (0.45, 0.58, 0.74), (0.96, 0.96, 0.95), (0.55, 0.56, 0.58),
              (0.86, 0.72, 0.42), (0.55, 0.66, 0.60)]

    def __init__(self, sh, xy_m, ang_deg, color_i, seed, L=4.5, W=1.8):
        self.sh, self.rs = sh, np.random.default_rng(seed)
        self.c = np.asarray(xy_m, F)
        self.a = math.radians(ang_deg)
        self.color = self.COLORS[color_i % len(self.COLORS)]
        hx, hy = L / 2, W / 2
        self.loc = {
            "body": rounded_rect(hx, hy, 0.42, 7),
            "wind": np.array([(0.42, -0.66), (1.02, -0.78), (1.02, 0.78), (0.42, 0.66)], F),
            "rear": np.array([(-1.65, -0.72), (-1.22, -0.66), (-1.22, 0.66), (-1.65, 0.72)], F),
            "mirL": np.array([(0.55, -hy), (0.75, -hy), (0.70, -hy - 0.16), (0.58, -hy - 0.16)], F),
            "mirR": np.array([(0.55, hy), (0.75, hy), (0.70, hy + 0.16), (0.58, hy + 0.16)], F),
        }

    def P(self, k):
        return self.sh.px(rot(self.loc[k], self.a) + self.c)

    def prepare(self):
        self.bb_, _ = self.prep_masks([self.P("body")])

    def draw(self, style):
        sh, rs, bb = self.sh, self.rs, self.bb_
        body = self.mask([self.P("body"), self.P("mirL"), self.P("mirR")], bb)
        glass = self.mask([self.P("wind"), self.P("rear")], bb)
        col = np.asarray(self.color, F)
        pal = [tuple(col), tuple(col * 0.78), (0.30, 0.36, 0.44)]
        ang = self.a
        if not isinstance(style, Pen):
            if col.min() < 0.9:
                style.fill(sh, body, bb, pal[0], 0, rs, angle=ang)
            style.fill(sh, self.own_rim(body, sh.ppm * 0.30), bb,
                       pal[1] if col.min() < 0.9 else (0.78, 0.78, 0.80), 1, rs, angle=ang)
            style.fill(sh, glass, bb, pal[2], 0, rs, angle=ang + 0.8)
        else:
            style.fill(sh, glass, bb, None, 2, rs, angle=ang)
        if style.line:
            w = style.line_w * 0.8
            closed_curve(sh, self.P("body"), w, rs, gap_prob=0.2)
            for k in ("wind", "rear"):
                poly_outline(sh, self.P(k), w * 0.8, rs, overshoot_mm=0.2)


class Person(Solid):
    height = 1.7
    COLORS = [(0.78, 0.40, 0.36), (0.40, 0.52, 0.70), (0.86, 0.70, 0.36), (0.42, 0.62, 0.58),
              (0.60, 0.55, 0.65), (0.70, 0.70, 0.72)]

    def __init__(self, sh, xy_m, ang_deg, color_i, seed):
        self.sh, self.rs = sh, np.random.default_rng(seed)
        self.c, self.a = np.asarray(xy_m, F), math.radians(ang_deg)
        self.color = self.COLORS[color_i % len(self.COLORS)]
        t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        self.loc_body = np.stack([0.15 * np.cos(t), 0.27 * np.sin(t)], 1)
        self.head_c = np.array([0.03, 0.0], F)

    def prepare(self):
        sh = self.sh
        body = sh.px(rot(self.loc_body, self.a) + self.c)
        self.body = body
        # тень человека — вытянутый эллипс от ног
        d = sh.sdir
        t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        Ls = self.height * sh.shadow_k * 0.5
        e = np.stack([Ls * np.cos(t), 0.14 * np.sin(t)], 1)
        ang = math.atan2(d[1], d[0])
        sp = sh.px(rot(e, -ang) + self.c + np.array([d[0], d[1]], F) * Ls)
        P = np.vstack([body, sp])
        pad = sh.mm(1.5)
        self.bb_ = sh.bbox(P[:, 0].min() - pad, P[:, 1].min() - pad, P[:, 0].max() + pad, P[:, 1].max() + pad)
        s = sh.sl(self.bb_)
        m = self.mask([body], self.bb_)
        sh.top[s] = np.maximum(sh.top[s], m)
        sh.shad[s] = np.maximum(sh.shad[s], self.mask([sp], self.bb_))

    def draw(self, style):
        sh, rs, bb = self.sh, self.rs, self.bb_
        m = self.mask([self.body], bb)
        if not isinstance(style, Pen):
            style.fill(sh, m, bb, self.color, 0, rs, angle=self.a)
        if style.line:
            closed_curve(sh, self.body, style.line_w * 0.7, rs, gap_prob=0.0)
        hc = sh.px(self.c + rot(self.head_c[None], self.a)[0])
        r = max(2, int(0.10 * sh.ppm))
        if isinstance(style, Pastel):
            hm = np.zeros(m.shape, F)
            cv2.circle(hm, (int(hc[0] - bb[0]), int(hc[1] - bb[1])), r, 1.0, -1, cv2.LINE_AA)
            sh.over(hm * 0.9, (0.30, 0.26, 0.24), bb)
        else:
            cv2.circle(sh.ink, (int(hc[0]), int(hc[1])), r, 0.85, -1, cv2.LINE_AA)


class Bench(Solid):
    """Параметрическая скамейка-дуга вокруг дерева."""
    height = 0.45

    def __init__(self, sh, center_m, r_m, a0_deg, a1_deg, seed, width=0.55):
        self.sh, self.rs = sh, np.random.default_rng(seed)
        t = np.radians(np.linspace(a0_deg, a1_deg, 40))
        c = np.asarray(center_m, F)
        r1, r2 = r_m, r_m + width
        outer = np.stack([c[0] + r2 * np.cos(t), c[1] + r2 * np.sin(t)], 1)
        inner = np.stack([c[0] + r1 * np.cos(t[::-1]), c[1] + r1 * np.sin(t[::-1])], 1)
        self.poly = sh.px(np.vstack([outer, inner]))
        self.slats = [sh.px(np.array([[c[0] + r1 * math.cos(a), c[1] + r1 * math.sin(a)],
                                      [c[0] + r2 * math.cos(a), c[1] + r2 * math.sin(a)]], F))
                      for a in np.radians(np.linspace(a0_deg, a1_deg, 18))[1:-1]]

    def prepare(self):
        self.bb_, _ = self.prep_masks([self.poly])

    def draw(self, style):
        sh, rs, bb = self.sh, self.rs, self.bb_
        m = self.mask([self.poly], bb)
        if not isinstance(style, Pen):
            style.fill(sh, m, bb, PAL["wood"][0], 0, rs)
            style.fill(sh, self.own_rim(m, sh.ppm * 0.12), bb, PAL["wood"][2], 1, rs)
        if style.line:
            closed_curve(sh, self.poly, style.line_w * 0.8, rs, gap_prob=0.0)
            for s_ in self.slats:
                hand_line(sh, s_, style.line_w * 0.45, rs, 0.6, tremor_mm=0.02)


class Building(Solid):
    height = 9.0

    def __init__(self, sh, poly_m, seed):
        self.sh, self.rs = sh, np.random.default_rng(seed)
        self.poly = sh.px(poly_m)

    def prepare(self):
        self.bb_, _ = self.prep_masks([self.poly])

    def draw(self, style):
        sh, rs, bb = self.sh, self.rs, self.bb_
        m = self.mask([self.poly], bb)
        if isinstance(style, Pen):
            style.fill(sh, m, bb, None, 1, rs, angle=math.radians(-45) - 0.0, sparse=1.0)
        else:
            style.fill(sh, m, bb, PAL["roof"][0], 0, rs, angle=math.radians(0))
            # парапет: тень у внутренних кромок
            inner = cv2.erode(m, np.ones((int(sh.ppm * 0.5) | 1,) * 2, np.uint8))
            style.fill(sh, m - inner * 1.0, bb, PAL["roof"][1], 1, rs, angle=math.radians(0))
        if style.line:
            poly_outline(sh, self.poly, style.line_w * 2.2, rs, overshoot_mm=1.4)
            # внутренний контур парапета
            c = self.poly.mean(0)
            cnt, _ = cv2.findContours((cv2.erode(m, np.ones((int(sh.ppm * 0.5) | 1,) * 2, np.uint8)) > 0.5)
                                      .astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for ct in cnt:
                p = ct[:, 0, :].astype(F) + np.array([bb[0], bb[1]], F)
                poly_outline(sh, p, style.line_w * 0.7, rs, overshoot_mm=0.5, alpha=0.8)
