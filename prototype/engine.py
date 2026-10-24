"""Процедурный движок ручной подачи генплана (прототип).

Принципы (из методик кафедр):
- порядок слоёв: контур -> светлый тон -> объём (собственная тень) -> тёмный тон -> падающая тень
- падающая тень считается движком отдельно, в одном направлении на весь лист
- собственная тень считается от глобального света, поэтому символ можно вращать
- размеры штрихов задаются в мм листа, а не в метрах
- минимум 3 оттенка на цвет
"""
import math
import numpy as np
import cv2

F = np.float32


# ---------------------------------------------------------------- утилиты
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fbm(h, w, scale, octaves=4, seed=0, pers=0.5):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), F)
    amp, tot, s = 1.0, 0.0, float(scale)
    for _ in range(octaves):
        if s < 1.5:
            break
        gh, gw = int(math.ceil(h / s)) + 3, int(math.ceil(w / s)) + 3
        g = rng.random((gh, gw)).astype(F)
        up = cv2.resize(g, (int(math.ceil(gw * s)), int(math.ceil(gh * s))),
                        interpolation=cv2.INTER_CUBIC)
        oy, ox = rng.integers(0, max(1, int(s))), rng.integers(0, max(1, int(s)))
        out += amp * up[oy:oy + h, ox:ox + w]
        tot += amp
        amp *= pers
        s /= 2
    out /= tot
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1).astype(F)


def resample(pts, step):
    pts = np.asarray(pts, F)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    if L[-1] < 1e-3:
        return pts
    n = max(2, int(L[-1] / step) + 1)
    u = np.linspace(0, L[-1], n)
    return np.stack([np.interp(u, L, pts[:, 0]), np.interp(u, L, pts[:, 1])], 1).astype(F)


def shift(m, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def raster(h, w, strokes, thick):
    L = np.zeros((h, w), np.uint8)
    for i, pts in enumerate(strokes):
        t = thick[i] if isinstance(thick, (list, tuple, np.ndarray)) else thick
        cv2.polylines(L, [np.round(np.asarray(pts) * 16).astype(np.int32)], False, 255,
                      int(max(1, round(t))), cv2.LINE_AA, shift=4)
    return L.astype(F) / 255.0


def runs_along(a, p0, d, umin, umax, step, thr):
    us = np.arange(umin, umax, step)
    pts = p0[None, :] + np.outer(us, d)
    xi, yi = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    h, w = a.shape
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    v = np.zeros(len(us), F)
    v[ok] = a[yi[ok], xi[ok]]
    ins = (v > thr).astype(int)
    df = np.diff(np.concatenate([[0], ins, [0]]))
    st, en = np.where(df == 1)[0], np.where(df == -1)[0]
    return [(us[s], us[e - 1]) for s, e in zip(st, en)]


def parallel_strokes(a, angle, spacing, rs, overshoot, thr=0.3, jit=0.04, bow=0.08):
    h, w = a.shape
    n0 = np.array([-math.sin(angle), math.cos(angle)], F)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], F)
    tp = corners @ n0
    d0 = np.array([math.cos(angle), math.sin(angle)], F)
    up = corners @ d0
    out = []
    t = tp.min() + rs.uniform(0, spacing)
    while t < tp.max():
        ang = angle + rs.normal(0, jit)
        d = np.array([math.cos(ang), math.sin(ang)], F)
        nn = np.array([-d[1], d[0]], F)
        p0 = n0 * t
        for u0, u1 in runs_along(a, p0, d, up.min() - 5, up.max() + 5, max(2.0, spacing / 5), thr):
            u0 -= rs.uniform(-0.5, 1.0) * overshoot
            u1 += rs.uniform(-0.5, 1.0) * overshoot
            if u1 - u0 < 2:
                continue
            us = np.linspace(u0, u1, 7)
            off = rs.normal(0, spacing * bow) * np.sin(np.linspace(0, math.pi, 7))
            out.append(p0[None, :] + np.outer(us, d) + np.outer(off, nn))
        t += spacing * rs.uniform(0.85, 1.12)
    return out


def sample_points(a, n, rs, thr=0.5):
    ys, xs = np.nonzero(a > thr)
    if len(xs) == 0 or n <= 0:
        return np.zeros((0, 2), F)
    idx = rs.choice(len(xs), size=min(n, len(xs) * 4), replace=True)
    return np.stack([xs[idx], ys[idx]], 1).astype(F) + rs.uniform(-0.5, 0.5, (len(idx), 2))


def line_kernel(k, ang):
    K = np.zeros((k, k), F)
    c = k // 2
    dx, dy = math.cos(ang) * c, math.sin(ang) * c
    cv2.line(K, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1.0, 1)
    return K / K.sum()


# ---------------------------------------------------------------- лист
class Sheet:
    def __init__(self, w_mm, h_mm, dpi=300, scale=200, seed=7, paper=(0.988, 0.978, 0.952),
                 marker_angle=-38, shadow_deg=40, shadow_k=0.55, light_elev=35):
        self.ppmm = dpi / 25.4
        self.W, self.H = int(round(w_mm * self.ppmm)), int(round(h_mm * self.ppmm))
        self.ppm = self.ppmm * 1000.0 / scale
        self.img = np.ones((self.H, self.W, 3), F) * np.array(paper, F)
        rng = np.random.default_rng(seed)
        H, W = self.H, self.W
        self.grain = (0.65 * fbm(H, W, self.ppmm * 0.45, 2, seed) +
                      0.35 * rng.random((H, W)).astype(F))
        self.nmid = fbm(H, W, self.ppmm * 2.5, 3, seed + 1)
        self.nmid2 = fbm(H, W, self.ppmm * 1.6, 3, seed + 5)
        self.nlow = fbm(H, W, self.ppmm * 12, 3, seed + 2)
        self.nlow2 = fbm(H, W, self.ppmm * 9, 3, seed + 3)
        self.marker_angle = math.radians(marker_angle)
        k = int(self.ppmm * 5) | 1
        st = cv2.filter2D(rng.random((H, W)).astype(F), -1, line_kernel(k, self.marker_angle))
        self.streak = np.clip((st - st.mean()) / (st.std() * 3 + 1e-6) + 0.5, 0, 1)
        self.ink = np.zeros((H, W), F)
        self.top = np.zeros((H, W), F)
        self.shad = np.zeros((H, W), F)
        a = math.radians(shadow_deg)
        self.sdir = np.array([math.cos(a), math.sin(a)], F)      # куда падают тени
        self.shadow_k = shadow_k                                  # длина тени / высота
        e = math.radians(light_elev)
        self.L = np.array([-self.sdir[0] * math.cos(e), -self.sdir[1] * math.cos(e), math.sin(e)], F)
        self.shadow_theta = math.atan2(self.sdir[1], self.sdir[0])

    # координаты
    def px(self, xy_m, origin_mm=(10, 10)):
        xy = np.asarray(xy_m, F)
        return xy * self.ppm + np.array(origin_mm, F) * self.ppmm

    def mm(self, v):
        return v * self.ppmm

    def bbox(self, x0, y0, x1, y1):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.W, int(math.ceil(x1))), min(self.H, int(math.ceil(y1)))
        return (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))

    def full(self):
        return (0, 0, self.W, self.H)

    @staticmethod
    def sl(bb):
        return (slice(bb[1], bb[3]), slice(bb[0], bb[2]))

    # композиция
    def mul(self, a, color, bb):
        v = self.img[self.sl(bb)]
        v *= 1.0 - a[..., None] * (1.0 - np.asarray(color, F))

    def over(self, a, color, bb):
        v = self.img[self.sl(bb)]
        c = np.asarray(color, F)
        v[:] = v * (1.0 - a[..., None]) + c * a[..., None]

    def beer(self, d, color, bb):
        v = self.img[self.sl(bb)]
        c = np.asarray(color, F)
        v *= np.power(np.clip(c, 1e-3, 1.0), d[..., None])

    def ink_add(self, layer, bb, alpha=1.0):
        s = self.sl(bb)
        self.ink[s] = np.maximum(self.ink[s], layer * alpha)

    def poly_mask(self, polys_px, bb=None):
        bb = bb or self.full()
        h, w = bb[3] - bb[1], bb[2] - bb[0]
        m = np.zeros((h, w), np.uint8)
        pl = [np.round((np.asarray(p, F) - np.array([bb[0], bb[1]], F)) * 16).astype(np.int32)
              for p in polys_px]
        for p in pl:   # объединение (fillPoly с несколькими контурами даёт XOR)
            t = np.zeros_like(m)
            cv2.fillPoly(t, [p], 255, cv2.LINE_AA, shift=4)
            np.maximum(m, t, out=m)
        return m.astype(F) / 255.0

    def finish(self, style):
        a = self.ink * style.line_alpha
        if style.line_tooth:
            a = a * np.clip((self.grain - 0.2) * 2.2, 0, 1)
        self.img *= 1.0 - a[..., None] * (1.0 - np.asarray(style.line_color, F))
        self.img *= (0.965 + 0.035 * self.grain)[..., None]
        return np.clip(self.img * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- линии от руки
def hand_line(sh, pts, w_mm, rs, alpha=1.0, tremor_mm=0.10, closed=False):
    pts = np.asarray(pts, F)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    p = resample(pts, sh.mm(0.6))
    n = len(p)
    if n < 2:
        return
    noise = rs.normal(0, 1, (n + 16, 2)).astype(F)
    k = np.hanning(15).astype(F)
    k /= k.sum()
    noise = np.stack([np.convolve(noise[:, 0], k, 'same'), np.convolve(noise[:, 1], k, 'same')], 1)[8:8 + n]
    noise /= (noise.std() + 1e-6)
    p = p + noise * sh.mm(tremor_mm)
    pad = sh.mm(w_mm) + 4
    bb = sh.bbox(p[:, 0].min() - pad, p[:, 1].min() - pad, p[:, 0].max() + pad, p[:, 1].max() + pad)
    h, w = bb[3] - bb[1], bb[2] - bb[0]
    L = np.zeros((h, w), np.uint8)
    q = p - np.array([bb[0], bb[1]], F)
    i = 0
    while i < n - 1:
        j = min(n, i + int(rs.integers(12, 40)))
        t = max(1, int(round(sh.mm(w_mm) * rs.uniform(0.8, 1.3))))
        cv2.polylines(L, [np.round(q[i:j] * 16).astype(np.int32)], False, 255, t, cv2.LINE_AA, shift=4)
        i = j - 1
    sh.ink_add(L.astype(F) / 255.0, bb, alpha)


def poly_outline(sh, poly, w_mm, rs, overshoot_mm=1.0, alpha=1.0):
    """Ребра отдельными линиями с перехлёстом на углах, как рисуют архитекторы."""
    P = np.asarray(poly, F)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        d = b - a
        L = np.linalg.norm(d)
        if L < 1:
            continue
        d /= L
        o1 = sh.mm(overshoot_mm) * rs.uniform(0.3, 1.2)
        o2 = sh.mm(overshoot_mm) * rs.uniform(0.3, 1.2)
        hand_line(sh, [a - d * o1, b + d * o2], w_mm, rs, alpha)


def closed_curve(sh, pts, w_mm, rs, alpha=1.0, gap_prob=0.5):
    """Замкнутый контур с перехлёстом в начале и (иногда) разрывом-«перезаходом»."""
    P = np.asarray(pts, F)
    n = len(P)
    over = int(n * rs.uniform(0.03, 0.08))
    if rs.random() < gap_prob:
        k = int(rs.integers(n // 3, 2 * n // 3))
        hand_line(sh, P[:k + 2], w_mm, rs, alpha)
        hand_line(sh, np.vstack([P[k - 1:], P[:over]]), w_mm, rs, alpha)
    else:
        hand_line(sh, np.vstack([P, P[:over]]), w_mm, rs, alpha)


# ---------------------------------------------------------------- стили
class Style:
    name = "base"
    line = True
    line_color = (0.10, 0.10, 0.12)
    line_alpha = 0.95
    line_tooth = False
    line_w = 0.30          # мм, основной контур
    opaque = False
    shadow_color = (0.62, 0.66, 0.64)

    def fill(self, sh, a, bb, color, level, rs, form=None, sparse=1.0, angle=None):
        raise NotImplementedError

    def shadow(self, sh, a, bb, rs):
        self.fill(sh, a, bb, self.shadow_color, 0, rs)

    def lawn(self, sh, a, bb, pal, rs):
        self.fill(sh, a, bb, pal[0], 0, rs)
        self.fill(sh, a, bb, pal[1], 1, rs, sparse=0.35)
        self.fill(sh, a, bb, pal[2], 2, rs, sparse=0.25)

    def stipple(self, sh, a, bb, rs, per_mm2, r_mm=0.16, alpha=0.8):
        pass


class Marker(Style):
    name = "pero_marker"

    def __init__(self, ink=True):
        self.line = ink
        self.w_mm = (5.0, 3.8, 2.6)
        self.alpha = (0.80, 0.85, 0.88)

    def fill(self, sh, a, bb, color, level, rs, form=None, sparse=1.0, angle=None):
        if a.max() < 0.05:
            return
        h, w = a.shape
        s = sh.sl(bb)
        W = sh.mm(self.w_mm[level])
        alpha = self.alpha[level]
        clip = np.clip(cv2.GaussianBlur(a, (0, 0), sh.mm(0.5)) * 2.0, 0, 1)
        streak = 0.75 + 0.25 * sh.streak[s]
        ang = sh.marker_angle if angle is None else angle
        if level == 0 or form is None:
            if level == 2 and form is None:
                W = sh.mm(self.w_mm[1])
            st = parallel_strokes(a, ang + (0.10 if level == 1 else 0.0), W * 0.82, rs, W * 0.25)
            if sparse < 1.0:   # «уривчасті рухи»: короткие обрывки штрихов
                out = []
                for p in st:
                    q = resample(p, W * 0.5)
                    i = 0
                    while i < len(q) - 2:
                        j = i + int(rs.integers(4, 12))
                        if rs.random() < sparse:
                            out.append(q[i:min(j, len(q))])
                        i = j + int(rs.integers(2, 10))
                st = [p for p in out if len(p) >= 2]
                alpha *= 0.8
            for grp in (st[0::2], st[1::2]):
                if grp:
                    cov = raster(h, w, grp, W) * clip * streak
                    sh.mul(cov * alpha, color, bb)
            return
        cx, cy, R = form
        if level == 1:      # дуги по форме на теневой стороне
            area = float(a.sum())
            n = int(area / (W * R * 0.9) * 1.3) + 3
            groups = [[], [], []]
            for k in range(n):
                rr = R * rs.uniform(0.25, 1.0)
                thc = sh.shadow_theta + rs.normal(0, 0.75)
                span = rs.uniform(0.6, 1.5)
                ts = np.linspace(thc - span / 2, thc + span / 2, 14)
                groups[k % 3].append(np.stack([cx + rr * np.cos(ts), cy + rr * np.sin(ts)], 1))
            for g in groups:
                if g:
                    cov = raster(h, w, g, W) * clip * streak
                    sh.mul(cov * alpha, color, bb)
        else:               # короткие мазки в глубокой тени
            area = float(a.sum())
            n = int(area / (W * W * 3.0)) + 2
            P = sample_points(a, n, rs, 0.45)
            groups = [[], [], []]
            for k, (x, y) in enumerate(P):
                th = math.atan2(y - cy, x - cx) + math.pi / 2 + rs.normal(0, 0.5)
                l = W * rs.uniform(0.8, 2.0)
                d = np.array([math.cos(th), math.sin(th)], F)
                groups[k % 3].append(np.array([[x, y] - d * l / 2, [x, y] + d * l / 2], F))
            for g in groups:
                if g:
                    cov = raster(h, w, g, W) * clip * streak
                    sh.mul(cov * alpha, color, bb)

    def lawn(self, sh, a, bb, pal, rs):
        self.fill(sh, a, bb, pal[0], 0, rs)
        # «уривчасті рухи, нерівномірно» — прерывистые штрихи вторым тоном и охрой
        self.fill(sh, a, bb, pal[1], 1, rs, sparse=0.38)
        self.fill(sh, a, bb, pal[2], 2, rs, sparse=0.18)


class Pencil(Style):
    name = "karandash"
    line_color = (0.30, 0.30, 0.33)
    line_alpha = 0.85
    line_tooth = True
    line_w = 0.25
    shadow_color = (0.52, 0.54, 0.58)

    def __init__(self):
        self.ang = (math.radians(55), math.radians(-35), math.radians(82))
        self.spacing = (1.15, 0.95, 0.75)
        self.pressure = (0.70, 0.80, 0.90)

    def fill(self, sh, a, bb, color, level, rs, form=None, sparse=1.0, angle=None):
        if a.max() < 0.05:
            return
        h, w = a.shape
        s = sh.sl(bb)
        ang = self.ang[level] if angle is None else angle + (0, 1.4, 0.5)[level]
        sp = sh.mm(self.spacing[level])
        lines = parallel_strokes(a, ang, sp, rs, sh.mm(0.8), jit=0.02, bow=0.15)
        dashes = []
        Ld = sh.mm(7.0)
        for pts in lines:
            p = resample(pts, sh.mm(0.5))
            i = 0
            while i < len(p) - 1:
                j = min(len(p), i + max(2, int(Ld / sh.mm(0.5) * rs.uniform(0.6, 1.3))))
                if rs.random() < sparse:
                    seg = p[i:j].copy()
                    c = seg.mean(0)
                    th = rs.normal(0, 0.05)
                    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], F)
                    dashes.append((seg - c) @ R.T + c)
                i = j + int(rs.integers(0, 3))
        if not dashes:
            return
        lay = raster(h, w, dashes, max(1.0, sh.mm(0.32)))
        tooth = np.clip((sh.grain[s] - (1 - self.pressure[level])) * 2.6, 0, 1)
        clip = np.clip(cv2.GaussianBlur(a, (0, 0), sh.mm(0.4)) * 2.0, 0, 1)
        sh.mul(lay * tooth * clip * 0.92, color, bb)

    def shadow(self, sh, a, bb, rs):
        self.fill(sh, a, bb, self.shadow_color, 1, rs, angle=math.radians(45) - 1.4)

    def lawn(self, sh, a, bb, pal, rs):
        self.fill(sh, a, bb, (0.80, 0.88, 0.56), 0, rs, sparse=0.55)
        self.fill(sh, a, bb, pal[2], 1, rs, sparse=0.12)


class Pastel(Style):
    name = "pastel"
    line = True
    line_color = (0.22, 0.28, 0.24)
    line_alpha = 0.55
    line_tooth = True
    line_w = 0.35
    opaque = True
    shadow_color = (0.50, 0.56, 0.53)

    def __init__(self):
        self.w_mm = (7.0, 5.5, 4.2)
        self.press = (0.80, 0.82, 0.78)

    def fill(self, sh, a, bb, color, level, rs, form=None, sparse=1.0, angle=None, opacity=0.95):
        if a.max() < 0.05:
            return
        h, w = a.shape
        s = sh.sl(bb)
        W = sh.mm(self.w_mm[level])
        Ls = sh.mm(11.0)
        n = int(float(a.sum()) / (W * Ls) * 2.4 * sparse) + 1
        P = sample_points(a, n, rs, 0.3)
        st = []
        for x, y in P:
            if form is not None:
                th = math.atan2(y - form[1], x - form[0]) + math.pi / 2 + rs.normal(0, 0.35)
            else:
                th = (sh.marker_angle if angle is None else angle) + rs.normal(0, 0.45)
            d = np.array([math.cos(th), math.sin(th)], F)
            l = Ls * rs.uniform(0.6, 1.2)
            st.append(np.array([[x, y] - d * l / 2, [x, y], [x, y] + d * l / 2], F)
                      + np.array([[0, 0], [rs.normal(0, W * 0.1)] * 2, [0, 0]], F))
        lay = cv2.GaussianBlur(raster(h, w, st, W * 0.8), (0, 0), sh.mm(0.35))
        tooth = np.clip((sh.grain[s] - (1 - self.press[level])) * 3.2, 0, 1)
        clip = np.clip(cv2.GaussianBlur(a, (0, 0), sh.mm(0.3)) * 1.3, 0, 1)
        cov = np.clip(lay * 1.25, 0, 1) * tooth * clip
        sh.over(cov * opacity, color, bb)

    def shadow(self, sh, a, bb, rs):
        s = sh.sl(bb)
        tooth = np.clip((sh.grain[s] - 0.15) * 1.6, 0, 1)
        sh.mul(np.clip(a, 0, 1) * (0.30 + 0.25 * tooth), self.shadow_color, bb)

    def lawn(self, sh, a, bb, pal, rs):
        self.fill(sh, a, bb, pal[0], 0, rs, opacity=0.8)
        self.fill(sh, a, bb, pal[1], 1, rs, sparse=0.25, opacity=0.7)


class Watercolor(Style):
    name = "akvarel"
    shadow_color = (0.72, 0.70, 0.82)

    def __init__(self, ink=False):
        self.line = True
        self.ink_on = ink
        if ink:
            self.name = "pero_akvarel"
        else:
            self.line_color = (0.45, 0.45, 0.47)
            self.line_alpha = 0.45
            self.line_tooth = True
            self.line_w = 0.18
        self.dens = (0.95, 0.90, 0.95)
        self.edge_noise = 0.34

    def fill(self, sh, a, bb, color, level, rs, form=None, sparse=1.0, angle=None, color2=None):
        if a.max() < 0.05:
            return
        s = sh.sl(bb)
        nm, nl, g = sh.nmid[s], sh.nlow[s], sh.grain[s]
        a2 = cv2.GaussianBlur(a, (0, 0), sh.mm(0.6))
        m = smoothstep(0.36, 0.64, a2 + self.edge_noise * (nm - 0.5))
        if self.ink_on and level == 0:   # краска «съезжает» с контура пера
            m = shift(m, sh.mm(0.8) * rs.uniform(0.5, 1.2), -sh.mm(0.5) * rs.uniform(0.5, 1.2))
        if level == 0 and form is not None:   # просветы бумаги — только на светлой стороне кроны
            cx, cy, R = form
            h, w = a.shape
            yy, xx = np.mgrid[0:h, 0:w].astype(F)
            lit = np.clip(-((xx - cx) * sh.sdir[0] + (yy - cy) * sh.sdir[1]) / R + 0.2, 0, 1)
            m *= 1.0 - 0.95 * smoothstep(0.84, 0.88, sh.nmid2[s]) * lit
        if sparse < 1.0:
            m *= smoothstep(1 - sparse - 0.05, 1 - sparse + 0.05, sh.nlow2[s])
        b = cv2.GaussianBlur(m, (0, 0), sh.mm(1.0))
        edge = np.clip(m - b, 0, 1)
        d = self.dens[level] * (m * (0.70 + 0.60 * nl) + 2.4 * edge) * (0.80 + 0.40 * g)
        if color2 is None:
            c = np.asarray(color, F)
        else:
            t = sh.nlow2[s][..., None]
            c = np.asarray(color, F) * (1 - t) + np.asarray(color2, F) * t
        sh.beer(d, c, bb)

    def lawn(self, sh, a, bb, pal, rs):
        self.fill(sh, a, bb, pal[0], 0, rs, color2=pal[1])
        self.fill(sh, a, bb, pal[1], 1, rs, sparse=0.30)


class Pen(Style):
    name = "pero"

    def fill(self, sh, a, bb, color, level, rs, form=None, sparse=1.0, angle=None):
        if level == 0 or a.max() < 0.05:
            return
        h, w = a.shape
        ang = math.radians(45) if level == 1 else math.radians(-45)
        if angle is not None:
            ang = angle + (0, 0.8, -0.8)[level]
        st = parallel_strokes(a, ang, sh.mm(1.0 if level == 1 else 0.85), rs, sh.mm(0.3),
                              thr=0.5, jit=0.015, bow=0.05)
        if sparse < 1:
            st = [p for p in st if rs.random() < sparse]
        lay = raster(h, w, st, max(1.0, sh.mm(0.18)))
        sh.ink_add(lay * np.clip(a * 1.5, 0, 1), bb, 0.9)

    def shadow(self, sh, a, bb, rs):
        h, w = a.shape
        st = parallel_strokes(a, math.radians(45), sh.mm(0.6), rs, sh.mm(0.2), thr=0.5, jit=0.01, bow=0.03)
        lay = raster(h, w, st, max(1.0, sh.mm(0.16)))
        sh.ink_add(lay * np.clip(a * 1.5, 0, 1), bb, 0.85)

    def lawn(self, sh, a, bb, pal, rs):
        stipple(sh, a, bb, rs, per_mm2=0.9)


def stipple(sh, a, bb, rs, per_mm2=1.0, r_mm=0.15, alpha=0.85):
    h, w = a.shape
    n = int(float((a > 0.5).sum()) / (sh.ppmm ** 2) * per_mm2)
    P = sample_points(a, n, rs, 0.5)
    # неравномерно: гуще по краям и пятнами
    keep = rs.random(len(P)) < (0.35 + 0.65 * sh.nmid[sh.sl(bb)][P[:, 1].astype(int).clip(0, h - 1),
                                                                    P[:, 0].astype(int).clip(0, w - 1)])
    L = np.zeros((h, w), np.uint8)
    r = max(1, int(round(sh.mm(r_mm))))
    for x, y in P[keep]:
        cv2.circle(L, (int(x * 16), int(y * 16)), r * 16, 255, -1, cv2.LINE_AA, shift=4)
    sh.ink_add(L.astype(F) / 255, bb, alpha)


STYLES = {
    "pero": Pen,
    "karandash": Pencil,
    "pastel": Pastel,
    "akvarel": lambda: Watercolor(ink=False),
    "pero_akvarel": lambda: Watercolor(ink=True),
    "pero_marker": lambda: Marker(ink=True),
}

# ---------------------------------------------------------------- палитры (3+ оттенка)
PAL = {
    "decid":  [(0.80, 0.87, 0.50), (0.56, 0.69, 0.32), (0.31, 0.44, 0.21), (0.94, 0.95, 0.70)],
    "decid2": [(0.86, 0.86, 0.50), (0.66, 0.68, 0.31), (0.40, 0.45, 0.22), (0.96, 0.94, 0.68)],
    "spruce": [(0.62, 0.76, 0.62), (0.37, 0.55, 0.45), (0.19, 0.33, 0.29), (0.80, 0.88, 0.76)],
    "shrub":  [(0.77, 0.86, 0.58), (0.50, 0.67, 0.40), (0.29, 0.43, 0.26), (0.92, 0.95, 0.74)],
    "column": [(0.70, 0.82, 0.52), (0.44, 0.61, 0.33), (0.24, 0.39, 0.22), (0.88, 0.93, 0.70)],
    "lawn":   [(0.86, 0.91, 0.64), (0.70, 0.82, 0.48), (0.84, 0.76, 0.56)],
    "paving": [(0.91, 0.87, 0.79), (0.82, 0.77, 0.68), (0.72, 0.66, 0.58)],
    "asphalt": [(0.80, 0.80, 0.81), (0.70, 0.70, 0.72), (0.60, 0.60, 0.63)],
    "gravel": [(0.92, 0.86, 0.70), (0.84, 0.76, 0.58), (0.74, 0.66, 0.50)],
    "roof":   [(0.86, 0.84, 0.82), (0.74, 0.72, 0.70), (0.60, 0.58, 0.57)],
    "wood":   [(0.86, 0.70, 0.48), (0.72, 0.54, 0.34), (0.55, 0.40, 0.25)],
}
