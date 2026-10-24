import sys, time, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from engine import Sheet, STYLES, PAL, stipple, poly_outline, hand_line, Pen, F
from objects import Crown, Group, Car, Person, Bench, Building

import os
_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
          "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]
FONT = next((f for f in _FONTS if os.path.exists(f)), _FONTS[0])
STYLE_ORDER = ["pero", "karandash", "pastel", "akvarel", "pero_akvarel", "pero_marker"]
STYLE_RU = {"pero": "перо", "karandash": "карандаш", "pastel": "пастель", "akvarel": "акварель",
            "pero_akvarel": "перо + акварель", "pero_marker": "перо + маркер"}


def render(sh, style, objs, ground=(), seed=0):
    rs = np.random.default_rng(seed)
    for o in objs:
        o.prepare()
    for fn in ground:
        fn(sh, style, rs)
    sm = sh.shad * (1 - sh.top)
    if getattr(sh, 'site', None) is not None:
        sm = sm * sh.site
    if sm.max() > 0.05:
        style.shadow(sh, sm, sh.full(), rs)
    for o in objs:
        o.draw(style)
    return sh.finish(style)


# ------------------------------------------------------------ лист символов
def symbol_cell(row, style_key, seed):
    sh = Sheet(80, 80, dpi=200, scale=100, seed=seed)
    st = STYLES[style_key]()
    c = (2.3, 2.3)
    if row == 0:
        objs = [Crown(sh, "decid", c, 4.4, seed, pen=1)]
    elif row == 1:
        objs = [Crown(sh, "decid", c, 4.4, seed + 3, pal="decid2", flat=True, pen=0)]
    elif row == 2:
        objs = [Crown(sh, "spruce", c, 4.0, seed + 5)]
    else:
        objs = [Group(sh, [(1.6, 2.7), (3.1, 2.8), (2.3, 1.7)], [2.0, 1.8, 2.1], seed + 7)]
    return render(sh, st, objs, seed=seed)


def symbol_sheet(path):
    rows = ["лиственное, объём", "лиственное, плоско + ветви", "ель", "группа кустов"]
    cells = []
    t0 = time.time()
    for r in range(4):
        row = []
        for j, k in enumerate(STYLE_ORDER):
            row.append(symbol_cell(r, k, 100 + r * 10 + j))
        cells.append(row)
        print("row", r, round(time.time() - t0, 1), "s", flush=True)
    ch, cw = cells[0][0].shape[:2]
    top, left = 70, 230
    W, H = left + cw * 6, top + ch * 4
    im = Image.new("RGB", (W, H), (252, 250, 244))
    for r in range(4):
        for j in range(6):
            im.paste(Image.fromarray(cells[r][j]), (left + j * cw, top + r * ch))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 30)
    for j, k in enumerate(STYLE_ORDER):
        tw = d.textlength(STYLE_RU[k], font=f)
        d.text((left + j * cw + (cw - tw) / 2, 22), STYLE_RU[k], fill=(60, 60, 60), font=f)
    f2 = ImageFont.truetype(FONT, 26)
    for r, name in enumerate(rows):
        y = top + r * ch + ch // 2
        for i, part in enumerate(name.split(", ")):
            d.text((18, y - 30 + i * 34), part, fill=(60, 60, 60), font=f2)
    im.save(path, quality=92)
    return im


if __name__ == "__main__":
    symbol_sheet(sys.argv[1] if len(sys.argv) > 1 else "symbols.jpg")
