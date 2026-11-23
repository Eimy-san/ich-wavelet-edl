"""Patch Fig.3.png and Fig.4.png annotation legibility:
- Fig3 b/c: move + enlarge the semi-transparent gray '95% coverage' labels to
  solid dark text placed clear of curves, left of the dashed reference line.
- Fig3 d: redraw the 16 rotated bar value labels larger and in solid dark gray.
- Fig4 a: move the frameless legend (which sits on top of curves) to the empty
  upper-left region of the axes, redrawing entries verbatim.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import scipy.ndimage as ndi

matplotlib.get_data_path()
FONT = matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"
DARK = (51, 51, 51)


def spread(reg):
    return reg.max(axis=-1) - reg.min(axis=-1)


# ============================ Fig.3 ============================
im = Image.open("revision/manuscript/figures/Fig.3.png").convert("RGB")
a = np.asarray(im).astype(np.int64)


def erase_grayish(y0, y1, x0, x1, lum_hi=225):
    reg = a[y0:y1, x0:x1]
    sp = spread(reg)
    lum = reg.mean(axis=2)
    m = (sp < 50) & (lum < lum_hi)
    reg[m] = [255, 255, 255]
    a[y0:y1, x0:x1] = reg
    return m


def copy_strip(x0, x1, ys, yt, h):
    a[yt : yt + h, x0:x1] = a[ys : ys + h, x0:x1].copy()


# ---- panel b: old label + dashed-line continuation ----
erase_grayish(253, 308, 3644, 3913)
copy_strip(3776, 3785, 420, 253, 55)  # continue dashed vline through erased area

# ---- panel c: old label + dashed-line continuation ----
erase_grayish(2956, 3017, 1649, 1917)
copy_strip(1780, 1789, 2150, 2956, 61)

im2 = Image.fromarray(a.astype(np.uint8))
d = ImageDraw.Draw(im2)
font_lbl = ImageFont.truetype(FONT, 55)

# choose an empty spot for each label by scanning candidates
f32 = np.asarray(im2).astype(np.int64)


def region_clear(y0, y1, x0, x1):
    reg = f32[y0:y1, x0:x1]
    sp = spread(reg)
    lum = reg.mean(axis=2)
    bad = ((sp > 12) | (lum < 235)) & ~((np.abs(reg[:, :, 2] - 120) < 60) & False)
    # gridline/dashline/axis px are gray: allow them
    grayish = sp <= 12
    bad = (sp > 12) | ((lum < 235) & grayish & (lum > 150) & False)
    return (
        not ((sp > 12) & (lum < 250)).any()
        and not ((lum < 240) & (sp <= 12) & (lum < 200)).any()
    )


def place_label(cands, text):
    for cy, xe in cands:
        bb = d.textbbox((0, 0), text, font=font_lbl, anchor="rm")
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        if region_clear(
            int(cy - h / 2 - 4), int(cy + h / 2 + 4), int(xe - w - 4), int(xe + 4)
        ):
            d.text((xe, cy), text, font=font_lbl, fill=DARK, anchor="rm")
            return (cy, xe)
    raise RuntimeError("no clear spot for label " + text)


place_label([(1360, 3756), (1340, 3756), (1300, 3700)], "95% coverage")
place_label([(2400, 1760), (2300, 1760), (2500, 1760), (1900, 1830)], "95% coverage")

# ---- panel d: redraw 16 bar value labels ----
vals = {
    "AUC": (
        [
            (2346, 2369, 1991, 2068, 2357, 2076),
            (2421, 2445, 1985, 2068, 2433, 2071),
            (2494, 2518, 1985, 2068, 2506, 2073),
            (2570, 2593, 1991, 2068, 2581, 2076),
        ],
        ["0.933", "0.938", "0.936", "0.933"],
    ),
    "AURC": (
        [
            (2735, 2759, 2976, 3053, 2747, 3059),
            (2810, 2834, 2979, 3053, 2822, 3061),
            (2885, 2909, 2973, 3053, 2897, 3060),
            (2959, 2983, 2969, 3053, 2971, 3057),
        ],
        ["0.018", "0.015", "0.017", "0.019"],
    ),
    "ECE": (
        [
            (3123, 3147, 2961, 3040, 3135, 3048),
            (3197, 3222, 2943, 3020, 3209, 3027),
            (3273, 3297, 2931, 3007, 3285, 3013),
            (3346, 3370, 2937, 3016, 3358, 3020),
        ],
        ["0.029", "0.049", "0.060", "0.055"],
    ),
    "Acc": (
        [
            (3513, 3537, 1972, 2051, 3525, 2057),
            (3587, 3611, 1969, 2047, 3599, 2053),
            (3662, 3686, 1972, 2048, 3674, 2056),
            (3737, 3761, 1969, 2047, 3749, 2053),
        ],
        ["0.950", "0.954", "0.951", "0.955"],
    ),
}
font_num = ImageFont.truetype(FONT, 46)
for grp, (boxes, texts) in vals.items():
    for (x0, x1, y0, y1, cx, bartop), txt in zip(boxes, texts):
        ey1 = min(y1 + 6, bartop - 3)
        ey0 = y0 - 8
        a2 = np.asarray(im2).astype(np.int64)
        reg = a2[ey0:ey1, x0 - 6 : x1 + 6]
        reg[:, :] = [255, 255, 255]
        a2[ey0:ey1, x0 - 6 : x1 + 6] = reg
        # restore gridline y=2003 if crossed
        if ey0 <= 2003 <= ey1:
            a2[2001:2006, x0 - 6 : x1 + 6] = [248, 248, 248]
        im2 = Image.fromarray(a2.astype(np.uint8))
        d = ImageDraw.Draw(im2)
        # tilted 45-degree text, lower-left end anchored just above the bar top
        tmp = Image.new("RGBA", (300, 70), (0, 0, 0, 0))
        dt = ImageDraw.Draw(tmp)
        dt.text((150, 35), txt, font=font_num, fill=DARK, anchor="mm")
        rot = tmp.rotate(45, expand=True)
        bx0, by0, bx1, by1 = rot.getbbox()
        im2.paste(
            rot,
            (int(cx - 14 - 8 - bx0), int(bartop - 12 + 8 - by1)),
            rot,
        )
im2.save("revision/manuscript/figures/Fig.3.png")
print("Fig.3 patched")

# ============================ Fig.4 ============================
im = Image.open("revision/manuscript/figures/Fig.4.png").convert("RGB")
a = np.asarray(im).astype(np.int64)


# sample exact legend colors from the sample segments
def class_color(y0, y1, x0, x1, pick):
    reg = a[y0:y1, x0:x1].reshape(-1, 3)
    sp = spread(reg)
    cand = reg[sp > 30]
    if pick == "minR":
        return tuple(cand[cand[:, 0].argmin()])
    return tuple(cand[len(cand) // 2])


C_BLUE = class_color(1136, 1150, 1416, 1500, "minR")
C_ORANGE = class_color(1196, 1210, 1416, 1500, "med")
C_GREEN = class_color(1255, 1270, 1416, 1500, "med")
C_RED = class_color(1315, 1330, 1416, 1500, "med")
C_PERFECT = (166, 166, 166)
print("legend colors", C_BLUE, C_ORANGE, C_GREEN, C_RED)

# erase legend text + gray elements in the legend area
y0, y1, x0, x1 = 1050, 1350, 1398, 1860
reg = a[y0:y1, x0:x1]
sp = spread(reg)
lum = reg.mean(axis=2)
grayish = (sp < 45) & (lum < 235)
# keep vertically-continuous colored curves: colored px untouched already
reg[grayish] = [255, 255, 255]
a[y0:y1, x0:x1] = reg

# second pass: remove light antialiased text halos in the old legend area
reg = a[y0:y1, x0:x1]
sp = spread(reg)
lum = reg.mean(axis=2)
reg[(sp < 45) & (lum < 252)] = [255, 255, 255]
a[y0:y1, x0:x1] = reg

# reconstruct the old legend area: classify pixels, close curve masks per color,
# repaint curves solid, whiten everything else
RY0, RY1, RX0, RX1 = 1040, 1360, 1390, 1870
reg = a[RY0:RY1, RX0:RX1]
sp = spread(reg)
lum = reg.mean(axis=2)
curve_colors = [np.array(c) for c in (C_BLUE, C_ORANGE, C_GREEN, C_RED)]
dist = np.stack([np.abs(reg - c).sum(axis=2) for c in curve_colors], axis=0)
nearest = dist.argmin(axis=0)
ndist = dist.min(axis=0)
chromatic = (sp > 22) & (ndist < 300)
curveish = chromatic & (lum > 28)
whiteish = (lum >= 235) & (sp <= 22)
textish = ~curveish & ~whiteish
for ci, c in enumerate(curve_colors):
    m = curveish & (nearest == ci)
    if m.sum() < 30:
        continue
    closed = ndi.binary_closing(m, structure=np.ones((11, 11)))
    closed = ndi.binary_fill_holes(closed)
    # limit growth: only accept bridged px adjacent (within 3px) to original mask
    grow = ndi.binary_dilation(m, iterations=3) & closed & ~m
    final = m | grow
    reg[final] = c
reg[textish | (whiteish & (sp > 8) & False)] = [255, 255, 255]
a[RY0:RY1, RX0:RX1] = reg

# erase the colored sample segments (isolated short runs at 4 bands),
# preserving curve pixels that continue vertically beyond the band
BANDS = [
    (1132, 1152, C_BLUE),
    (1192, 1212, C_ORANGE),
    (1251, 1272, C_GREEN),
    (1311, 1332, C_RED),
]
for by0, by1, c in BANDS:
    c = np.array(c)
    reg = a[by0:by1, 1405:1505]
    dist = np.abs(reg - c).sum(axis=2)
    m = dist < 150
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        ay, ax = by0 + y, 1405 + x
        above = (np.abs(a[ay - 28 : ay - 12, ax] - c).sum(axis=1) < 150).any()
        below = (np.abs(a[ay + 12 : ay + 28, ax] - c).sum(axis=1) < 150).any()
        if not (above and below):
            a[ay, ax] = [255, 255, 255]

# verify the destination area is empty, then draw the legend there
dest = a[300:660, 300:820]
dsp = spread(dest)
dlum = dest.mean(axis=2)
assert ((dsp > 25) | (dlum < 235)).sum() < 400, "destination not empty: %d px" % (
    ((dsp > 25) | (dlum < 235)).sum(),
)

im2 = Image.fromarray(a.astype(np.uint8))
d = ImageDraw.Draw(im2)
font_leg = ImageFont.truetype(FONT, 42)
entries = [
    ("Perfect", C_PERFECT, "dash"),
    ("Baseline", C_BLUE, "marker"),
    ("+Wavelet", C_ORANGE, "marker"),
    ("+EDL", C_GREEN, "marker"),
    ("+Wavelet+EDL", C_RED, "marker"),
]
sy = 363
for name, col, style in entries:
    if style == "dash":
        for k in range(0, 80, 14):
            d.line((336 + k, sy, 336 + k + 8, sy), fill=col, width=5)
    else:
        d.line((336, sy, 416, sy), fill=col, width=5)
        d.ellipse((368, sy - 9, 386, sy + 9), fill=col)
    d.text((446, sy), name, font=font_leg, fill=(0, 0, 0), anchor="lm")
    sy += 60
im2.save("revision/manuscript/figures/Fig.4.png")
print("Fig.4 patched")
