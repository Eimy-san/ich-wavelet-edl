"""Patch Fig.1.png: fix risk-coverage schematic (coverage semantics), backbone
stage sizes (28x28x512 insertion after second residual stage), and wavelet
module highlight position. Run on an ORIGINAL Fig.1.png (idempotent-safe)."""

import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

SRC = sys.argv[1] if len(sys.argv) > 1 else "revision/manuscript/figures/Fig.1.png"
im = Image.open(SRC).convert("RGB")
a = np.asarray(im).astype(np.float64)

CREAM = np.array([255, 241, 225], dtype=np.float64)
ORANGE = (199, 131, 32)


def orange_mask(reg):
    R, G, B = reg[:, :, 0], reg[:, :, 1], reg[:, :, 2]
    return (R > 120) & (R - G > 15) & (G - B > 15) & (B < 200)


def dilate(m, it=2):
    m = m.copy()
    for _ in range(it):
        m2 = m.copy()
        m2[1:, :] |= m[:-1, :]
        m2[:-1, :] |= m[1:, :]
        m2[:, 1:] |= m[:, :-1]
        m2[:, :-1] |= m[:, 1:]
        m = m2
    return m


# ---- 1) erase old wavelet tag (three orange text lines) ----
x0, y0, x1, y1 = 1400, 250, 2030, 480
a[y0:y1, x0:x1] = [254, 254, 254]

# ---- 2) clean old dashed rect area: remove cream wash + orange border ----
x0, y0, x1, y1 = 1250, 470, 2160, 1120
reg = a[y0:y1, x0:x1]
R, G, B = reg[:, :, 0], reg[:, :, 1], reg[:, :, 2]
lum = reg.mean(axis=2)
warm = np.clip(((R - B) - 6) / 18.0, 0, 1) * np.clip((lum - 200) / 40.0, 0, 1)
w = np.clip(warm, 0, 1)
reg = reg * (1 - w[:, :, None]) + np.array([254, 254, 254]) * w[:, :, None]
m = dilate(orange_mask(reg), 2)
if m.any():
    from scipy.ndimage import median_filter

    keep = ~m
    filled = reg.copy()
    for ch in range(3):
        med = median_filter(
            np.where(keep, reg[:, :, ch], 0.0), size=(9, 9), mode="nearest"
        )
        wsum = median_filter(keep.astype(float), size=(9, 9), mode="nearest")
        est = med / np.maximum(wsum, 1e-6)
        filled[:, :, ch] = np.where(m, est, reg[:, :, ch])
    reg = filled
a[y0:y1, x0:x1] = reg

# ---- 3) erase old diagonal connectors (stop above detail panel border) ----
x0, y0, x1, y1 = 400, 975, 2110, 1234
reg = a[y0:y1, x0:x1]
m = dilate(orange_mask(reg), 2)
if m.any():
    from scipy.ndimage import median_filter

    keep = ~m
    filled = reg.copy()
    for ch in range(3):
        med = median_filter(
            np.where(keep, reg[:, :, ch], 0.0), size=(9, 9), mode="nearest"
        )
        wsum = median_filter(keep.astype(float), size=(9, 9), mode="nearest")
        est = med / np.maximum(wsum, 1e-6)
        filled[:, :, ch] = np.where(m, est, reg[:, :, ch])
    reg = filled
a[y0:y1, x0:x1] = reg

# ---- 4) new cream tint around Stage 2 ----
rx0, ry0, rx1, ry1 = 855, 500, 1255, 1065
reg = a[ry0:ry1, rx0:rx1]
R, G, B = reg[:, :, 0], reg[:, :, 1], reg[:, :, 2]
light = np.clip(((np.minimum(np.minimum(R, G), B) - 228) / 24.0), 0, 1)[:, :, None]
reg = reg * (1 - light) + CREAM * light
a[ry0:ry1, rx0:rx1] = reg


def draw_dashed_rect(
    arr, x0, y0, x1, y1, r=22, dash=16, gap=11, width=5, color=ORANGE, skip_dark=True
):
    h, w = arr.shape[:2]
    pts = []

    def seg(p, q):
        n = int(math.hypot(q[0] - p[0], q[1] - p[1]))
        for i in range(n + 1):
            t = i / max(n, 1)
            pts.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))

    def arc(c, a0, a1):
        for k in range(0, 91):
            t = math.radians(a0 + (a1 - a0) * k / 90)
            pts.append((c[0] + r * math.cos(t), c[1] + r * math.sin(t)))

    seg((x0 + r, y0), (x1 - r, y0))
    arc((x1 - r, y0 + r), -90, 0)
    seg((x1, y0 + r), (x1, y1 - r))
    arc((x1 - r, y1 - r), 0, 90)
    seg((x1 - r, y1), (x0 + r, y1))
    arc((x0 + r, y1 - r), 90, 180)
    seg((x0, y1 - r), (x0, y0 + r))
    arc((x0 + r, y0 + r), 180, 270)

    acc, state = 0, True
    for px, py in pts:
        if state:
            xi, yi = int(round(px)), int(round(py))
            if 0 <= xi < w and 0 <= yi < h:
                if skip_dark and arr[yi, xi].mean() < 140:
                    acc += 1
                    continue
                hw = width // 2
                arr[max(0, yi - hw) : yi + hw + 1, max(0, xi - hw) : xi + hw + 1] = (
                    color
                )
        acc += 1
        if state and acc >= dash:
            state, acc = False, 0
        elif not state and acc >= gap:
            state, acc = True, 0


def draw_dashed_line(arr, p, q, dash=16, gap=11, width=5, color=ORANGE):
    h, w = arr.shape[:2]
    n = int(math.hypot(q[0] - p[0], q[1] - p[1]))
    acc, state = 0, True
    for i in range(n + 1):
        t = i / max(n, 1)
        if state:
            xi = int(round(p[0] + (q[0] - p[0]) * t))
            yi = int(round(p[1] + (q[1] - p[1]) * t))
            if 0 <= xi < w and 0 <= yi < h:
                hw = width // 2
                arr[max(0, yi - hw) : yi + hw + 1, max(0, xi - hw) : xi + hw + 1] = (
                    color
                )
        acc += 1
        if state and acc >= dash:
            state, acc = False, 0
        elif not state and acc >= gap:
            state, acc = True, 0


draw_dashed_rect(a, rx0, ry0, rx1, ry1)
draw_dashed_line(a, (940, 1063), (505, 1236))
draw_dashed_line(a, (1180, 1063), (2063, 1236))

# ---- 5) redraw tag at new position ----
a8 = Image.fromarray(a.astype(np.uint8))
d = ImageDraw.Draw(a8)
font = ImageFont.truetype(matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf", 50)
for i, ln in enumerate(["Wavelet", "Frequency-Domain", "Attention"]):
    d.text((1270, 278 + i * 61), ln, font=font, fill=ORANGE)

# ---- 6) stage size labels ----
font_lbl = font


def rowmedian_fill(px0: int, py0: int, px1: int, py1: int, clean_cols: tuple) -> None:
    """Fill label area with per-row median color from clean columns of same box."""
    reg = np.asarray(a8.crop((px0, py0, px1, py1))).astype(np.float64)
    colpix = np.asarray(a8.crop((clean_cols[0], py0, clean_cols[1], py1))).astype(
        np.float64
    )
    med = np.median(colpix, axis=1)[:, None, :]
    out = np.repeat(med, px1 - px0, axis=1)
    a8.paste(Image.fromarray(out.astype(np.uint8)), (px0, py0))


def set_label(box, clean_cols, new_text, center):
    px0, py0, px1, py1 = box
    rowmedian_fill(px0, py0, px1, py1, clean_cols)
    d.text(
        (center[0], center[1]),
        new_text,
        font=font_lbl,
        fill=(255, 255, 255),
        anchor="mm",
    )


set_label((955, 376, 1155, 446), (900, 950), "56²×64", (1055, 411))
set_label((970, 788, 1145, 858), (900, 960), "28²×512", (1055, 823))
set_label((1400, 788, 1575, 858), (1330, 1395), "14²×1024", (1485, 823))
set_label((1825, 788, 2000, 858), (1760, 1820), "7²×2048", (1912, 823))

# ---- 7) risk-coverage schematic: white-out + matplotlib panel ----
sx0, sy0, sx1, sy1 = 2360, 1435, 3720, 2080
a8.paste((254, 254, 254), (sx0, sy0, sx1, sy1))

PW, PH = sx1 - sx0, sy1 - sy0
fig = plt.figure(figsize=(PW / 100, PH / 100), dpi=100)
ax = fig.add_axes([0.115, 0.175, 0.845, 0.70])
fp = FontProperties(family="DejaVu Sans")
xs_ = np.linspace(0, 1, 200)
acc = 0.985 - 0.058 * xs_**1.7
ax.axvspan(0, 0.05, color="#E8A08A", alpha=0.30, lw=0)
ax.axvspan(0.05, 1.0, color="#9FCAC5", alpha=0.25, lw=0)
ax.plot(xs_, acc, color="#126D79", lw=5, solid_capstyle="round")
ax.axvline(0.05, color="#B0532F", lw=2.5, ls=(0, (6, 4)), alpha=0.9)
ax.text(
    0.015,
    1.045,
    "Abstain / Radiologist",
    ha="left",
    va="bottom",
    transform=ax.transAxes,
    fontsize=11.5,
    color="#8A3A20",
    fontproperties=fp,
)
ax.text(
    0.035,
    0.34,
    "5%",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=11,
    color="#8A3A20",
    fontproperties=fp,
)
ax.text(
    0.545,
    0.78,
    "Model prediction",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=17,
    color="#0E4E58",
    fontproperties=fp,
)
ax.text(
    0.545,
    0.62,
    "(95% of examinations)",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=13.5,
    color="#0E4E58",
    fontproperties=fp,
)
ax.set_xlim(0, 1)
ax.set_ylim(0.88, 1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Coverage", fontsize=16, fontproperties=fp, labelpad=6)
ax.set_ylabel("Accuracy", fontsize=16, fontproperties=fp, labelpad=6)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color("#7A8A93")
    ax.spines[s].set_linewidth(1.6)
fig.text(
    0.5,
    0.045,
    "R I S K – C O V E R A G E   C O N C E P T",
    ha="center",
    fontsize=13.5,
    color="#5A6A72",
    fontproperties=fp,
)
fig.savefig("/tmp/rc_panel.png", dpi=100)
plt.close(fig)

panel = Image.open("/tmp/rc_panel.png").convert("RGB").resize((PW, PH), Image.LANCZOS)
a8.paste(panel, (sx0, sy0))
a8.save(SRC)
print("patched", SRC)
