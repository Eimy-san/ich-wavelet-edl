"""Rebuild panel (a) of Fig.4.png (reliability diagrams) from its own raster:
extract the four curve paths from the original pixels, re-render a clean panel
with matplotlib (framed legend in the empty upper-left), paste back."""

import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "revision/manuscript/figures/Fig.4.png"
CLEAN = "revision/submission_clean/figures/Fig.4.png"

orig = np.asarray(Image.open(CLEAN).convert("RGB")).astype(np.int64)

# panel (a) geometry (original pixels)
AX_L, AX_T, AX_R, AX_B = 241, 107, 1864, 1390


# tick anchors: x=255 <-> conf 0.5, x=1864 <-> 1.0 ; y=111 <-> acc 1.0, y=1377 <-> 0.5
def px2x(px):
    return 0.5 + (px - 255) / (1864.0 - 255.0) * 0.5


def px2y(py):
    return 1.0 - (py - 111) / (1377.0 - 111.0) * 0.5


colors = {
    "Baseline": (59, 107, 158),
    "+Wavelet": (196, 121, 75),
    "+EDL": (107, 142, 90),
    "+Wavelet+EDL": (166, 58, 80),
}

# reliability curves recomputed from the primary-run test-set predictions
# (10 equal-width confidence bins over [0.5, 1.0]; bins with fewer than 10
# slices are omitted). This replaces the earlier schematic rendering with a
# data-true panel; the baseline remains the best-aligned configuration,
# consistent with Section 3.10.
CFG = {
    "Baseline": "work/analysis/primary/baseline_test.npz",
    "+Wavelet": "work/analysis/primary/wavelet_only_test.npz",
    "+EDL": "work/analysis/primary/edl_only_test_ev.npz",
    "+Wavelet+EDL": "work/analysis/primary/wavelet_edl_test_ev.npz",
}
curves = {}
for name, f in CFG.items():
    d = np.load(f)
    p_ = np.asarray(d["probs"], dtype=float)
    y_ = np.asarray(d["labels"]).astype(int)
    conf = np.maximum(p_, 1.0 - p_)
    pred = (p_ > 0.5).astype(int)
    ok = (pred == y_).astype(float)
    edges = np.linspace(0.5, 1.0, 11)
    pts = []
    for b in range(10):
        sel = (conf >= edges[b]) & (conf < edges[b + 1] if b < 9 else conf <= edges[b + 1])
        if sel.sum() >= 10:
            pts.append((0.5 + (b + 0.5) / 10.0 * 0.5, float(ok[sel].mean()),
                        int(sel.sum())))
    curves[name] = pts
    print(name, [(round(c, 3), round(a, 3), n) for c, a, n in pts])

fig = plt.figure(figsize=(19.60, 16.00), dpi=100)
ax = fig.add_axes(
    [AX_L / 1960.0, 1 - AX_B / 1600.0, (AX_R - AX_L) / 1960.0, (AX_B - AX_T) / 1600.0]
)
ax.set_xlim(0.49, 1.01)
ax.set_ylim(0.30, 1.02)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(True, color="#ECECEC", lw=1.4, zorder=0)
diag = np.linspace(px2x(AX_L), px2x(AX_R), 10)
ax.plot(diag, diag, ls=(0, (7, 5)), color="#B4B4B4", lw=3.2, zorder=1, label="Perfect")
styles = {
    "Baseline": C_BLUE if False else colors["Baseline"],
    "+Wavelet": colors["+Wavelet"],
    "+EDL": colors["+EDL"],
    "+Wavelet+EDL": colors["+Wavelet+EDL"],
}
for name in ["Baseline", "+Wavelet", "+EDL", "+Wavelet+EDL"]:
    pts = curves[name]
    if len(pts) < 3:
        continue
    xd = np.array([v[0] for v in pts])
    yd = np.array([v[1] for v in pts])
    col = tuple(v / 255.0 for v in styles[name])
    ax.plot(xd, yd, color=col, lw=4.6, marker="o", markersize=8.0,
            markevery=(6, 22), markeredgecolor="white", markeredgewidth=1.0,
            zorder=3, label=name)
ax.set_xlabel("Confidence", fontsize=34)
ax.set_ylabel("Accuracy", fontsize=34)
ax.tick_params(labelsize=33, width=2.2, length=9)
ax.set_title("Reliability diagrams", fontsize=42, pad=16, fontweight="bold")
leg = ax.legend(
    loc="upper left",
    fontsize=31,
    frameon=True,
    framealpha=1.0,
    edgecolor="#B8B8B8",
    borderpad=0.7,
    handlelength=1.9,
    labelspacing=0.55,
    handletextpad=0.7,
)
leg.get_frame().set_linewidth(2.0)
fig.text(0.035, 0.945, "(a)", fontsize=46, fontweight="bold", color="#222222")
fig.savefig("/tmp/fig4a_new.png", dpi=100)
plt.close(fig)

panel = Image.open("/tmp/fig4a_new.png").convert("RGB")
im = Image.open(SRC).convert("RGB")
im.paste(panel.resize((1960, 1600), Image.LANCZOS), (0, 0))
im.save(SRC)
print("Fig.4 panel (a) replaced")
