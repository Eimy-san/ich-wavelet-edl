"""Fig.1 architecture correction:
1. Wavelet branch returns to the main path AFTER STAGE 2 (not after Stage 4).
2. The four Haar subbands are produced in PARALLEL by the DWT and processed
   independently (no LL->LH / HL->HH serial chains).
3. Adds the missing skip connection (original input) into the residual '+' node
   and a DWT label at the decomposition point.
Runs on the CURRENT manuscript copy; idempotent (erases + redraws fixed regions).
"""

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SLATE = (61, 69, 88)
FONT = matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"
P = "revision/manuscript/figures/Fig.1.png"
im = Image.open(P).convert("RGB")
a = np.asarray(im).astype(np.int64)


def erase_dark_band(p0, p1, hw=9, thr=130):
    """Erase dark pixels in a band around the segment p0->p1."""
    x0, y0 = p0
    x1, y1 = p1
    L = max(1, int(np.hypot(x1 - x0, y1 - y0)))
    for t in range(L + 1):
        cx = x0 + (x1 - x0) * t / L
        cy = y0 + (y1 - y0) * t / L
        xa, xb = int(cx - hw), int(cx + hw) + 1
        ya, yb = int(cy - hw), int(cy + hw) + 1
        reg = a[ya:yb, xa:xb]
        m = (reg < thr).all(axis=2)
        reg[m] = [255, 255, 255]
        a[ya:yb, xa:xb] = reg


def erase_rect(x0, y0, x1, y1):
    a[y0:y1, x0:x1] = [255, 255, 255]


# ---- erase old leader diagonals (wrong association with Stage 3/4) ----
erase_dark_band((502, 1239), (1319, 1015))
erase_dark_band((1648, 1016), (2066, 1235))
# ---- erase old return-line arrowhead below the Stage4->Head connector ----
erase_rect(2140, 758, 2186, 796)
# ---- erase serial chain arrows in the column gap ----
erase_rect(1074, 1405, 1154, 1452)
erase_rect(1074, 1730, 1154, 1777)
# ---- erase the input fork curves (replaced by a straight bus) ----
erase_rect(506, 1556, 665, 1635)
# ---- erase the old shared up/down feed arrows (double-headed look) ----
erase_rect(744, 1567, 766, 1590)
erase_rect(744, 1597, 766, 1617)
erase_rect(746, 1589, 762, 1598)
erase_rect(1318, 1567, 1340, 1590)
erase_rect(1318, 1597, 1340, 1617)
erase_rect(1320, 1589, 1336, 1598)

# ---- repair corners nicked by the leader erases (mirror intact corners) ----
def repair_corner(donor_box, target_xy, flip="v"):
    x0, y0, x1, y1 = donor_box
    blk = a[y0:y1, x0:x1]
    if flip == "v":
        blk = blk[::-1]
    else:
        blk = blk[:, ::-1]
    tx, ty = target_xy
    a[ty : ty + blk.shape[0], tx : tx + blk.shape[1]] = blk

# Stage 3 bottom corners (donors: intact top corners, flipped vertically)
repair_corner((1319, 545, 1357, 573), (1319, 999), "v")
repair_corner((1614, 545, 1652, 573), (1614, 999), "v")
# LL bottom-left and HL top-left corners (donors: opposite corners of same box)
repair_corner((633, 1290, 671, 1322), (633, 1537), "v")
repair_corner((633, 1869, 671, 1897), (633, 1620), "v")
# column-gap arrow zones: refill with the gap shading shifted from below/above
a[1405:1452, 1074:1154] = a[1460:1507, 1074:1154]
a[1730:1777, 1074:1154] = a[1782:1829, 1074:1154]

im2 = Image.fromarray(a.astype(np.uint8))
d = ImageDraw.Draw(im2)


def line(pts, w=4):
    d.line(pts, fill=SLATE, width=w, joint="curve")


def arrow(tip, direction, length=17, half=8):
    x, y = tip
    if direction == "up":
        d.polygon([(x, y), (x - half, y + length), (x + half, y + length)], fill=SLATE)
    elif direction == "down":
        d.polygon([(x, y), (x - half, y - length), (x + half, y - length)], fill=SLATE)
    elif direction == "left":
        d.polygon([(x, y), (x + length, y - half), (x + length, y + half)], fill=SLATE)
    elif direction == "right":
        d.polygon([(x, y), (x - length, y - half), (x - length, y + half)], fill=SLATE)


def dot(x, y, r=5):
    d.ellipse([x - r, y - r, x + r, y + r], fill=SLATE)


# ---- input bus through the between-rows corridor ----
line([(500, 1593), (1450, 1593)])
line([(736, 1593), (772, 1593)])
# four parallel feeds, up and down taps separated for unambiguous direction
dot(726, 1593)
line([(726, 1593), (726, 1575)])
arrow((726, 1568), "up")
dot(782, 1593)
line([(782, 1593), (782, 1612)])
arrow((782, 1618), "down")
dot(1306, 1593)
line([(1306, 1593), (1306, 1575)])
arrow((1306, 1568), "up")
dot(1354, 1593)
line([(1354, 1593), (1354, 1612)])
arrow((1354, 1618), "down")

# ---- outputs of all four subbands converge on the hub at x1650 ----
line([(854, 1290), (854, 1265), (1650, 1265)])  # LL out (top corridor)
line([(854, 1897), (854, 1918), (1650, 1918)])  # HL out (bottom corridor)
line([(1650, 1265), (1650, 1918)])  # hub (covers existing span)

# ---- skip connection: original input into the '+' node ----
line([(450, 1594), (450, 1938), (2068, 1938), (2068, 1650)])
arrow((2068, 1640), "up")
dot(450, 1594)

# ---- DWT label at the decomposition point ----
f = ImageFont.truetype(FONT, 26)
d.text((450, 1558), "DWT", font=f, fill=SLATE, anchor="mb")

# ---- return path: '+' node -> up -> left above Stage 3/4 -> down after Stage 2 ----
line([(2160, 1592), (2160, 472), (1270, 472), (1270, 742)])
arrow((1270, 752), "down")

im2.save(P)
print("Fig.1 architecture corrected")
