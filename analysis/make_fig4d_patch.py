import sys
import matplotlib
from PIL import Image, ImageDraw, ImageFont

FONT = matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"
FONT_B = matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans-Bold.ttf"

SRC, DST = sys.argv[1], sys.argv[2]
im = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(im)
# axes (d): (2237,1808)-(3860,3088); legend frameless abs region (3395,2815)-(3845,3045)
# entry rows (abs): samples x3412-3491 (80px), text x3521, rows y-centers ~2843,2903,2962,3022
d.rectangle([3395, 2810, 3850, 3050], fill=(255, 255, 255))
# restore horizontal gridlines crossing the erased band: y ~2823 and ~3076
for gy in (2823, 3076):
    if 2810 <= gy <= 3050:
        col = im.getpixel((2500, gy))
        d.line([(3395, gy), (3850, gy)], fill=col, width=1)
# restore dashed vertical line x ~3779 crossing the band
d.rectangle([3775, 2810, 3784, 3050], fill=(255, 255, 255))
strip = im.crop((3775, 2500, 3784, 2760))
im.paste(strip, (3775, 2810))
# erase old '95% coverage' annotation (3645,2977)-(3913,3025)
d.rectangle([3645, 2977, 3913, 3025], fill=(255, 255, 255))
im.paste(strip, (3775, 2977))

# redraw legend at lower-left empty area (same y rows, x shifted -1122)
DX = -1122
COL = {
    "Baseline": (62, 115, 166),
    "+Wavelet": (199, 122, 78),
    "+EDL": (112, 147, 95),
    "+Wavelet+EDL": (164, 61, 84),
}
rows = {"Baseline": 2843, "+Wavelet": 2903, "+EDL": 2962, "+Wavelet+EDL": 3022}
F = ImageFont.truetype(FONT, 42)
for name, y in rows.items():
    c = COL[name]
    d.line([(3412 + DX, y), (3492 + DX, y)], fill=c, width=6)
    d.text((3521 + DX, y), name, font=F, fill=(26, 26, 26), anchor="lm")

# redraw '95% coverage' enlarged, solid, right-aligned ending left of the dashed line
FB = ImageFont.truetype(FONT_B, 55)
d.text((3756, 2997), "95% coverage", font=FB, fill=(51, 51, 51), anchor="rm")

im.save(DST)
print("Fig.4 panel (d) legend/annotation fixed")
