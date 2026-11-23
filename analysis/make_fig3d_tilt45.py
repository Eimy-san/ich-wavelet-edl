import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"
DARK = (51, 51, 51)
GRID_YS = (2003, 2217, 2433, 2648, 2863)
P = "revision/manuscript/figures/Fig.3.png"
im = Image.open(P).convert("RGB")
groups = [
    ([2357, 2433, 2506, 2581], 2076, ["0.933", "0.938", "0.936", "0.933"]),
    ([2747, 2822, 2897, 2971], 3061, ["0.018", "0.015", "0.017", "0.019"]),
    ([3135, 3209, 3285, 3358], 3048, ["0.029", "0.049", "0.060", "0.055"]),
    ([3525, 3599, 3674, 3749], 2057, ["0.950", "0.954", "0.951", "0.955"]),
]
a = np.asarray(im).astype(np.int64)
for cxs, bmax, _ in groups:
    ex0, ex1 = cxs[0] - 40, cxs[3] + 40
    ey0, ey1 = bmax - 240, bmax - 6
    a[ey0:ey1, ex0:ex1] = [255, 255, 255]
    for gy in GRID_YS:
        if ey0 <= gy <= ey1:
            a[gy - 2 : gy + 3, ex0:ex1] = [248, 248, 248]
im2 = Image.fromarray(a.astype(np.uint8))
d = ImageDraw.Draw(im2)
font_num = ImageFont.truetype(FONT, 46)
TOPS = {
    0: [2076, 2071, 2073, 2076],
    1: [3059, 3061, 3060, 3057],
    2: [3048, 3027, 3013, 3020],
    3: [2057, 2053, 2056, 2053],
}
for gi, (cxs, _, texts) in enumerate(groups):
    for cx, bartop, txt in zip(cxs, TOPS[gi], texts):
        tmp = Image.new("RGBA", (300, 70), (0, 0, 0, 0))
        dt = ImageDraw.Draw(tmp)
        dt.text((150, 35), txt, font=font_num, fill=DARK, anchor="mm")
        rot = tmp.rotate(45, expand=True)
        bx0, by0, bx1, by1 = rot.getbbox()
        im2.paste(rot, (int(cx - 22 - bx0), int(bartop - 4 - by1)), rot)
im2.save(P)
print("Fig.3 panel (d) band-cleared and numbers tilted to 45 degrees")
