"""Gera site/img a partir da arte do convite (1599x2000).

Uso: python tools/vectorize.py convite.jpg site/img
Requer: pillow numpy scikit-image

- casa.png: recorte da casa com o papel transparente.
- nome.svg: "Gabriela e Júnior" vetorizado, com animação de caneta (contorno e depois tinta).
"""
import sys
import numpy as np
from PIL import Image
from skimage.filters import gaussian
from skimage.measure import find_contours

SRC, OUT = sys.argv[1], sys.argv[2]
rgb = np.asarray(Image.open(SRC).convert("RGB")).astype(float)
lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114

HOUSE = (275, 615, 1265, 1295)  # x0, y0, x1, y1
NAME = (260, 1320, 1286, 1620)
INK = "#2b2523"
THIN = 0.68  # nível do contorno: maior = traço mais fino (0.5 = original)


def house():
    x0, y0, x1, y1 = HOUSE
    alpha = np.clip((235 - lum[y0:y1, x0:x1]) / 175, 0, 1)
    alpha[: 945 - y0, : 400 - x0] = 0  # ramo do canto superior esquerdo invade o recorte
    out = np.zeros(alpha.shape + (4,), np.uint8)
    out[..., :3] = (63, 58, 56)
    out[..., 3] = (alpha * 255).astype(np.uint8)
    Image.fromarray(out).save(f"{OUT}/casa.png", optimize=True)


def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    a, b = np.array(pts[0]), np.array(pts[-1])
    d = b - a
    n = np.hypot(*d) or 1.0
    dist = [abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / n for p in pts[1:-1]]
    i = int(np.argmax(dist)) + 1
    if dist[i - 1] > eps:
        return rdp(pts[: i + 1], eps)[:-1] + rdp(pts[i:], eps)
    return [pts[0], pts[-1]]


def name(start=2.0, spread=1.6, dur=2.4):
    x0, y0, x1, y1 = NAME
    field = np.pad(gaussian((lum[y0:y1, x0:x1] < 222).astype(float), 0.7), 2)
    contours = []
    for c in find_contours(field, THIN):
        if len(c) < 40:
            continue
        if c[:, 1].min() > 1000 and c[:, 0].max() < 40:
            continue  # pedaço do ramo vizinho que cai no recorte
        c = [tuple(p) for p in c]
        m = len(c) // 2  # contorno fechado: simplifica em duas metades
        c = [(y - 2, x - 2) for y, x in rdp(c[: m + 1], 0.5) + rdp(c[m:], 0.5)[1:]]
        contours.append((min(x for _, x in c), "M" + " L".join("%.1f %.1f" % (x, y) for y, x in c) + "Z"))
    contours.sort()
    lo, hi = contours[0][0], contours[-1][0]
    strokes = "\n".join(
        f'<path class="s" d="{d}" pathLength="1" style="animation-delay:{start + (k - lo) / (hi - lo) * spread:.2f}s"/>'
        for k, d in contours
    )
    end = start + spread + dur
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {x1 - x0} {y1 - y0}">
<style>
.s{{fill:none;stroke:{INK};stroke-width:.7;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:1;animation:d {dur}s ease-in-out forwards,o .8s ease {end:.2f}s forwards}}
.f{{opacity:0;animation:i .8s ease {end - .3:.2f}s forwards}}
@keyframes d{{to{{stroke-dashoffset:0}}}}
@keyframes o{{to{{opacity:0}}}}
@keyframes i{{to{{opacity:1}}}}
@media (prefers-reduced-motion:reduce){{.s{{display:none}}.f{{animation:none;opacity:1}}}}
</style>
{strokes}
<path class="f" fill="{INK}" fill-rule="evenodd" d="{"".join(d for _, d in contours)}"/>
</svg>"""
    open(f"{OUT}/nome.svg", "w").write(svg)


house()
name()
