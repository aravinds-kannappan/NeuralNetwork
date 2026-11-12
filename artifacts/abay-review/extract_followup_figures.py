"""Crop original paper figures without redrawing; requires pymupdf."""
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ('bsf-vision.pdf', 11, (104, 32, 510, 272), 'bsf-curve-manifold-figure11.png'),
    ('instagent.pdf', 0, (54, 185, 558, 480), 'instagent-architecture-figure1.png'),
]
for filename, page, rect, output in SOURCES:
    with pymupdf.open(ROOT / filename) as doc:
        doc[page].get_pixmap(matrix=pymupdf.Matrix(3,3), clip=pymupdf.Rect(*rect)).save(ROOT/output)
    print(output)
