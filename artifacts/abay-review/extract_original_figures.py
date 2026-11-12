"""Crop original figures, without redrawing or changing their measurements.

Source: Abay Bektursun, Borrowed Geometry, arXiv:2605.00333v2, CC BY 4.0.
Download: https://arxiv.org/pdf/2605.00333v2
Run: uv run --isolated --no-project --with pymupdf python extract_original_figures.py
"""
from pathlib import Path
import pymupdf

HERE = Path(__file__).resolve().parent
doc = pymupdf.open(HERE / "borrowed-geometry-v2.pdf")
for page_index, box, name in [
    (4, (96, 68, 510, 325), "borrowed-geometry-figure2.png"),
    (8, (96, 72, 510, 350), "borrowed-geometry-figure4.png"),
]:
    doc[page_index].get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(box)).save(HERE / name)
    print(name)
