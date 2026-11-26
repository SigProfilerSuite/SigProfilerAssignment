"""Unit tests for the SBS1536 decomposition page (dummy images, no genome needed)."""
import io

import pandas as pd
import pytest
from PIL import Image
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from SigProfilerAssignment.DecompositionPlots import PlotDecomposition_SBS1536 as spd_1536


def _image():
    buff = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buff, format="PNG")
    buff.seek(0)
    return ImageReader(buff)


def _canvas():
    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=landscape(letter))
    c.setFont("Arial-Bold", 7.19)
    return buff, c


STATISTICS = pd.DataFrame({"Cosine Similarity": [0.99], "Correlation Coefficient": [0.98], "L1 Norm %": [5.0],
                           "L2 Norm %": [4.0], "KL Divergence": [0.01]})


@pytest.mark.parametrize("n_bases", [1, 2, 3, 4, 5, 6])
def test_page_with_reconstruction(n_bases):
    bases = [["SBS%d" % (i + 1), 100.0 / n_bases] for i in range(n_bases)]
    buff, c = _canvas()
    spd_1536.gen_plot(
        "SBS1536A", bases, "unused_output_path", "project", c, True, STATISTICS, "v3.6", "custom text",
        {"SBS1536A": _image()}, {b[0]: _image() for b in bases}, {"SBS1536A": _image()},
    )
    c.save()
    assert buff.getvalue().startswith(b"%PDF")


@pytest.mark.parametrize("n_bases", [1, 2, 3])
def test_page_without_reconstruction(n_bases):
    # the de novo signature is drawn alone when reconstruction=False
    bases = [["SBS%d" % (i + 1), 100.0 / n_bases] for i in range(n_bases)]
    buff, c = _canvas()
    spd_1536.gen_plot(
        "SBS1536A", bases, "unused_output_path", "project", c, False, None, None, None,
        {"SBS1536A": _image()}, {b[0]: _image() for b in bases}, {"SBS1536A": _image()},
    )
    c.save()
    assert buff.getvalue().startswith(b"%PDF")
