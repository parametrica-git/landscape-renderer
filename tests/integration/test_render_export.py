import numpy as np
import ezdxf
import pytest
from PIL import Image

from landscape_renderer.cad.reader import sha256
from landscape_renderer.export import save_dxf_bundle, save_pdf, save_png
from landscape_renderer.pipeline import Session
from landscape_renderer.project import Project


@pytest.fixture(scope="module")
def session():
    from conftest import SITE, PROFILE
    p = Project(source=str(SITE), profile=str(PROFILE))
    p.render.dpi = 100
    return Session.open(p)


@pytest.fixture(scope="module")
def result(session):
    return session.render()


def test_size_matches_paper(result):
    w_mm, h_mm = result.sheet_mm
    assert abs(result.image.size[0] - w_mm / 25.4 * result.dpi) <= 2
    assert abs(result.image.size[1] - h_mm / 25.4 * result.dpi) <= 2


def test_deterministic(session, result):
    again = session.render()
    assert np.array_equal(np.asarray(result.image), np.asarray(again.image))


def test_quality_metrics(result):
    """Стартовые метрики по маскам (SPEC 7.2): чистота поля бумаги и сдержанность цвета."""
    a = np.asarray(result.image.convert("RGB")).astype(np.float32) / 255
    m = int(result.origin_mm[0] / 25.4 * result.dpi) - 3
    paper = np.concatenate([a[:m].reshape(-1, 3), a[-m:].reshape(-1, 3)])
    import cv2
    # «чистота бумаги»: поле совпадает с выбранной бумагой (тёплая, не белая) — ΔE76 ≤ 4, L* ≥ 94
    ref = np.array([[[0.988, 0.978, 0.952]]], np.float32)
    to_lab = lambda x: cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB).reshape(-1, 3)  # noqa: E731
    lab, ref_lab = to_lab(paper[None]), to_lab(ref)
    dE = np.linalg.norm(lab - ref_lab, axis=1)
    assert np.mean((dE <= 4) & (lab[:, 0] >= 94)) >= 0.85
    hsv = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    S = hsv[:, 1] / 255
    colored = S[S > 0.08]
    assert np.median(colored) <= 0.40


def test_exports(tmp_path, session, result):
    before = sha256(session.cad.path)
    png = save_png(result, tmp_path / "a.png")
    assert Image.open(png).info.get("dpi", (0,))[0] == pytest.approx(result.dpi, abs=1)
    pdf = save_pdf(result, tmp_path / "a.pdf")
    assert open(pdf, "rb").read(4) == b"%PDF"
    meta = save_dxf_bundle(result, session.scene, session.cad.path, tmp_path / "b", name="p")
    meta2 = save_dxf_bundle(result, session.scene, meta["dxf"], tmp_path / "c", name="p")
    for m in (meta, meta2):
        doc = ezdxf.readfile(m["dxf"])
        imgs = list(doc.layouts.get(session.viewport.layout).query("IMAGE"))
        assert len(imgs) == 1 and imgs[0].has_xdata("LANDSCAPE_RENDERER")   # без дублей
    vp = session.viewport
    x, y = meta["insert"]
    assert x == pytest.approx(vp.paper_center_mm[0] - vp.paper_size_mm[0] / 2 - result.origin_mm[0])
    assert y == pytest.approx(vp.paper_center_mm[1] - vp.paper_size_mm[1] / 2 - result.origin_mm[1])
    assert sha256(session.cad.path) == before      # исходник не тронут


def test_cli_report(site, capsys):
    from landscape_renderer.cli import main
    assert main(["report", site]) == 0
    assert "подтверждено" in capsys.readouterr().out
