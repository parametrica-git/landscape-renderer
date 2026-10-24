from landscape_renderer.cad import read_dxf, pick_viewport
from conftest import make_dxf


def test_fixture_viewport(site):
    d = read_dxf(site)
    vp = pick_viewport(d)
    assert vp.layout == "Генплан 1-100"
    assert vp.scale_den == 100
    assert vp.window == (-1.0, -1.0, 23.0, 16.0)
    assert d.units_to_m == 1.0 and not d.warnings


def test_scale_in_millimetre_drawing(tmp_path):
    # чертёж в мм: вид высотой 34 000 мм на экране 170 мм = 1:200
    p = make_dxf(tmp_path / "mm.dxf", units=4, vp_height=34000, center=(11000, 7500),
                 entities=[("ГАЗОН", [(0, 0), (10000, 0), (10000, 5000), (0, 5000)])])
    vp = pick_viewport(read_dxf(p))
    assert vp.scale_den == 200


def test_unknown_units_warns(tmp_path):
    p = make_dxf(tmp_path / "nounits.dxf", units=0)
    d = read_dxf(p)
    assert d.units_to_m is None
    assert any("Единицы" in w for w in d.warnings)


def test_insert_attributes_cyrillic(site):
    d = read_dxf(site)
    trees = [e for e in d.entities if e.block == "ДЕРЕВО"]
    assert trees and all("ВИД" in e.attribs and "D" in e.attribs for e in trees)
