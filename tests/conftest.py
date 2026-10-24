import os
from pathlib import Path

import ezdxf
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "tests" / "fixtures" / "dxf" / "simple_site.dxf"
PROFILE = ROOT / "packs" / "profiles" / "ru-landscape-starter.json"


@pytest.fixture
def site():
    return str(SITE)


@pytest.fixture
def profile_path():
    return str(PROFILE)


def make_dxf(path, units=6, vp_height=17.0, vp_size=(240, 170), center=(11, 7.5), entities=None):
    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = units
    msp = doc.modelspace()
    for layer, pts in (entities or [("ГАЗОН", [(0, 0), (10, 0), (10, 5), (0, 5)])]):
        if layer not in doc.layers:
            doc.layers.add(layer)
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})
    lay = doc.layouts.get("Layout1")
    lay.add_viewport(center=(150, 150), size=vp_size, view_center_point=center, view_height=vp_height)
    doc.saveas(path)
    return str(path)
