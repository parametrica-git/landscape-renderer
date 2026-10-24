import sys, ezdxf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy

def view(path, out, layout="Model", dark=True):
    doc = ezdxf.readfile(path)
    lay = doc.modelspace() if layout == "Model" else doc.layouts.get(layout)
    fig = plt.figure(figsize=(14, 10), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    cfg = Configuration(background_policy=BackgroundPolicy.CUSTOM if dark else BackgroundPolicy.WHITE,
                        custom_bg_color="#212830" if dark else "#ffffff",
                        color_policy=ColorPolicy.COLOR if dark else ColorPolicy.BLACK,
                        lineweight_scaling=1.6)
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=cfg).draw_layout(lay, finalize=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(out)

if __name__ == "__main__":
    import os; os.makedirs("out", exist_ok=True)
    view("../tests/fixtures/dxf/simple_site.dxf", "out/cad_model.png", "Model", True)
    view("../tests/fixtures/dxf/simple_site.dxf", "out/cad_layout.png", "Генплан 1-100", False)
