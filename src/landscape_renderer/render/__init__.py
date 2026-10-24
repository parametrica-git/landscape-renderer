from .procedural import ProceduralBackend, RenderResult, STYLE_KEYS
from .npr.common import STYLE_RU


def get_backend(name: str = "cpu-procedural"):
    """RendererBackend: render(scene, settings, preview_scale) -> RenderResult."""
    if name == "cpu-procedural":
        return ProceduralBackend()
    raise KeyError(name)
