"""Нейтральная модель CAD-данных: не зависит от ezdxf, хранит происхождение каждого объекта."""
from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]


@dataclass
class CadEntity:
    handle: str
    dxftype: str
    layer: str
    points: list[Point] = field(default_factory=list)      # модельные координаты (единицы чертежа)
    closed: bool = False
    block: str | None = None
    attribs: dict[str, str] = field(default_factory=dict)
    insert: Point | None = None
    rotation: float = 0.0
    xscale: float = 1.0
    yscale: float = 1.0
    width: float | None = None                              # постоянная ширина полилинии
    text: str | None = None
    text_height: float | None = None
    parts: list[list[Point]] = field(default_factory=list)  # замкнутые контуры внутри блока (мод. коорд.)

    @property
    def key(self) -> str:
        return f"{self.dxftype}:{self.handle}"


@dataclass
class ViewportInfo:
    handle: str
    layout: str
    center: Point            # центр вида в модели
    view_height: float       # высота вида в единицах модели
    paper_size_mm: Point     # размер видового экрана на листе
    paper_center_mm: Point   # центр видового экрана на листе
    scale_den: float         # 1:N
    window: tuple[float, float, float, float]  # x0, y0, x1, y1 в модели
    frozen_layers: list[str] = field(default_factory=list)
    twist_deg: float = 0.0

    @property
    def label(self) -> str:
        w, h = self.paper_size_mm
        return f"VP {self.handle} · 1:{self.scale_den:g} · {w:.0f}×{h:.0f} мм"


@dataclass
class LayoutInfo:
    name: str
    paper_size_mm: Point | None
    viewports: list[ViewportInfo]


@dataclass
class LayerInfo:
    name: str
    on: bool = True
    frozen: bool = False
    color: int = 7


@dataclass
class CadDocument:
    path: str
    sha256: str
    insunits: int
    units_to_m: float | None     # None — единицы не заданы, нужно подтверждение
    layers: dict[str, LayerInfo]
    layouts: list[LayoutInfo]
    entities: list[CadEntity]
    warnings: list[str] = field(default_factory=list)
    converted_from: str | None = None

    def layout(self, name: str | None) -> LayoutInfo:
        if name is None:
            return self.layouts[0]
        for lay in self.layouts:
            if lay.name == name:
                return lay
        raise KeyError(f"Лист «{name}» не найден. Есть: {[l.name for l in self.layouts]}")
