"""Окно PySide6 (MVP): открыть → лист → правила → стиль → превью → правка объекта → экспорт."""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QSpinBox, QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget)

from .. import __version__
from ..export.writers import save_dxf_bundle, save_pdf, save_png
from ..mapping import CLASSES, CONFIRMED, SUGGESTED, UNMAPPED
from ..mapping.profile import Profile
from ..pipeline import Session
from ..project import Project
from ..render import STYLE_KEYS, STYLE_RU
from ..render.procedural import ProceduralBackend

STATE_RU = {CONFIRMED: "✓ подтверждено", SUGGESTED: "? предложено", UNMAPPED: "✗ не сопоставлено"}
STATE_COLOR = {CONFIRMED: "#2e7d32", SUGGESTED: "#b26a00", UNMAPPED: "#c62828"}
STATUS_RU = {"existing": "существующее", "new": "новое", "remove": "удаляемое"}


def pil_to_qpixmap(img) -> QPixmap:
    img = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    q = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class _Signals(QObject):
    done = Signal(object, float)
    failed = Signal(str)


class RenderJob(QRunnable):
    def __init__(self, session: Session, scale: float):
        super().__init__()
        self.session, self.scale, self.sig = session, scale, _Signals()

    @Slot()
    def run(self):
        try:
            self.sig.done.emit(self.session.render(self.scale), self.scale)
        except Exception:
            self.sig.failed.emit(traceback.format_exc())


class Preview(QGraphicsView):
    clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#e9e6df"))
        self._fit = True

    def set_image(self, pm: QPixmap):
        self.item.setPixmap(pm)
        self.scene().setSceneRect(self.item.boundingRect())
        if self._fit:
            self.fitInView(self.item, Qt.KeepAspectRatio)

    def wheelEvent(self, ev):
        self._fit = False
        f = 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2
        self.scale(f, f)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._fit and not self.item.pixmap().isNull():
            self.fitInView(self.item, Qt.KeepAspectRatio)

    def mouseDoubleClickEvent(self, ev):
        p = self.mapToScene(ev.position().toPoint())
        self.clicked.emit(p.x(), p.y())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Landscape Renderer {__version__}")
        self.resize(1500, 920)
        self.session: Session | None = None
        self.last = None
        self.pool = QThreadPool.globalInstance()
        self.selected = None

        self.preview = Preview()
        self.preview.clicked.connect(self.pick)
        self.setCentralWidget(self.preview)
        self._toolbar()
        self._left()
        self._right()
        self.statusBar().showMessage("Откройте DXF/DWG или проект .lscapeproj")

    # ---------------------------------------------------------------- панели
    def _toolbar(self):
        tb = QToolBar("Файл")
        self.addToolBar(tb)
        for text, fn, key in (("Открыть…", self.open_dialog, QKeySequence.Open),
                              ("Обновить из CAD", self.reload_source, QKeySequence.Refresh),
                              ("Сохранить проект", self.save_project, QKeySequence.Save),
                              ("PNG", lambda: self.export("png"), None), ("PDF", lambda: self.export("pdf"), None),
                              ("DXF-комплект", lambda: self.export("dxf"), None)):
            a = QAction(text, self)
            if key:
                a.setShortcut(key)
            a.triggered.connect(fn)
            tb.addAction(a)

    def _left(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Лист и видовой экран")
        f = QFormLayout(g)
        self.cb_layout, self.cb_vp = QComboBox(), QComboBox()
        self.cb_layout.currentTextChanged.connect(self.on_layout)
        self.cb_vp.currentIndexChanged.connect(self.on_vp)
        self.lb_scale = QLabel("—")
        f.addRow("Лист", self.cb_layout)
        f.addRow("Видовой экран", self.cb_vp)
        f.addRow("Масштаб", self.lb_scale)
        v.addWidget(g)
        g = QGroupBox("Сопоставление: слой / блок → элемент подачи")
        gv = QVBoxLayout(g)
        h = QHBoxLayout()
        self.lb_profile = QLabel("профиль: —")
        b = QPushButton("Профиль…")
        b.clicked.connect(self.choose_profile)
        h.addWidget(self.lb_profile, 1)
        h.addWidget(b)
        gv.addLayout(h)
        self.lb_summary = QLabel("")
        gv.addWidget(self.lb_summary)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "Слой / блок", "Шт.", "Элемент подачи"])
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        gv.addWidget(self.table)
        self.lb_warn = QLabel("")
        self.lb_warn.setWordWrap(True)
        self.lb_warn.setStyleSheet("color:#8a4b00")
        gv.addWidget(self.lb_warn)
        v.addWidget(g, 1)
        d = QDockWidget("Чертёж", self)
        d.setWidget(w)
        d.setFeatures(QDockWidget.DockWidgetMovable)
        d.setMinimumWidth(470)
        self.addDockWidget(Qt.LeftDockWidgetArea, d)

    def _right(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Стиль")
        f = QFormLayout(g)
        self.cb_style = QComboBox()
        for k in STYLE_KEYS:
            self.cb_style.addItem(STYLE_RU.get(k, k), k)
        self.cb_style.setCurrentIndex(STYLE_KEYS.index("pero_akvarel"))
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 99999)
        self.sp_seed.setValue(21)
        self.sp_az = QSpinBox()
        self.sp_az.setRange(0, 359)
        self.sp_az.setValue(315)
        self.sp_az.setSuffix("°")
        self.sp_sh = QDoubleSpinBox()
        self.sp_sh.setRange(0.1, 1.5)
        self.sp_sh.setSingleStep(0.05)
        self.sp_sh.setValue(0.55)
        self.sp_dpi = QSpinBox()
        self.sp_dpi.setRange(72, 600)
        self.sp_dpi.setValue(300)
        self.ch_ann = QCheckBox("Надписи из CAD поверх")
        self.ch_ann.setChecked(True)
        self.ch_st = QCheckBox("Статусы: новое / удаляемое")
        self.ch_st.setChecked(True)
        f.addRow("Техника", self.cb_style)
        f.addRow("Свет (азимут)", self.sp_az)
        f.addRow("Длина теней", self.sp_sh)
        f.addRow("Вариация (seed)", self.sp_seed)
        f.addRow("DPI экспорта", self.sp_dpi)
        f.addRow(self.ch_ann)
        f.addRow(self.ch_st)
        self.bt_render = QPushButton("Обновить превью")
        self.bt_render.clicked.connect(self.render_preview)
        f.addRow(self.bt_render)
        v.addWidget(g)
        g = QGroupBox("Объект (двойной щелчок на превью)")
        f = QFormLayout(g)
        self.lb_obj = QLabel("—")
        self.lb_obj.setWordWrap(True)
        self.cb_status = QComboBox()
        for k, t in STATUS_RU.items():
            self.cb_status.addItem(t, k)
        self.bt_var = QPushButton("Другой вариант")
        self.bt_hide = QPushButton("Скрыть")
        self.bt_status = QPushButton("Применить статус")
        self.bt_var.clicked.connect(lambda: self.obj_action("variant"))
        self.bt_hide.clicked.connect(lambda: self.obj_action("hide"))
        self.bt_status.clicked.connect(lambda: self.obj_action("status"))
        f.addRow(self.lb_obj)
        f.addRow("Проектный статус", self.cb_status)
        hb = QHBoxLayout()
        for b in (self.bt_status, self.bt_var, self.bt_hide):
            hb.addWidget(b)
        f.addRow(hb)
        v.addWidget(g)
        v.addStretch(1)
        d = QDockWidget("Подача", self)
        d.setWidget(w)
        d.setFeatures(QDockWidget.DockWidgetMovable)
        d.setMinimumWidth(330)
        self.addDockWidget(Qt.RightDockWidgetArea, d)

    # ---------------------------------------------------------------- данные
    def open_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Открыть", "", "Чертежи и проекты (*.dxf *.dwg *.lscapeproj)")
        if p:
            self.open_path(p)

    def open_path(self, path: str):
        try:
            proj = Project.load(path) if path.lower().endswith(".lscapeproj") else Project(source=path)
            self.session = Session.open(proj)
        except Exception as e:
            QMessageBox.critical(self, "Не удалось открыть", str(e))
            return
        s = self.session
        self.setWindowTitle(f"Landscape Renderer — {Path(s.project.source).name}")
        self.cb_layout.blockSignals(True)
        self.cb_layout.clear()
        for lay in s.cad.layouts:
            if lay.viewports:
                self.cb_layout.addItem(lay.name)
        self.cb_layout.setCurrentText(s.viewport.layout)
        self.cb_layout.blockSignals(False)
        self._fill_vps()
        r = s.project.render
        self.cb_style.setCurrentIndex(STYLE_KEYS.index(r.style) if r.style in STYLE_KEYS else 0)
        self.sp_seed.setValue(r.seed)
        self.sp_az.setValue(int(r.light_azimuth_deg))
        self.refresh_mapping()
        warns = list(s.cad.warnings) + (["Источник изменился с прошлого сохранения."] if s.source_changed else [])
        self.lb_warn.setText("\n".join("! " + w for w in warns))
        self.render_preview()

    def _fill_vps(self):
        s = self.session
        self.cb_vp.blockSignals(True)
        self.cb_vp.clear()
        lay = s.cad.layout(self.cb_layout.currentText()) if self.cb_layout.count() else None
        for vp in (lay.viewports if lay else []):
            self.cb_vp.addItem(vp.label, vp.handle)
        i = self.cb_vp.findData(s.viewport.handle)
        self.cb_vp.setCurrentIndex(max(0, i))
        self.cb_vp.blockSignals(False)
        self.lb_scale.setText(f"1:{s.viewport.scale_den:g}; на листе {s.viewport.paper_size_mm[0]:.0f}×"
                              f"{s.viewport.paper_size_mm[1]:.0f} мм")

    def on_layout(self, name):
        if self.session and name:
            self.session.select_viewport(name)
            self._fill_vps()
            self.refresh_mapping()
            self.render_preview()

    def on_vp(self, _):
        if self.session and self.cb_vp.currentData():
            self.session.select_viewport(self.cb_layout.currentText(), self.cb_vp.currentData())
            self._fill_vps()
            self.refresh_mapping()
            self.render_preview()

    def choose_profile(self):
        if not self.session:
            return
        p, _ = QFileDialog.getOpenFileName(self, "Профиль сопоставления", "", "Профиль (*.json)")
        if p:
            self.session.profile = Profile.load(p)
            self.session.project.profile = p
            self.session.rebuild()
            self.refresh_mapping()
            self.render_preview()

    def refresh_mapping(self):
        s = self.session
        groups, summ = s.report()
        self.lb_profile.setText(f"профиль: {s.profile.name if s.profile else '— (автоподбор)'}")
        self.lb_summary.setText(f"<b style='color:{STATE_COLOR[CONFIRMED]}'>{summ[CONFIRMED]}</b> подтверждено · "
                                f"<b style='color:{STATE_COLOR[SUGGESTED]}'>{summ[SUGGESTED]}</b> предложено · "
                                f"<b style='color:{STATE_COLOR[UNMAPPED]}'>{summ[UNMAPPED]}</b> не сопоставлено")
        self.table.setRowCount(len(groups))
        for r, g in enumerate(groups):
            it = QTableWidgetItem(STATE_RU[g.state][0])
            it.setToolTip(STATE_RU[g.state] + (f" · правило {g.rule}" if g.rule else ""))
            it.setTextAlignment(Qt.AlignCenter)
            it.setForeground(QColor(STATE_COLOR[g.state]))
            self.table.setItem(r, 0, it)
            name = ("▣ " if g.kind == "block" else "≡ ") + g.name
            if g.conflicts:
                name += "  ⚠"
            it = QTableWidgetItem(name)
            it.setToolTip(f"{g.key} · типы: " + ", ".join(f"{t}×{n}" for t, n in g.entity_types.items()))
            self.table.setItem(r, 1, it)
            self.table.setItem(r, 2, QTableWidgetItem(str(g.count)))
            cb = QComboBox()
            cb.addItem("—", None)
            for k, t in CLASSES.items():
                cb.addItem(t, k)
            cb.setCurrentIndex(max(0, cb.findData(g.cls)))
            cb.currentIndexChanged.connect(lambda _i, key=g.key, c=cb: self.set_manual(key, c.currentData()))
            self.table.setCellWidget(r, 3, cb)

    def set_manual(self, key, cls):
        s = self.session
        if cls is None:
            s.project.manual.pop(key, None)
        else:
            s.project.manual[key] = cls
        s.rebuild()
        self.refresh_mapping()
        self.render_preview()

    def _settings(self):
        r = self.session.project.render
        r.style = self.cb_style.currentData()
        r.seed = self.sp_seed.value()
        r.light_azimuth_deg = float(self.sp_az.value())
        r.shadow_k = float(self.sp_sh.value())
        r.annotations = self.ch_ann.isChecked()
        r.statuses = self.ch_st.isChecked()
        r.dpi = self.sp_dpi.value()

    # ---------------------------------------------------------------- рендер
    def render_preview(self, sync: bool = False):
        if not self.session:
            return
        self._settings()
        scale = min(1.0, 110.0 / self.session.project.render.dpi)
        self.bt_render.setEnabled(False)
        self.statusBar().showMessage("Строю превью…")
        job = RenderJob(self.session, scale)
        job.sig.done.connect(self.on_rendered)
        job.sig.failed.connect(self.on_failed)
        if sync:
            job.run()
        else:
            self.pool.start(job)

    def on_rendered(self, res, scale):
        self.last = res
        self.preview.set_image(pil_to_qpixmap(res.image))
        self.bt_render.setEnabled(True)
        n = len(self.session.scene.objects)
        self.statusBar().showMessage(f"Превью {res.image.size[0]}×{res.image.size[1]} px · объектов {n} · "
                                     f"{STYLE_RU.get(self.session.project.render.style)}")

    def on_failed(self, tb):
        self.bt_render.setEnabled(True)
        QMessageBox.critical(self, "Ошибка рендера", tb[-2000:])

    # ---------------------------------------------------------------- объекты
    def pick(self, x_px, y_px):
        if not (self.session and self.last):
            return
        res = self.last
        ppmm = res.dpi / 25.4
        k = self.session.viewport.scale_den / 1000.0
        xm = (x_px / ppmm - res.origin_mm[0]) * k
        ym = (y_px / ppmm - res.origin_mm[1]) * k
        best, bd = None, 1e9
        for o in self.session.scene.objects:
            if o.cls.startswith("vegetation") and o.center and o.diameter_m:
                d = math.dist(o.center, (xm, ym))
                if d <= o.diameter_m / 2 and d < bd:
                    best, bd = o, d
        if best is None:
            import cv2
            for o in self.session.scene.objects:
                for p in o.polys:
                    if len(p) >= 3 and cv2.pointPolygonTest(np.asarray(p, np.float32), (xm, ym), False) >= 0:
                        best = o
        self.select(best)

    def select(self, o):
        self.selected = o
        if o is None:
            self.lb_obj.setText("—")
            return
        txt = f"<b>{CLASSES.get(o.cls, o.cls)}</b> · handle {o.id}<br>слой {o.layer}"
        if o.block:
            txt += f", блок {o.block}"
        if o.plant:
            txt += f"<br>вид: {o.plant}"
        if o.diameter_m:
            txt += f", D = {o.diameter_m:.1f} м"
        self.lb_obj.setText(txt)
        self.cb_status.setCurrentIndex(max(0, self.cb_status.findData(o.status)))

    def obj_action(self, what):
        o, s = self.selected, self.session
        if not (o and s):
            return
        ov = s.project.overrides.get(o.id, {})
        if what == "variant":
            s.set_override(o.id, variant=int(ov.get("variant", 0)) + 1)
        elif what == "hide":
            s.set_override(o.id, hidden=True)
        elif what == "status":
            s.set_override(o.id, status=self.cb_status.currentData())
        self.render_preview()

    # ---------------------------------------------------------------- файлы
    def reload_source(self):
        if self.session:
            self.open_path(self.session.project.path or self.session.project.source)

    def save_project(self):
        if not self.session:
            return
        self._settings()
        p = self.session.project.path or str(Path(self.session.project.source).with_suffix(".lscapeproj"))
        p, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", p, "Проект (*.lscapeproj)")
        if p:
            self.session.project.save(p)
            self.statusBar().showMessage(f"Проект сохранён: {p}")

    def export(self, kind):
        if not self.session:
            return
        self._settings()
        stem = Path(self.session.project.source).stem + "_" + self.session.project.render.style
        start = str(Path(self.session.project.source).parent / stem)
        if kind == "dxf":
            d = QFileDialog.getExistingDirectory(self, "Папка для комплекта", str(Path(start).parent))
            if not d:
                return
        else:
            p, _ = QFileDialog.getSaveFileName(self, "Экспорт", f"{start}.{kind}", f"*.{kind}")
            if not p:
                return
        self.statusBar().showMessage("Экспорт в полном разрешении…")
        QApplication.processEvents()
        res = ProceduralBackend().render(self.session.scene, self.session.project.render, 1.0)
        if kind == "png":
            out = save_png(res, p)
        elif kind == "pdf":
            out = save_pdf(res, p)
        else:
            out = save_dxf_bundle(res, self.session.scene, self.session.cad.path, Path(d) / f"{stem}_bundle",
                                  name=stem, project_id=self.session.project.project_id)["dxf"]
        self.statusBar().showMessage(f"Сохранено: {out}")


def main(argv=None):
    app = QApplication(argv or sys.argv)
    app.setApplicationName("Landscape Renderer")
    w = MainWindow()
    w.show()
    args = (argv or sys.argv)[1:]
    if args:
        w.open_path(args[0])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
