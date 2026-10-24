import pytest

pytest.importorskip("PySide6")


def test_ui_opens_and_renders(site):
    from PySide6.QtWidgets import QApplication
    from landscape_renderer.ui.app import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    orig = w.render_preview
    w.render_preview = lambda sync=True: orig(sync=True)
    w.open_path(site)
    assert w.cb_layout.currentText() == "Генплан 1-100"
    assert w.table.rowCount() > 10
    assert w.last is not None
    obj = next(o for o in w.session.scene.objects if o.block == "ДЕРЕВО")
    w.select(obj)
    w.obj_action("hide")
    assert w.session.project.overrides[obj.id]["hidden"] is True
