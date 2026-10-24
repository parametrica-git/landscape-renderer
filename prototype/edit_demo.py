"""Правки дизайнера в CAD: двигаем, меняем породу и размер, добавляем, меняем вид в цветнике."""
import ezdxf
doc = ezdxf.readfile("../tests/fixtures/dxf/simple_site.dxf")
msp = doc.modelspace()
log = []
trees = [e for e in msp.query("INSERT") if e.dxf.name in ("ДЕРЕВО", "ХВОЙНОЕ")]
for e in trees:
    a = {x.dxf.tag: x for x in e.attribs}
    vid = a["ВИД"].dxf.text
    if "клён" in vid:                       # 1. сдвинуть клён на 3 м вправо
        p = e.dxf.insert
        e.dxf.insert = (p.x + 3.0, p.y + 0.4)
        for x in e.attribs:
            q = x.dxf.insert; x.dxf.insert = (q.x + 3.0, q.y + 0.4)
        log.append("клён сдвинут на 3 м вправо")
    if "липа" in vid:                       # 2. заменить липу на иргу, D 4.6 -> 3.2
        a["ВИД"].dxf.text = "ирга канадская"
        a["D"].dxf.text = "3.2"
        e.dxf.xscale = e.dxf.yscale = 1.6
        log.append("липа -> ирга, крона 4.6 -> 3.2 м")
# 3. добавить новое дерево и ель
for blk, lay, xy, D, vid in (("ДЕРЕВО", "ДЕРЕВЬЯ", (1.8, 3.6), 3.6, "липа мелколистная"),
                             ("ХВОЙНОЕ", "ХВОЙНЫЕ", (20.0, 6.0), 2.2, "ель колючая")):
    ins = msp.add_blockref(blk, xy, dxfattribs={"layer": lay, "xscale": D / 2, "yscale": D / 2})
    ins.add_auto_attribs({"ВИД": vid, "D": str(D)})
    log.append(f"добавлено: {vid}, D {D} м")
# 4. в цветнике гауру заменить на лаванду (просто сменой слоя контура)
for e in msp.query("LWPOLYLINE"):
    if e.dxf.layer == "ЦВ_гаура":
        e.dxf.layer = "ЦВ_лаванда"
        log.append("пятно гауры -> лаванда (смена слоя)")
doc.saveas("../tests/fixtures/dxf/simple_site_edited.dxf")
print("\n".join(log))
