"""DWG → DXF средствами, которые есть у пользователя.

Порядок поиска:
1. AutoCAD Core Console (accoreconsole.exe) — лицензия AutoCAD пользователя.
2. ODA File Converter — ВНИМАНИЕ: без членства в ODA разрешён только для некоммерческого
   использования (https://www.opendesign.com/node/3218). Используется лишь по явному разрешению.
Исходный DWG не изменяется, результат пишется во временную папку.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class DwgConversionError(RuntimeError):
    pass


def find_accoreconsole() -> str | None:
    for pat in (r"C:\Program Files\Autodesk\AutoCAD *\accoreconsole.exe",):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return shutil.which("accoreconsole")


def find_oda() -> str | None:
    for pat in (r"C:\Program Files\ODA\ODAFileConverter*\ODAFileConverter.exe",
                "/usr/bin/ODAFileConverter", "/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return shutil.which("ODAFileConverter")


def convert(dwg: str | Path, allow_oda: bool = False, timeout: int = 300) -> str:
    dwg = Path(dwg).resolve()
    out_dir = Path(tempfile.mkdtemp(prefix="lr_dwg_"))
    out = out_dir / (dwg.stem + ".dxf")
    acc = find_accoreconsole()
    if acc:
        scr = out_dir / "to_dxf.scr"
        scr.write_text(f'_.DXFOUT\n"{out}"\n_V\n2018\n16\n_.QUIT\n_Y\n', encoding="utf-8")
        r = subprocess.run([acc, "/i", str(dwg), "/s", str(scr)], capture_output=True, timeout=timeout)
        if out.exists():
            return str(out)
        raise DwgConversionError(f"accoreconsole не создал DXF (код {r.returncode}). "
                                 "Сохраните чертёж как DXF вручную.")
    oda = find_oda()
    if oda and allow_oda:
        src_dir = out_dir / "src"
        src_dir.mkdir()
        shutil.copy2(dwg, src_dir / dwg.name)
        r = subprocess.run([oda, str(src_dir), str(out_dir), "ACAD2018", "DXF", "0", "1", "*.DWG"],
                           capture_output=True, timeout=timeout)
        if out.exists():
            return str(out)
        raise DwgConversionError(f"ODA File Converter не создал DXF (код {r.returncode}).")
    if oda and not allow_oda:
        raise DwgConversionError("Найден ODA File Converter, но без членства ODA он разрешён только "
                                 "для некоммерческого использования. Включите его явно (--allow-oda) "
                                 "или сохраните чертёж как DXF в своём CAD.")
    raise DwgConversionError("Не найден конвертер DWG. Сохраните чертёж как DXF (AutoCAD: «Сохранить как» → DXF).")


def open_any(path: str | Path, allow_oda: bool = False):
    from .reader import read_dxf
    p = str(path)
    if p.lower().endswith(".dwg"):
        dxf = convert(p, allow_oda=allow_oda)
        doc = read_dxf(dxf)
        doc.converted_from = p
        return doc
    return read_dxf(p)


if os.name != "nt":  # на не-Windows accoreconsole не бывает
    find_accoreconsole = lambda: None  # noqa: E731
