"""Профили сопоставления «чертёж → семантика» (MAP-01…08).

Правило:
    {"id": "...", "priority": 100,
     "match": {"entity": ["INSERT"], "layer": ["*ГАЗОН*"], "block": ["ДЕРЕВО"],
               "attributes": {"ВИД": "present" | "<glob>"}, "layer_regex": "..."},
     "assign": {"class": "...", "status": "existing|remove|new", "plant_from_attribute": "ВИД",
                "diameter_from_attribute": "D", "diameter_unit": "m", "plant_from_layer_suffix": true,
                "height_m": 1.6}}
Порядок: ручные назначения → правила профиля (priority, затем специфичность) → автоподбор.
Правила только со "status" — модификаторы, применяются поверх класса.
"""
from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..cad.model import CadEntity

CONFIRMED, SUGGESTED, UNMAPPED = "confirmed", "suggested", "unmapped"

CLASSES = {
    "building": "Здание", "hard.terrace": "Терраса / мощение", "hard.deck": "Настил",
    "hard.stone": "Плиты-шаги", "rock.boulder": "Валуны", "ground.lawn": "Газон", "ground.gravel": "Гравий",
    "water.pond": "Водоём", "planting.bed": "Цветник (пятно)", "vegetation.tree.deciduous": "Дерево лиственное",
    "vegetation.tree.conifer": "Хвойное", "vegetation.tree.columnar": "Колонновидное",
    "vegetation.shrub": "Кустарник", "vegetation.hedge": "Живая изгородь", "furniture": "Мебель",
    "maf": "МАФ", "boundary": "Граница участка", "annotation": "Надписи", "ignore": "Не рисовать",
}

# Автоподбор: нормализованные фрагменты имён → класс (RU/EN)
SYNONYMS = {
    "ground.lawn": ["газон", "lawn", "grass", "трава"],
    "ground.gravel": ["гравий", "gravel", "щебень", "отсыпка"],
    "water.pond": ["водо", "пруд", "water", "pond", "pool", "бассейн"],
    "hard.deck": ["настил", "deck", "декинг"],
    "hard.terrace": ["терраса", "мощение", "плитка", "paving", "terrace", "patio", "брусчатка"],
    "hard.stone": ["шаг", "stepping"],
    "rock.boulder": ["камни", "валун", "boulder", "rock"],
    "building": ["дом", "здание", "building", "house", "ар_", "a-wall"],
    "planting.bed": ["цв_", "цветник", "миксбордер", "bed", "planting", "клумба"],
    "vegetation.tree.conifer": ["хвойн", "ель", "сосна", "conifer", "pine", "spruce"],
    "vegetation.tree.columnar": ["можжевельник", "туя", "колонн", "juniper", "thuja"],
    "vegetation.tree.deciduous": ["дерев", "tree", "клён", "липа", "дуб"],
    "vegetation.shrub": ["куст", "shrub", "bush"],
    "vegetation.hedge": ["изгород", "hedge"],
    "furniture": ["мебель", "furniture", "стол", "стул"],
    "maf": ["маф", "скамь", "bench", "site_furn"],
    "boundary": ["границ", "boundary", "property", "участок"],
    "annotation": ["текст", "text", "подпис", "anno"],
}


def _norm(s: str | None) -> str:
    return (s or "").strip().lower().replace("ё", "е")


@dataclass
class Rule:
    id: str
    priority: int
    match: dict
    assign: dict

    @property
    def specificity(self) -> int:
        n = 0
        for k, v in self.match.items():
            n += len(v) if isinstance(v, dict) else 1
        return n

    @property
    def is_modifier(self) -> bool:
        return "class" not in self.assign

    def matches(self, e: CadEntity) -> bool:
        m = self.match
        if "entity" in m and e.dxftype not in m["entity"]:
            return False
        if "layer" in m and not any(fnmatch.fnmatchcase(_norm(e.layer), _norm(g)) for g in m["layer"]):
            return False
        if "layer_regex" in m:
            pat = m["layer_regex"]
            if len(pat) > 200 or not re.fullmatch(pat, e.layer or "", flags=re.I):
                return False
        if "block" in m:
            if not e.block or not any(fnmatch.fnmatchcase(_norm(e.block), _norm(g)) for g in m["block"]):
                return False
        for tag, cond in (m.get("attributes") or {}).items():
            val = e.attribs.get(tag)
            if val is None:
                return False
            if cond not in ("present", "*") and not fnmatch.fnmatchcase(_norm(val), _norm(cond)):
                return False
        return True


@dataclass
class Profile:
    id: str
    version: str
    name: str
    rules: list[Rule]
    source: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d, str(path))

    @classmethod
    def from_dict(cls, d: dict, source: str | None = None) -> "Profile":
        if str(d.get("schema_version", "1.0")).split(".")[0] != "1":
            raise ValueError(f"Неподдерживаемая версия схемы профиля: {d.get('schema_version')}")
        rules = [Rule(r["id"], int(r.get("priority", 100)), r.get("match", {}), r.get("assign", {}))
                 for r in d.get("rules", [])]
        return cls(d["id"], d.get("version", "0"), d.get("name", d["id"]), rules, source)


@dataclass
class Assignment:
    cls: str | None
    state: str                       # confirmed / suggested / unmapped
    rule: str | None = None
    status: str = "existing"         # проектный статус: existing / remove / new
    params: dict = field(default_factory=dict)
    conflict: str | None = None


def suggest(e: CadEntity) -> str | None:
    names = [_norm(e.block), _norm(e.layer)]
    for cls, keys in SYNONYMS.items():
        for n in names:
            if n and any(k in n for k in keys):
                if cls.startswith("vegetation") and e.dxftype != "INSERT" and cls != "vegetation.hedge":
                    continue
                return cls
    return None


def manual_key(e: CadEntity) -> list[str]:
    keys = [f"handle:{e.handle}"]
    if e.block:
        keys.append(f"block:{e.block}")
    keys.append(f"layer:{e.layer}")
    return keys


def assign(e: CadEntity, profile: Profile | None, manual: dict[str, str] | None = None) -> Assignment:
    manual = manual or {}
    status = "existing"
    params: dict = {}
    if profile:
        for r in sorted((r for r in profile.rules if r.is_modifier and r.matches(e)), key=lambda r: -r.priority):
            status = r.assign.get("status", status)
            params.update({k: v for k, v in r.assign.items() if k != "status"})
    for k in manual_key(e):
        if k in manual:
            return Assignment(manual[k], CONFIRMED, f"manual:{k}", status, params)
    if profile:
        hits = sorted((r for r in profile.rules if not r.is_modifier and r.matches(e)),
                      key=lambda r: (-r.priority, -r.specificity))
        if hits:
            top = hits[0]
            conflict = None
            if len(hits) > 1:
                second = hits[1]
                if (second.priority, second.specificity) == (top.priority, top.specificity) \
                        and second.assign.get("class") != top.assign.get("class"):
                    conflict = f"правила «{top.id}» и «{second.id}» равны по приоритету и специфичности"
            p = dict(params)
            p.update({k: v for k, v in top.assign.items() if k not in ("class", "status")})
            st = top.assign.get("status", status)
            return Assignment(top.assign["class"], CONFIRMED, top.id, st, p, conflict)
    s = suggest(e)
    if s:
        return Assignment(s, SUGGESTED, "synonyms", status, params)
    return Assignment(None, UNMAPPED, None, status, params)
