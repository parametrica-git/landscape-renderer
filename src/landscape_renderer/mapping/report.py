"""Отчёт сопоставления (MAP-03, EXP-06): группы «слой / блок» со статусами."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..cad.model import CadEntity
from .profile import CONFIRMED, SUGGESTED, UNMAPPED, Assignment


@dataclass
class Group:
    key: str               # "block:ДЕРЕВО" или "layer:ГАЗОН"
    kind: str              # block / layer
    name: str
    entity_types: Counter
    count: int
    cls: str | None
    state: str
    rule: str | None
    conflicts: list[str] = field(default_factory=list)
    statuses: Counter = field(default_factory=Counter)


def group_key(e: CadEntity) -> tuple[str, str, str]:
    if e.dxftype == "INSERT" and e.block:
        return f"block:{e.block}", "block", e.block
    return f"layer:{e.layer}", "layer", e.layer


def build_report(entities: list[CadEntity], assigns: list[Assignment]) -> list[Group]:
    groups: dict[str, Group] = {}
    clss: dict[str, Counter] = defaultdict(Counter)
    for e, a in zip(entities, assigns):
        k, kind, name = group_key(e)
        g = groups.get(k)
        if g is None:
            g = groups[k] = Group(k, kind, name, Counter(), 0, None, UNMAPPED, None)
        g.entity_types[e.dxftype] += 1
        g.count += 1
        g.statuses[a.status] += 1
        clss[k][(a.cls, a.state, a.rule)] += 1
        if a.conflict:
            g.conflicts.append(a.conflict)
    order = {CONFIRMED: 0, SUGGESTED: 1, UNMAPPED: 2}
    for k, g in groups.items():
        (cls, state, rule), _ = clss[k].most_common(1)[0]
        g.cls, g.state, g.rule = cls, state, rule
    # сначала то, что требует внимания: не сопоставлено → предложено → подтверждено
    return sorted(groups.values(), key=lambda g: (-order[g.state], g.kind, g.name))


def summary(groups: list[Group]) -> dict:
    c = Counter()
    for g in groups:
        c[g.state] += g.count
    return {CONFIRMED: c[CONFIRMED], SUGGESTED: c[SUGGESTED], UNMAPPED: c[UNMAPPED]}
