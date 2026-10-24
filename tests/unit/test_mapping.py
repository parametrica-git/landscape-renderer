from landscape_renderer.cad.model import CadEntity
from landscape_renderer.mapping import Profile, assign, CONFIRMED, SUGGESTED, UNMAPPED


def P(rules):
    return Profile.from_dict({"schema_version": "1.0", "id": "t", "rules": rules})


def E(layer="0", t="LWPOLYLINE", block=None, attribs=None):
    return CadEntity(handle="1", dxftype=t, layer=layer, block=block, attribs=attribs or {})


def test_glob_case_insensitive():
    p = P([{"id": "lawn", "match": {"layer": ["*газон*"]}, "assign": {"class": "ground.lawn"}}])
    a = assign(E("ЗОНА_ГАЗОН_1"), p)
    assert (a.cls, a.state) == ("ground.lawn", CONFIRMED)


def test_priority_then_specificity():
    p = P([{"id": "any", "priority": 100, "match": {"layer": ["*"]}, "assign": {"class": "ignore"}},
           {"id": "tree", "priority": 100, "match": {"entity": ["INSERT"], "block": ["ДЕРЕВО"]},
            "assign": {"class": "vegetation.tree.deciduous"}}])
    assert assign(E("X", "INSERT", "ДЕРЕВО"), p).rule == "tree"


def test_conflict_reported():
    p = P([{"id": "a", "match": {"layer": ["L"]}, "assign": {"class": "ground.lawn"}},
           {"id": "b", "match": {"layer": ["L"]}, "assign": {"class": "ground.gravel"}}])
    assert assign(E("L"), p).conflict


def test_attribute_condition_and_status_modifier():
    p = P([{"id": "tree", "match": {"block": ["ДЕРЕВО"], "attributes": {"ВИД": "present"}},
            "assign": {"class": "vegetation.tree.deciduous"}},
           {"id": "rm", "priority": 200, "match": {"attributes": {"СТАТУС": "удал*"}}, "assign": {"status": "remove"}}])
    a = assign(E("0", "INSERT", "ДЕРЕВО", {"ВИД": "липа", "СТАТУС": "удаляемое"}), p)
    assert a.cls == "vegetation.tree.deciduous" and a.status == "remove"
    assert assign(E("0", "INSERT", "ДЕРЕВО", {}), p).state != CONFIRMED


def test_manual_beats_profile():
    p = P([{"id": "lawn", "match": {"layer": ["ГАЗОН"]}, "assign": {"class": "ground.lawn"}}])
    a = assign(E("ГАЗОН"), p, {"layer:ГАЗОН": "ground.gravel"})
    assert (a.cls, a.state) == ("ground.gravel", CONFIRMED)


def test_autosuggest_and_unmapped():
    assert assign(E("lawn_area"), None).state == SUGGESTED
    assert assign(E("XYZ_123"), None).state == UNMAPPED


def test_regex_optional():
    p = P([{"id": "r", "match": {"layer_regex": r"GR(A|O)SS_\d+"}, "assign": {"class": "ground.lawn"}}])
    assert assign(E("GRASS_01"), p).cls == "ground.lawn"
    assert assign(E("GRASS_X"), p).rule != "r"   # regex не сработал → только подсказка синонимов


def test_starter_profile_maps_everything(site, profile_path):
    from landscape_renderer.cad import read_dxf
    from landscape_renderer.mapping import build_report, summary
    d = read_dxf(site)
    p = Profile.load(profile_path)
    s = summary(build_report(d.entities, [assign(e, p) for e in d.entities]))
    assert s[UNMAPPED] == 0 and s[SUGGESTED] == 0
