from landscape_renderer.cad import read_dxf, pick_viewport
from landscape_renderer.mapping import Profile
from landscape_renderer.project import Project
from landscape_renderer.scene import build_scene, match_overrides


def test_roundtrip_relative_paths(tmp_path, site, profile_path):
    p = Project(source=site, layout="Генплан 1-100", profile=profile_path,
                manual={"layer:ГАЗОН": "ground.lawn"}, overrides={"ABC": {"hidden": True}})
    p.render.seed = 7
    f = p.save(str(tmp_path / "x.lscapeproj"))
    q = Project.load(f)
    assert q.source == site and q.render.seed == 7 and q.overrides["ABC"]["hidden"]
    assert q.manual == p.manual


def test_override_survives_handle_change(site, profile_path):
    d = read_dxf(site)
    vp = pick_viewport(d)
    prof = Profile.load(profile_path)
    s = build_scene(d, vp, prof)
    tree = next(o for o in s.objects if o.block == "ДЕРЕВО")
    ov = {"OLD_HANDLE": {"variant": 2, "_fp": dict(tree.fingerprint, handle="OLD_HANDLE")}}
    out, warn = match_overrides(ov, s)
    assert tree.id in out and out[tree.id]["variant"] == 2 and warn


def test_scene_uses_attributes(site, profile_path):
    d = read_dxf(site)
    s = build_scene(d, pick_viewport(d), Profile.load(profile_path))
    trees = s.of("vegetation.tree.deciduous")
    assert {round(o.diameter_m, 1) for o in trees} == {5.4, 4.6, 3.0}
    assert all(o.plant for o in trees)
