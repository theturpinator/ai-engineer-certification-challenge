"""Pure-function tests for the multi-car garage model: legacy migration and
chat-driven car targeting. No network, no DB."""

from app import _match_car, _migrate


def test_migrate_wraps_legacy_flat_profile():
    legacy = {"year": 2016, "trim": "GT", "mods": ["intake"], "goals": ["track"]}
    p = _migrate(legacy)
    (car,) = p["cars"]
    assert car["id"]
    assert car["year"] == 2016 and car["trim"] == "GT" and car["mods"] == ["intake"]
    assert p["goals"] == ["track"]
    assert "year" not in p and "mods" not in p
    assert _migrate(dict(p)) == p  # idempotent


def test_migrate_leaves_new_shape_and_empty_alone():
    assert _migrate({}) == {}
    p = {"cars": [{"id": "a", "year": 1990}], "goals": []}
    assert _migrate(dict(p)) == p


GT = {"id": "aaa", "year": 2016, "trim": "GT Premium", "generation": "S550"}
FOX = {"id": "bbb", "year": 1990, "trim": "LX 5.0", "generation": "Fox-body"}


def test_match_by_id_and_description():
    cars = [GT, FOX]
    assert _match_car(cars, "bbb", {}) is FOX
    assert _match_car(cars, "my 2016 GT", {}) is GT
    assert _match_car(cars, "the Fox-body", {}) is FOX
    assert _match_car(cars, "the LX", {"mods": ["gears"]}) is FOX


def test_unmatched_description_or_identity_creates_new():
    cars = [GT, FOX]
    assert _match_car(cars, "my new 1969 Mach 1", {"year": 1969}) is None
    assert _match_car(cars, None, {"year": 1969}) is None
    assert _match_car([], "my 2016 GT", {"year": 2016}) is None


def test_no_description_falls_back_sensibly():
    assert _match_car([GT], None, {"mods": ["exhaust"]}) is GT  # lone car
    assert _match_car([GT], None, {"year": 2016, "color": "red"}) is GT  # agrees
    assert _match_car([GT], None, {"year": 1990}) is None  # conflicts -> new car
    assert _match_car([GT, FOX], None, {"year": 1990}) is FOX  # identity picks
    assert _match_car([GT, FOX], None, {"mods": ["oil"]}) is GT  # ambiguous -> first
