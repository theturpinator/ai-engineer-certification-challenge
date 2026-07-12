"""Pure-function tests for the multi-car garage model: legacy migration,
chat-driven car targeting, and the portrait/stats staleness fingerprints.
No network, no DB."""

from app import (_autofill_generation, _build_fp, _derive_generation, _identity_fp,
                 _match_car, _migrate, _portrait_action, _stats_fp)


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


def test_partial_car_creation_and_later_year_update():
    # "I have a blue Mustang GT" (no year) -> no match anywhere -> new car
    assert _match_car([], "blue Mustang GT", {"trim": "GT", "color": "blue"}) is None
    assert _match_car([GT, FOX], "a blue mustang", {"color": "blue"}) is None
    # the year arriving later targets the SAME partial car
    partial = {"id": "ccc", "trim": "GT", "color": "blue"}
    assert _match_car([partial], "my blue GT", {"year": 2016}) is partial
    assert _match_car([partial], None, {"year": 2016}) is partial  # lone car


def test_derive_generation_mapping():
    for year, gen in [
        (1964, "First generation"), (1965, "First generation"),
        (1973, "First generation"), (1974, "Mustang II"), (1978, "Mustang II"),
        (1979, "Fox-body"), (1993, "Fox-body"), (1994, "SN95"), (2004, "SN95"),
        (2005, "S197"), (2014, "S197"), (2015, "S550"), (2023, "S550"),
        (2024, "S650"), (2027, "S650"),
    ]:
        assert _derive_generation(year) == gen, year
    assert _derive_generation("2016") == "S550"  # tolerates string years
    assert _derive_generation(1930) is None
    assert _derive_generation("soon") is None
    assert _derive_generation(None) is None


def test_autofill_generation_only_when_missing():
    car = {"id": "x", "year": 2016}
    _autofill_generation(car)
    assert car["generation"] == "S550"
    keep = {"id": "x", "year": 2016, "generation": "custom"}
    _autofill_generation(keep)
    assert keep["generation"] == "custom"  # never overwrites
    partial = {"id": "x", "trim": "GT"}
    _autofill_generation(partial)
    assert "generation" not in partial  # no year -> untouched


CAR = {"id": "aaa", "year": 2016, "generation": "S550", "trim": "GT",
       "color": "Race Red", "mods": ["cold air intake"]}


def _stored(car):
    return (_identity_fp(car), _build_fp(car))


def test_portrait_action_generate_edit_skip():
    assert _portrait_action(None, CAR) == "generate"  # no portrait yet
    assert _portrait_action(_stored(CAR), CAR) == "skip"  # nothing changed

    # mods-list change -> edit the stored photo, never re-roll
    modded = {**CAR, "mods": CAR["mods"] + ["rear spoiler"]}
    assert _portrait_action(_stored(CAR), modded) == "edit"
    # mod removed -> also an edit
    assert _portrait_action(_stored(modded), CAR) == "edit"

    # color change -> edit; identity change -> full regeneration
    assert _portrait_action(_stored(CAR), {**CAR, "color": "Grabber Blue"}) == "edit"
    assert _portrait_action(_stored(CAR), {**CAR, "year": 1969}) == "generate"
    assert _portrait_action(_stored(CAR), {**CAR, "trim": "EcoBoost"}) == "generate"
    assert _portrait_action(_stored(CAR), {**CAR, "generation": "Fox"}) == "generate"


def test_build_fp_order_and_case_insensitive():
    a = {**CAR, "mods": ["spoiler", "intake"], "color": "race red"}
    b = {**CAR, "mods": ["intake", "spoiler"], "color": "Race Red "}
    assert _build_fp(a) == _build_fp(b)  # sorted mods, normalized color


def test_stats_fp_tracks_identity_and_mods_not_color():
    assert _stats_fp(CAR) == _stats_fp({**CAR, "color": "green"})
    assert _stats_fp(CAR) != _stats_fp({**CAR, "mods": ["supercharger"]})
    assert _stats_fp(CAR) != _stats_fp({**CAR, "year": 1969})
