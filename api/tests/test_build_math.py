"""Pure-function tests for the deterministic dream-build math: alias
matching, generation resolution, and stock + installed = current,
current + wishlist = dream composition with clamping. No network, no DB."""

from app import BAR_STATS, _compose_bars, _gen_key, _match_entry, _sum_deltas

SC = {  # generic supercharger
    "id": "generic-supercharger", "name": "Supercharger",
    "aliases": ["blower", "whipple"],
    "deltas": {"S550": {"power": 15, "acceleration": 12, "top_speed": 8,
                        "handling": 0, "braking": 0},
               "Fox-body": {"power": 20, "acceleration": 15, "top_speed": 10,
                            "handling": 0, "braking": 0}},
}
CAI = {
    "id": "generic-cold-air-intake", "name": "Cold air intake",
    "aliases": ["cai", "intake"],
    "deltas": {"S550": {"power": 2, "acceleration": 1, "top_speed": 0,
                        "handling": 0, "braking": 0}},
}
TKX = {
    "id": "tremec-tkx", "name": "TREMEC TKX 5-Speed Manual Transmission",
    "aliases": ["TKX", "Tremec TKX"],
    "deltas": {"Fox-body": {"power": 0, "acceleration": 2, "top_speed": 1,
                            "handling": 0, "braking": 0},
               "S550": {"power": 0, "acceleration": 0, "top_speed": 0,
                        "handling": 0, "braking": 0}},
}
CATALOG = [SC, CAI, TKX]


def test_match_entry_by_name_alias_and_word_boundary():
    assert _match_entry("Supercharger", CATALOG) is SC
    assert _match_entry("whipple kit", CATALOG) is SC  # alias inside free text
    assert _match_entry("tkx", CATALOG) is TKX  # case-insensitive
    assert _match_entry("custom paint job", CATALOG) is None  # unrecognized
    assert _match_entry("painted intake manifold", CATALOG) is CAI
    assert _match_entry("blowers are cool", CATALOG) is None  # whole words only


def test_match_entry_longest_alias_wins():
    # "cold air intake" (full name) must beat the shorter "intake" alias
    assert _match_entry("JLT cold air intake", CATALOG) is CAI
    # both "tkx" and "tremec tkx" match; the longer, more specific one wins
    assert _match_entry("Tremec TKX swap", CATALOG) is TKX


def test_gen_key_normalizes_variants():
    for raw in ("S550", "s550", "S-550"):
        assert _gen_key(raw) == "S550"
    for raw in ("Fox-body", "Fox Body", "foxbody", "FOX BODY"):
        assert _gen_key(raw) == "Fox-body"
    assert _gen_key("First generation") == "First generation"
    assert _gen_key(None) is None
    assert _gen_key("classic") is None


def test_sum_deltas_per_generation_and_unmatched_zero():
    total = _sum_deltas(["whipple", "mystery part"], "S550", CATALOG)
    assert total["power"] == 15 and total["acceleration"] == 12
    fox = _sum_deltas(["whipple", "tkx"], "Fox-body", CATALOG)
    assert fox["power"] == 20 and fox["acceleration"] == 17
    assert _sum_deltas(["whipple"], None, CATALOG) == {s: 0 for s in BAR_STATS}


CAR = {"id": "a", "generation": "S550", "mods": [], "wishlist": [],
       "stats": {"power": 80, "acceleration": 78, "top_speed": 72,
                 "handling": 70, "braking": 72}}


def test_compose_stock_plus_installed_is_current():
    bars = _compose_bars({**CAR, "mods": ["supercharger"]}, CATALOG)
    assert bars["current"]["power"] == 95  # 80 + 15
    assert bars["current"]["acceleration"] == 90  # 78 + 12
    assert bars["current"]["handling"] == 70  # untouched stat
    assert bars["dream"] == bars["current"]  # empty wishlist


def test_compose_current_plus_wishlist_is_dream():
    bars = _compose_bars({**CAR, "mods": ["cold air intake"],
                          "wishlist": ["supercharger"]}, CATALOG)
    assert bars["current"]["power"] == 82  # 80 + 2
    assert bars["dream"]["power"] == 97  # 82 + 15
    assert bars["dream"]["acceleration"] == 91  # 78 + 1 + 12


def test_compose_clamps_to_100():
    hot = {**CAR, "stats": {**CAR["stats"], "power": 95},
           "mods": ["supercharger"], "wishlist": ["whipple"]}
    bars = _compose_bars(hot, CATALOG)
    assert bars["current"]["power"] == 100
    assert bars["dream"]["power"] == 100


def test_compose_no_stats_or_unknown_generation():
    assert _compose_bars({"id": "x", "mods": ["supercharger"]}, CATALOG) is None
    bars = _compose_bars({**CAR, "generation": "spaceship",
                          "mods": ["supercharger"]}, CATALOG)
    assert bars["current"]["power"] == 80  # no gen -> deltas contribute zero


def test_compose_same_input_same_output():
    car = {**CAR, "mods": ["supercharger", "tkx"], "wishlist": ["intake"]}
    assert _compose_bars(car, CATALOG) == _compose_bars(car, CATALOG)
