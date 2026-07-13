"""Seam-2 tests for ads ingestion: CSV row in, catalog entries out.
Deterministic, no network — the vision/LLM analysis stays behind the same
pure/impure split the article pipeline uses."""

from ingest_ads import (
    ELIGIBLE_CLASSIFICATIONS,
    GENERATION_NAMES,
    STATS,
    advertiser_record,
    canonical_website,
    catalog_entries,
    dedupe_records,
    embed_text,
    generic_entries,
    load_advertisers,
    normalize_deltas,
    zero_deltas,
)

UTM_LINK = "https://vendor.example/?utm_source=mustang-driver&utm_medium=sponsor-banner"


def make_row(**overrides):
    row = {
        "Advertisers Name": "Vendor Co",
        "Slug": "vendor-co",
        "Active in website": "true",
        "Updated On": "Tue Apr 05 2022 19:07:37 GMT+0000 (Coordinated Universal Time)",
        "google ad link": "",
        "Client banner Ad": "https://cdn.example/banner.jpg",
        "Banner Ad link": UTM_LINK,
        "300x250 square image": "https://cdn.example/square.png",
        "300x250 square ad link": UTM_LINK,
        "Small Banner Image": "",
        "Small banner link": "",
        "Website Link": "https://vendor.example",
    }
    row.update(overrides)
    return row


def test_load_advertisers_every_row_regardless_of_active_flag(tmp_path):
    """Issue #50: the active-in-website filter is gone — the whole roster
    enters the pipeline (the classification gate still rules eligibility)."""
    csv_path = tmp_path / "ads.csv"
    csv_path.write_text(
        "Advertisers Name,Slug,Active in website\n"
        "Active Co,active-co,true\n"
        "Dead Co,dead-co,false\n"
        "Blank Co,blank-co,\n"
    )
    rows = load_advertisers(csv_path)
    assert [r["Slug"] for r in rows] == ["active-co", "dead-co", "blank-co"]


def test_canonical_website_fallback_chain():
    # website column wins when present
    assert canonical_website(make_row()) == "https://vendor.example"
    # else the google-ad column, HTML tags stripped
    row = make_row(**{"Website Link": "",
                      "google ad link": '<p id="">https://ford.example/</p><p>x</p>'})
    assert canonical_website(row) == "https://ford.example/"
    # else the origin (scheme + host) of any creative click-through link
    row = make_row(**{"Website Link": "", "google ad link": ""})
    assert canonical_website(row) == "https://vendor.example"
    # nothing derivable
    row = make_row(**{"Website Link": "", "google ad link": "",
                      "Banner Ad link": "", "300x250 square ad link": ""})
    assert canonical_website(row) == ""


NEWER = "Wed Mar 27 2024 10:00:00 GMT+0000 (Coordinated Universal Time)"


def test_dedupe_collapses_same_domain_keeping_newest():
    old = advertiser_record(make_row(
        **{"Advertisers Name": "Vendor Giveaway 2022", "Slug": "vendor-2022",
           "Client banner Ad": "https://cdn.example/old-banner.jpg"}))
    new = advertiser_record(make_row(
        **{"Advertisers Name": "Vendor Giveaway 2024", "Slug": "vendor-2024",
           "Updated On": NEWER,
           # same advertiser, different subpage — domain is the identity
           "Website Link": "https://www.vendor.example/tickets/2024"}))
    other = advertiser_record(make_row(
        **{"Advertisers Name": "Unrelated Co", "Slug": "unrelated-co",
           "Website Link": "https://other.example"}))
    deduped = dedupe_records([old, new, other])
    assert {r["slug"] for r in deduped} == {"vendor-2024", "unrelated-co"}
    survivor = next(r for r in deduped if r["slug"] == "vendor-2024")
    # merged rows keep their distinct creatives under the survivor
    assert "https://cdn.example/old-banner.jpg" in [
        c["image"] for c in survivor["creatives"]]


def test_dedupe_falls_back_to_normalized_name():
    a = advertiser_record(make_row(
        **{"Advertisers Name": "Dream Give-a-Way", "Slug": "a", "Website Link": "",
           "google ad link": "", "Banner Ad link": "", "300x250 square ad link": ""}))
    b = advertiser_record(make_row(
        **{"Advertisers Name": "dream give a way ", "Slug": "b", "Website Link": "",
           "google ad link": "", "Banner Ad link": "", "300x250 square ad link": "",
           "Updated On": NEWER}))
    c = advertiser_record(make_row(
        **{"Advertisers Name": "Other Charity", "Slug": "c", "Website Link": "",
           "google ad link": "", "Banner Ad link": "", "300x250 square ad link": ""}))
    deduped = dedupe_records([a, b, c])
    assert {r["slug"] for r in deduped} == {"b", "c"}


def test_dedupe_never_merges_distinct_advertisers_on_a_link_shortener():
    a = advertiser_record(make_row(
        **{"Advertisers Name": "Dash Cam Co", "Slug": "dash-cam-co",
           "Website Link": "https://amzn.to/abc123"}))
    b = advertiser_record(make_row(
        **{"Advertisers Name": "Seat Cover Co", "Slug": "seat-cover-co",
           "Website Link": "https://amzn.to/xyz789"}))
    assert len(dedupe_records([a, b])) == 2


def test_advertiser_record_creatives_square_first_and_deduped():
    rec = advertiser_record(make_row(**{
        "Small Banner Image": "https://cdn.example/square.png",  # dup of square
        "Small banner link": "",
    }))
    assert rec["name"] == "Vendor Co" and rec["slug"] == "vendor-co"
    assert [c["image"] for c in rec["creatives"]] == [
        "https://cdn.example/square.png", "https://cdn.example/banner.jpg",
    ]
    assert rec["creatives"][0]["link"] == UTM_LINK  # UTM preserved verbatim


def test_advertiser_record_link_falls_back_to_website():
    rec = advertiser_record(make_row(**{"300x250 square ad link": ""}))
    assert rec["creatives"][0]["link"] == "https://vendor.example"


def test_vendor_products_become_recommendable_entries():
    rec = advertiser_record(make_row())
    analysis = {
        "classification": "product vendor",
        "description": "Parts vendor.",
        "products": [
            {"name": "TKX 5-Speed Transmission", "description": "A gearbox.",
             "categories": ["transmission"], "keywords": ["tkx"], "aliases": ["tkx"]},
            {"name": "Magnum XL", "description": "Another gearbox.",
             "categories": ["transmission"], "keywords": [], "aliases": []},
        ],
    }
    entries = catalog_entries(rec, analysis)
    assert len(entries) == 3  # two products + the advertiser-level entry
    e = entries[0]
    assert e["id"] == "vendor-co-tkx-5-speed-transmission"
    assert e["recommendable"] is True and e["sponsored"] is True
    assert e["advertiser"] == "Vendor Co"
    assert e["image"] == "https://cdn.example/square.png"
    assert e["link"] == UTM_LINK
    assert e["deltas"] is None  # filled by the impure delta step
    assert "kind" not in e  # product entries keep their shape


def test_vendor_gains_advertiser_level_entry():
    """Issue #50: every vendor/service advertiser gets one advertiser-level
    Sponsored entry linking to its canonical website — recommendable in chat,
    never Upgrade Shop-eligible."""
    rec = advertiser_record(make_row())
    analysis = {
        "classification": "product vendor",
        "description": "Parts vendor.",
        "products": [
            {"name": "TKX", "description": "x", "categories": ["transmission"],
             "keywords": [], "aliases": []},
            {"name": "Headers", "description": "x", "categories": ["exhaust"],
             "keywords": [], "aliases": []},
        ],
    }
    adv = next(e for e in catalog_entries(rec, analysis)
               if e.get("kind") == "advertiser")
    assert adv["name"] == "Vendor Co" and adv["advertiser"] == "Vendor Co"
    assert adv["recommendable"] is True and adv["sponsored"] is True
    assert adv["specific"] is False  # not a concrete installable product
    assert adv["link"] == "https://vendor.example"  # the canonical website
    assert set(adv["categories"]) == {"transmission", "exhaust"}  # the union
    assert adv["deltas"] == zero_deltas()  # an advertiser isn't an upgrade
    assert "Vendor Co" in embed_text(adv) and "Parts vendor." in embed_text(adv)


def test_non_vendor_ingested_but_never_recommendable():
    rec = advertiser_record(make_row())
    for cls in ("placeholder", "charity", "giveaway", "event"):
        (entry,) = catalog_entries(rec, {"classification": cls,
                                         "description": "x", "products": []})
        assert entry["recommendable"] is False and entry["sponsored"] is True
        assert entry["deltas"] == zero_deltas()  # all-zero, all generations
        assert entry.get("kind") != "advertiser"  # no advertiser-level entry
    assert ELIGIBLE_CLASSIFICATIONS == {"product vendor", "service"}


def test_generic_entries_unbranded_no_links():
    entries = generic_entries()
    names = {e["name"].lower() for e in entries}
    for expected in ("supercharger", "cat-back exhaust", "big brake kit",
                     "performance transmission", "cold air intake"):
        assert expected in names
    for e in entries:
        assert e["advertiser"] is None and e["image"] is None and e["link"] is None
        assert e["sponsored"] is False and e["recommendable"] is False
        assert e["aliases"]


def test_normalize_deltas_fills_and_coerces():
    raw = {"S550": {"power": "12", "handling": None, "junk": 9}}
    out = normalize_deltas(raw)
    assert set(out) == set(GENERATION_NAMES)
    assert out["S550"]["power"] == 12
    assert out["S550"]["handling"] == 0 and "junk" not in out["S550"]
    assert out["Fox-body"] == {s: 0 for s in STATS}


def test_generation_names_match_app():
    from app import GENERATIONS  # deltas must be keyed by the app's names
    assert tuple(g for _lo, _hi, g in GENERATIONS) == GENERATION_NAMES


def test_embed_text_carries_searchable_fields():
    entry, _adv = catalog_entries(
        advertiser_record(make_row()),
        {"classification": "product vendor", "description": "d",
         "products": [{"name": "TKX", "description": "A 5-speed gearbox.",
                       "categories": ["transmission"], "keywords": ["manual swap"],
                       "aliases": []}]},
    )
    text = embed_text(entry)
    for needle in ("TKX", "Vendor Co", "gearbox", "transmission", "manual swap"):
        assert needle in text


def test_specific_flag_gates_shop_eligibility():
    """Issue #27: only concrete installable products from product vendors are
    specific (Upgrade Shop-eligible); services and broad product lines stay
    chat-recommendable only."""
    rec = advertiser_record(make_row())
    products = [
        {"name": "S1 4K Dash Cam", "specific": True, "description": "x",
         "categories": ["electronics"], "keywords": [], "aliases": []},
        {"name": "Restoration Parts Catalog", "specific": False, "description": "x",
         "categories": ["restoration parts"], "keywords": [], "aliases": []},
    ]
    cam, line, adv = catalog_entries(rec, {"classification": "product vendor",
                                           "description": "d", "products": products})
    assert cam["specific"] is True and cam["recommendable"] is True
    assert line["specific"] is False and line["recommendable"] is True
    assert adv["kind"] == "advertiser" and adv["specific"] is False
    # service products are never shop-eligible, even if flagged specific
    tour, _adv = catalog_entries(rec, {"classification": "service", "description": "d",
                                       "products": [dict(products[0], name="Route 66 Tour")]})
    assert tour["recommendable"] is True and tour["specific"] is False
    # non-vendor campaigns aren't specific; generics always are
    (camp,) = catalog_entries(rec, {"classification": "giveaway",
                                    "description": "x", "products": []})
    assert camp["specific"] is False
    assert all(g["specific"] for g in generic_entries())


def test_stat_set_is_nine_grouped():
    from ingest_ads import OWNERSHIP_STATS, PERFORMANCE_STATS
    assert STATS == PERFORMANCE_STATS + OWNERSHIP_STATS
    assert OWNERSHIP_STATS == ("style", "comfort", "safety", "reliability")
    assert zero_deltas()["S550"] == {s: 0 for s in STATS}
