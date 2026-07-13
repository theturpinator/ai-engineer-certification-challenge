"""Seam-1: advertiser products (issue #18) at the HTTP boundary — real
Postgres, real product index, real LLM where routing is the behavior.
Garage enrichment (stats LLM + portrait image gen) is monkeypatched behind
the same pure/impure split the other seam-1 suites use: the behaviors under
test here are ad-event routing/payloads and deterministic bar composition,
not the stock-baseline judgment (covered by test_portrait_stability)."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

import app as app_module
from app import BAR_STATS, CATALOG, app
from test_chat_api import collect_events


def _delta(entry_id: str, gen: str, stat: str) -> int:
    entry = next(e for e in CATALOG if e["id"] == entry_id)
    return entry["deltas"][gen][stat]


@pytest.mark.asyncio
async def test_product_intent_streams_sponsored_cards(monkeypatch):
    """A shopping question routes to recommend_products and streams at most
    three ad events, each carrying the full sponsored-card payload with delta
    chips resolved for the active car's generation."""

    async def no_enrich(uid):
        pass

    monkeypatch.setattr(app_module, "_enrich_garage", no_enrich)
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                f"/garage/{user_id}/cars",
                json={"year": 1990, "trim": "LX 5.0", "color": "red"},
            )
            assert r.status_code == 201  # Fox-body derived; chips resolve to it

            events = await collect_events(
                client,
                "I want to swap a modern manual transmission into my Fox-body. "
                "What should I buy?",
                user_id,
            )
            tools = [e["name"] for e in events if e["type"] == "tool"]
            assert "recommend_products" in tools, tools
            ads = [e for e in events if e["type"] == "ad"]
            assert 1 <= len(ads) <= 3, ads
            recommendable_names = {e["name"] for e in CATALOG if e["recommendable"]}
            for ad in ads:
                assert ad["sponsored"] is True
                assert ad["product"] and ad["advertiser"] and ad["description"]
                assert ad["image"].startswith("http")
                assert ad["link"].startswith("http")
                assert set(ad["deltas"]) == set(BAR_STATS)  # per-generation chips
                # only recommendation-eligible entries are ever recommended
                assert ad["product"] in recommendable_names, ad
            answer = "".join(e["text"] for e in events if e["type"] == "token")
            assert len(answer) > 50


@pytest.mark.asyncio
async def test_where_to_buy_streams_an_advertiser_website_card():
    """Issue #50: a where-to-buy question surfaces an advertiser-level
    Sponsored card linking the sponsor's website — the roster answer, not
    just individual ad-creative products — and never more than three cards."""
    advertiser_links = {e["link"] for e in CATALOG if e.get("kind") == "advertiser"}
    assert advertiser_links  # the regenerated artifact carries advertiser entries
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            events = await collect_events(
                client, "Where should I buy a cat-back exhaust for my Mustang?",
                str(uuid.uuid4()),
            )
            ads = [e for e in events if e["type"] == "ad"]
            assert 1 <= len(ads) <= 3, ads
            assert all(ad["sponsored"] is True for ad in ads)
            assert any(ad["link"] in advertiser_links for ad in ads), ads


@pytest.mark.asyncio
async def test_archive_question_streams_zero_ads():
    """Archive questions answer purely from articles with citations, exactly
    as before — no ad events, no recommendation tool."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            events = await collect_events(
                client, "Tell me about the 1971-1973 Mustangs", str(uuid.uuid4())
            )
            assert [e for e in events if e["type"] == "ad"] == []
            tools = [e["name"] for e in events if e["type"] == "tool"]
            assert "search_archive" in tools
            assert "recommend_products" not in tools, tools
            (citations_event,) = [e for e in events if e["type"] == "citations"]
            assert citations_event["citations"]


STOCK = {"power": 50, "acceleration": 50, "top_speed": 50,
         "handling": 50, "braking": 50, "style": 50, "comfort": 50,
         "safety": 50, "reliability": 50, "hp": 225, "zero_to_sixty": 6.2,
         "top_speed_mph": 140}


@pytest.mark.asyncio
async def test_shop_and_bars_round_trip(monkeypatch):
    """Have-it / want-it through the existing garage endpoints: the Upgrade
    Shop lists per-car rows, PATCH writes land in mods/wishlist, and the
    returned bars reflect stock + installed = current, current + wishlist =
    dream, straight from the committed catalog deltas."""

    async def fake_stats(car):
        return {**STOCK, "fp": app_module._stats_fp(car)}

    async def no_portrait(uid, car):
        pass

    monkeypatch.setattr(app_module, "_generate_stats", fake_stats)
    monkeypatch.setattr(app_module, "_sync_portrait", no_portrait)
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                f"/garage/{user_id}/cars",
                json={"year": 1990, "trim": "LX 5.0", "color": "red"},
            )
            car_id = r.json()["id"]

            # enrichment (faked stats) ran as the POST's background task
            (car,) = (await client.get(f"/garage/{user_id}")).json()["profile"]["cars"]
            assert car["stats"]["power"] == 50
            assert car["bars"]["current"] == {s: 50 for s in BAR_STATS}
            assert car["bars"]["dream"] == car["bars"]["current"]

            # the shop: recommended strip is eligible sponsor products only;
            # the catalog mixes sponsored rows with unbranded generic ones
            shop = (await client.get(f"/garage/{user_id}/cars/{car_id}/shop")).json()
            assert 2 <= len(shop["recommended"]) <= 3
            for rec in shop["recommended"]:
                assert rec["sponsored"] and rec["advertiser"] and rec["link"]
            rows = {row["id"]: row for row in shop["catalog"]}
            assert "generic-supercharger" in rows
            assert rows["generic-supercharger"]["advertiser"] is None
            assert rows["generic-supercharger"]["link"] is None
            assert set(rows["generic-supercharger"]["deltas"]) == set(BAR_STATS)
            campaign_ids = {e["id"] for e in CATALOG
                            if e["sponsored"] and not e["recommendable"]}
            assert not campaign_ids & set(rows)  # no placeholder/charity/giveaway

            # "I have this" -> installed mods move the current bar
            r = await client.patch(
                f"/garage/{user_id}/cars/{car_id}",
                json={"mods": ["supercharger"]},
            )
            bars = r.json()["bars"]
            gain = _delta("generic-supercharger", "Fox-body", "power")
            assert gain > 0
            assert bars["current"]["power"] == min(100, 50 + gain)
            assert bars["dream"] == bars["current"]  # empty wishlist

            # "Add to wishlist" -> the dream extension, current untouched
            r = await client.patch(
                f"/garage/{user_id}/cars/{car_id}",
                json={"wishlist": ["big brake kit"]},
            )
            bars = r.json()["bars"]
            brake_gain = _delta("generic-big-brake-kit", "Fox-body", "braking")
            assert brake_gain > 0
            assert bars["current"]["power"] == min(100, 50 + gain)
            assert bars["current"]["braking"] == 50
            assert bars["dream"]["braking"] == 50 + brake_gain

            # unrecognized free text contributes zero, is still listed
            r = await client.patch(
                f"/garage/{user_id}/cars/{car_id}",
                json={"mods": ["supercharger", "lucky dice on the mirror"]},
            )
            assert r.json()["bars"]["current"]["power"] == min(100, 50 + gain)

            # the shop reflects ownership state after the writes
            shop = (await client.get(f"/garage/{user_id}/cars/{car_id}/shop")).json()
            rows = {row["id"]: row for row in shop["catalog"]}
            assert rows["generic-supercharger"]["installed"] is True
            assert rows["generic-big-brake-kit"]["wishlisted"] is True
            assert all(not rec["installed"] and not rec["wishlisted"]
                       for rec in shop["recommended"])
