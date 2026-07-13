"""Seam-1: live sponsor-site search + sponsor-forward answers (issue #51) —
real Postgres, real LLM where routing is the behavior, the live-search
provider (_tavily, the single network edge) stubbed with deterministic
payloads."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

import app as app_module
from app import CATALOG, app
from test_chat_api import collect_events

NPD_ENTRY = next(e for e in CATALOG if e.get("kind") == "advertiser"
                 and "npd" in e["advertiser"].lower())


@pytest.mark.asyncio
async def test_live_sponsor_results_stream_as_sponsored_cards(monkeypatch):
    """A niche-part question the catalog can't answer specifically routes to
    search_sponsor_sites; the live results stream as Sponsored cards (max
    three combined) wearing the sponsor's banner creative when the result has
    no image of its own."""
    calls = []

    async def fake_tavily(query, include_domains=None):
        calls.append({"query": query, "include_domains": include_domains})
        return [
            {"title": "1966 Mustang Taillight Bezel - Concours",
             "url": "https://www.npdlink.com/product/taillight-bezel-1966",
             "content": "Concours-correct taillight bezel for 1965-66 Mustangs."},
            {"title": "1966 Mustang Taillight Lens",
             "url": "https://www.npdlink.com/product/taillight-lens-1966",
             "content": "Show-quality replacement taillight lens."},
        ]

    monkeypatch.setattr(app_module, "_tavily", fake_tavily)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            events = await collect_events(
                client,
                "I'm restoring a 1966 Mustang and need a concours-correct "
                "taillight bezel. Where can I get one?",
                str(uuid.uuid4()),
            )
            tools = [e["name"] for e in events if e["type"] == "tool"]
            assert "search_sponsor_sites" in tools, tools
            # domain restriction: only sponsor sites are ever searched
            assert calls and all(c["include_domains"] for c in calls), calls

            ads = [e for e in events if e["type"] == "ad"]
            assert 1 <= len(ads) <= 3, ads
            live = [a for a in ads
                    if (a["link"] or "").startswith("https://www.npdlink.com/product/")]
            assert live, ads
            for ad in live:
                assert ad["sponsored"] is True
                assert ad["advertiser"] == NPD_ENTRY["advertiser"]
                # no image of its own -> the sponsor's banner creative
                assert ad["image"] == NPD_ENTRY["image"]


@pytest.mark.asyncio
async def test_upgrade_advice_cites_archive_articles():
    """Issue #51: upgrade-advice answers call the archive alongside product
    recommendation and cite at least one article inline."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            events = await collect_events(
                client,
                "What are good first mods for a 2016 Mustang GT?",
                str(uuid.uuid4()),
            )
            tools = [e["name"] for e in events if e["type"] == "tool"]
            assert "search_archive" in tools, tools
            assert "recommend_products" in tools, tools
            (citations_event,) = [e for e in events if e["type"] == "citations"]
            assert citations_event["citations"], "no archive citations"
            answer = "".join(e["text"] for e in events if e["type"] == "token")
            assert "mustangdriver.com" in answer.lower(), answer


@pytest.mark.asyncio
async def test_editorial_question_triggers_no_sponsor_content(monkeypatch):
    """History questions stay purely editorial: no sponsor tools, no ads."""

    async def fail_tavily(query, include_domains=None):
        raise AssertionError("live search must not run on editorial questions")

    monkeypatch.setattr(app_module, "_tavily", fail_tavily)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            events = await collect_events(
                client, "Tell me the story of the 1969 Boss 429", str(uuid.uuid4())
            )
            tools = [e["name"] for e in events if e["type"] == "tool"]
            assert "search_sponsor_sites" not in tools, tools
            assert "recommend_products" not in tools, tools
            assert [e for e in events if e["type"] == "ad"] == []
