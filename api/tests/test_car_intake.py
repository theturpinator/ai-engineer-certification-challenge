"""POST /garage/{user_id}/cars (picker intake) — real Postgres, no LLM:
enrichment is monkeypatched out (its real behavior is covered by
test_portrait_stability.py); this checks validation, creation, generation
derivation, persistence, and the duplicate 409."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

import app as app_module
from app import app


@pytest.mark.asyncio
async def test_post_car_validation_creation_and_409(monkeypatch):
    enriched = []

    async def fake_enrich(uid):
        enriched.append(uid)

    monkeypatch.setattr(app_module, "_enrich_garage", fake_enrich)
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # validation: all three identity fields required, year bounded
            for bad in [{},
                        {"year": 1969, "trim": "Mach 1"},
                        {"year": 1969, "color": "blue"},
                        {"trim": "GT", "color": "blue"},
                        {"year": 1810, "trim": "GT", "color": "blue"},
                        {"year": 2050, "trim": "GT", "color": "blue"},
                        {"year": 1969, "trim": "  ", "color": "blue"}]:
                r = await client.post(f"/garage/{user_id}/cars", json=bad)
                assert r.status_code == 422, bad

            r = await client.post(
                f"/garage/{user_id}/cars",
                json={"year": 1969, "trim": "Mach 1", "color": "Grabber Blue",
                      "nickname": "Sally"},
            )
            assert r.status_code == 201, r.text
            car = r.json()
            assert car["id"]
            assert car["year"] == 1969 and car["trim"] == "Mach 1"
            assert car["color"] == "Grabber Blue" and car["nickname"] == "Sally"
            assert car["generation"] == "First generation"  # derived from year
            assert user_id in enriched  # background enrichment kicked off

            # persisted and visible via GET /garage
            cars = (await client.get(f"/garage/{user_id}")).json()["profile"]["cars"]
            assert [c["id"] for c in cars] == [car["id"]]

            # obviously-identical car (same year+trim, case-insensitive) -> 409
            r = await client.post(
                f"/garage/{user_id}/cars",
                json={"year": 1969, "trim": "mach 1", "color": "Red"},
            )
            assert r.status_code == 409

            # a different trim same year is a genuinely different car
            r = await client.post(
                f"/garage/{user_id}/cars",
                json={"year": 1969, "trim": "Boss 429", "color": "Black"},
            )
            assert r.status_code == 201
            assert r.json()["generation"] == "First generation"
