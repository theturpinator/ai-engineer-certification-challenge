"""Seam-1: portrait + stats stability — real Postgres, real gpt-image-1 and
Sonnet via the gateway (one generation each; the mod/edit trigger logic is
covered by the pure fingerprint tests in test_garage_model.py).

Seed a car directly in the garage table, let GET /garage's opportunistic
enrichment fill in the portrait and stats, then assert repeated reads never
rewrite either: image bytes identical, stats identical, fingerprint present.
"""

import asyncio
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from psycopg.types.json import Json

import app as app_module
from app import _stats_fp, app


@pytest.mark.asyncio
async def test_portrait_and_stats_stable_across_visits():
    user_id = str(uuid.uuid4())
    car = {"id": "car1", "year": 2016, "generation": "S550", "trim": "GT",
           "color": "red"}
    async with app.router.lifespan_context(app):
        async with await app_module._db() as conn:
            await conn.execute(
                "INSERT INTO garage (user_id, profile) VALUES (%s, %s)",
                (user_id, Json({"cars": [car]})),
            )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # first visit fires background enrichment (stats + portrait)
            assert (await client.get(f"/garage/{user_id}")).status_code == 200

            img = None
            for _ in range(60):  # gpt-image-1 takes ~10-30s
                r = await client.get(f"/garage/{user_id}/cars/car1/image")
                if r.status_code == 200:
                    img = r.content
                    break
                await asyncio.sleep(2)
            assert img, "portrait not generated within ~2min"

            g1 = (await client.get(f"/garage/{user_id}")).json()
            stats = g1["profile"]["cars"][0]["stats"]
            assert stats and stats["fp"] == _stats_fp(car), stats
            assert "modified" not in stats  # stats are always the stock baseline

            # repeated visits: enrichment must no-op, nothing rewritten
            await asyncio.sleep(2)  # let the re-fired background task settle
            g2 = (await client.get(f"/garage/{user_id}")).json()
            assert g2["profile"]["cars"][0]["stats"] == stats
            r1 = await client.get(f"/garage/{user_id}/cars/car1/image")
            r2 = await client.get(f"/garage/{user_id}/cars/car1/image")
            assert r1.content == img and r2.content == img  # byte-identical
