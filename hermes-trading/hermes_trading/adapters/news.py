"""Market sentiment: Crypto Fear & Greed index from alternative.me (free, public)."""
from __future__ import annotations

import httpx

SCHEMA_VERSION = 1


async def fetch() -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        data = (await client.get("https://api.alternative.me/fng/?limit=1")).json()["data"][0]
    return {
        "schema_version": SCHEMA_VERSION,
        "fear_greed": int(data["value"]),
        "label": data["value_classification"],
    }
