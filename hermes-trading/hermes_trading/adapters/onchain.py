"""Bitcoin network activity from mempool.space (free, public)."""
from __future__ import annotations

import httpx

SCHEMA_VERSION = 1


async def fetch() -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        fees = (await client.get("https://mempool.space/api/v1/fees/recommended")).json()
        mempool = (await client.get("https://mempool.space/api/mempool")).json()
    return {
        "schema_version": SCHEMA_VERSION,
        "fastest_fee": fees["fastestFee"],
        "mempool_tx_count": mempool["count"],
    }
