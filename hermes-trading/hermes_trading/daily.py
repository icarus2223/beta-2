"""Daily-trend context: where yesterday's daily close sits relative to its moving average.

Uses completed daily candles only (today's bar is still forming), cached for an hour.
"""
from __future__ import annotations

import logging
import time

from .adapters.price import exchange_ids, get_exchange

log = logging.getLogger("daily")
_CACHE: dict = {}
CACHE_S = 3600


async def daily_context(asset: str, sma: int) -> dict:
    key = (asset, sma)
    hit = _CACHE.get(key)
    if hit and time.time() - hit["at"] < CACHE_S:
        return hit["data"]
    last_exc: Exception | None = None
    for exchange_id in exchange_ids():
        try:
            rows = await get_exchange(exchange_id).fetch_ohlcv(asset, "1d", limit=sma + 2)
            break
        except Exception as exc:
            last_exc = exc
            log.warning("%s daily candles failed: %s", exchange_id, exc)
    else:
        raise RuntimeError("no exchange returned daily candles") from last_exc
    closes = [r[4] for r in rows[:-1]]  # drop today's forming bar
    if len(closes) < sma:
        raise RuntimeError(f"only {len(closes)} daily candles, need {sma}")
    avg = sum(closes[-sma:]) / sma
    data = {"close": closes[-1], "sma": round(avg, 2), "vs_sma": closes[-1] / avg - 1}
    _CACHE[key] = {"at": time.time(), "data": data}
    return data


def passes(ctx: dict, f: dict) -> bool:
    return f["min_pct"] / 100 <= ctx["vs_sma"] < f["max_pct"] / 100
