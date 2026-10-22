"""OHLCV candles from a public ccxt exchange (no key needed), with fallback exchanges."""
from __future__ import annotations

import logging
import os

import ccxt.async_support as ccxt

SCHEMA_VERSION = 1
log = logging.getLogger("price")
_EXCHANGES: dict[str, ccxt.Exchange] = {}


def get_exchange(exchange_id: str) -> ccxt.Exchange:
    """One long-lived client per exchange, so the market list is downloaded once, not every tick."""
    if exchange_id not in _EXCHANGES:
        options = {"fetchMarkets": ["spot"]} if exchange_id == "binance" else {}  # skip futures market lists
        _EXCHANGES[exchange_id] = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 15000,
                                                              "options": options})
    return _EXCHANGES[exchange_id]


async def close_all() -> None:
    for exchange in _EXCHANGES.values():
        await exchange.close()
    _EXCHANGES.clear()


def exchange_ids() -> list[str]:
    primary = os.environ.get("EXCHANGE_ID", "binance")
    fallbacks = [e.strip() for e in os.environ.get("EXCHANGE_FALLBACKS", "kraken").split(",") if e.strip()]
    return [primary] + [e for e in fallbacks if e != primary]


async def fetch_ticker_last(symbol: str) -> float:
    last_exc: Exception | None = None
    for exchange_id in exchange_ids():
        try:
            return (await get_exchange(exchange_id).fetch_ticker(symbol))["last"]
        except Exception as exc:
            last_exc = exc
            log.warning("%s ticker %s failed: %s", exchange_id, symbol, exc)
    raise RuntimeError(f"no exchange returned {symbol}") from last_exc


async def fetch(asset: str = "BTC/USDT", timeframe: str = "15m", limit: int = 100) -> dict:
    last_exc: Exception | None = None
    for exchange_id in exchange_ids():
        try:
            candles = await get_exchange(exchange_id).fetch_ohlcv(asset, timeframe=timeframe, limit=limit)
            break
        except Exception as exc:
            last_exc = exc
            log.warning("%s candles failed, trying next exchange: %s", exchange_id, exc)
    else:
        raise RuntimeError("all exchanges failed") from last_exc
    closes = [c[4] for c in candles]
    return {
        "schema_version": SCHEMA_VERSION,
        "exchange": exchange_id,
        "asset": asset,
        "timeframe": timeframe,
        "closes": closes,
        # Full candles for pattern detection; the last one is still forming.
        "candles": [{"t": c[0], "o": c[1], "h": c[2], "l": c[3], "c": c[4]} for c in candles],
        "last": closes[-1],
        "ts": candles[-1][0],
    }
