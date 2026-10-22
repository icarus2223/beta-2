"""Macro backdrop: US dollar index and S&P 500 via yfinance. Cached for an hour."""
from __future__ import annotations

import asyncio
import time

import yfinance as yf

SCHEMA_VERSION = 1
_CACHE: dict = {"at": 0.0, "data": None}


def _pull() -> dict:
    out = {}
    for key, ticker in (("dxy", "DX-Y.NYB"), ("spx", "^GSPC")):
        hist = yf.Ticker(ticker).history(period="5d")
        out[key] = float(hist["Close"].iloc[-1]) if not hist.empty else None
    return out


async def fetch() -> dict:
    if _CACHE["data"] is None or time.time() - _CACHE["at"] > 3600:
        _CACHE["data"] = await asyncio.to_thread(_pull)
        _CACHE["at"] = time.time()
    return {"schema_version": SCHEMA_VERSION, **_CACHE["data"]}
