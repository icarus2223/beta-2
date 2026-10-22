"""Data adapters: price · on-chain · news · macro.

Each adapter module exposes `async def fetch() -> dict` returning a payload with
a `schema_version` field. `fetch_with_retry` wraps them with retries and schema checks.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

log = logging.getLogger("adapters")


class SchemaError(RuntimeError):
    """An adapter returned data in a shape we don't understand. Halts the loop."""


async def fetch_with_retry(
    name: str,
    fn: Callable[[], Awaitable[dict[str, Any]]],
    expected_schema: int,
    attempts: int = 3,
    base_delay: float = 1.0,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            data = await fn()
            if data.get("schema_version") != expected_schema:
                raise SchemaError(f"{name}: expected schema {expected_schema}, got {data.get('schema_version')}")
            return data
        except SchemaError:
            raise  # never retry a contract break
        except Exception as exc:  # network, rate limit, etc.
            last_exc = exc
            delay = base_delay * (2 ** i)
            log.warning("%s fetch failed (%s), retry %d/%d in %.1fs", name, exc, i + 1, attempts, delay)
            await asyncio.sleep(delay)
    raise RuntimeError(f"{name}: all {attempts} attempts failed") from last_exc
