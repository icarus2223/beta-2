"""Order execution. The loop decides; a broker carries it out and reports the real fill.

    PaperBroker      — simulated fills at the BTC/USDT price (default)
    Trading212Broker — real market orders on a Trading 212 Invest account (live only when both
                       HERMES_TRADING_MODE=live and HERMES_TRADING_I_ACCEPT_RISK=true)
"""
from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import httpx

log = logging.getLogger("broker")


class BrokerError(RuntimeError):
    """An order could not be placed or did not fill. Never retried automatically."""


@dataclass
class Fill:
    price: float      # per unit, in the instrument's currency
    quantity: float
    value: float      # price * quantity
    account_fraction: float = 0.0  # share of available cash this trade committed (for scoring)
    order_id: str | None = None


class PaperBroker:
    """Simulated fills. Buys pay half the spread above the price, sells receive half below it,
    so a round trip costs PAPER_SPREAD (default 0.2%, the assumed BTIC spread) like it would live."""

    name = "paper"

    def __init__(self) -> None:
        self.half_spread = float(os.environ.get("PAPER_SPREAD", "0.002")) / 2

    def market_open(self) -> bool:
        return True

    async def buy(self, fraction: float, ref_price: float) -> Fill:
        price = ref_price * (1 + self.half_spread)
        return Fill(price=price, quantity=fraction, value=price * fraction, account_fraction=fraction)

    async def sell(self, quantity: float, ref_price: float, stop_order_id: str | None = None) -> Fill:
        price = ref_price * (1 - self.half_spread)
        return Fill(price=price, quantity=quantity, value=price * quantity)

    async def mark_price(self, pos: dict, ref_price: float) -> float | None:
        return ref_price

    async def place_stop(self, quantity: float, entry_price: float, stop_pct: float) -> str | None:
        return None  # paper trades are only protected by the loop's own stop check

    async def stop_fill(self, order_id: str) -> Fill | None:
        return None


class Trading212Broker:
    """Market orders against https://live.trading212.com/api/v0 (spec: docs.trading212.com/api)."""

    name = "trading212"
    FILL_TIMEOUT_S = 60

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        env = os.environ
        self.ticker = env.get("T212_TICKER", "BTICd_EQ")
        self.max_order_value = float(env.get("T212_MAX_ORDER_EUR", "0"))
        self.qty_decimals = int(env.get("T212_QTY_DECIMALS", "2"))
        self.quote_symbol = env.get("T212_QUOTE_SYMBOL", "BTIC.SW")  # only used to estimate units per BTC
        self.tz = ZoneInfo(env.get("T212_MARKET_TZ", "Europe/Berlin"))
        open_s, close_s = env.get("T212_MARKET_HOURS", "09:05-17:25").split("-")
        self.open_t, self.close_t = dtime.fromisoformat(open_s), dtime.fromisoformat(close_s)

        key, secret = env.get("T212_API_KEY", ""), env.get("T212_API_SECRET", "")
        if not key:
            raise BrokerError("T212_API_KEY is empty in .env")
        if self.max_order_value <= 0:
            raise BrokerError("Set T212_MAX_ORDER_EUR in .env to the most one order may spend")
        auth = "Basic " + base64.b64encode(f"{key}:{secret}".encode()).decode() if secret else key
        base = env.get("T212_BASE_URL", "https://live.trading212.com")
        self.client = client or httpx.AsyncClient(base_url=base, timeout=15)
        self.client.headers["Authorization"] = auth
        self._units_per_btc: float | None = None

    # ---- market hours -------------------------------------------------------
    def market_open(self) -> bool:
        """Xetra-style weekday session, trimmed at both ends to avoid the open/close auctions."""
        now = datetime.now(self.tz)
        return now.weekday() < 5 and self.open_t <= now.time() <= self.close_t

    # ---- HTTP helpers -------------------------------------------------------
    async def _get(self, path: str, **params) -> httpx.Response:
        r = await self.client.get(f"/api/v0{path}", params=params or None)
        if r.status_code == 429:
            raise BrokerError(f"rate limited on {path}")
        return r

    async def _json(self, path: str, **params):
        r = await self._get(path, **params)
        if r.status_code >= 400:
            raise BrokerError(f"GET {path} → {r.status_code}: {r.text[:300]}")
        return r.json()

    async def available_cash(self) -> float:
        return float((await self._json("/equity/account/summary"))["cash"]["availableToTrade"])

    async def position(self) -> dict | None:
        for p in await self._json("/equity/positions"):
            if p["instrument"]["ticker"] == self.ticker:
                return p
        return None

    # ---- pricing ------------------------------------------------------------
    def _estimate_units_per_btc(self) -> float:
        """BTIC price / BTC price, from overlapping 15m bars. Drifts only by the fund fee, so cache it."""
        if self._units_per_btc is None:
            import yfinance as yf
            etp = yf.Ticker(self.quote_symbol).history(period="5d", interval="15m")["Close"]
            btc = yf.Ticker("BTC-USD").history(period="5d", interval="15m")["Close"]
            joined = etp.to_frame("etp").join(btc.to_frame("btc"), how="inner").dropna()
            if joined.empty:
                raise BrokerError(f"no overlapping quotes for {self.quote_symbol} to size the order")
            self._units_per_btc = statistics.median((joined.etp / joined.btc).tolist())
        return self._units_per_btc

    async def _estimate_price_eur(self) -> float:
        pos = await self.position()
        if pos:  # exact, from the broker
            return float(pos["currentPrice"])
        from .adapters.price import fetch_ticker_last
        btc_eur = await fetch_ticker_last("BTC/EUR")
        return await asyncio.to_thread(self._estimate_units_per_btc) * btc_eur

    # ---- orders -------------------------------------------------------------
    async def _market_order(self, quantity: float) -> Fill:
        # Not idempotent on Trading 212's side: one attempt only, never retry.
        r = await self.client.post("/api/v0/equity/orders/market",
                                   json={"ticker": self.ticker, "quantity": quantity, "extendedHours": False})
        if r.status_code >= 400:
            raise BrokerError(f"order rejected ({r.status_code}): {r.text[:300]}")
        order = r.json()
        order_id = order["id"]
        log.warning("LIVE ORDER %s %s x %s submitted (id %s)", "BUY" if quantity > 0 else "SELL",
                    self.ticker, abs(quantity), order_id)

        deadline = asyncio.get_running_loop().time() + self.FILL_TIMEOUT_S
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(2)
            r = await self._get(f"/equity/orders/{order_id}")
            if r.status_code == 404:  # no longer pending → filled, cancelled or rejected
                return await self._fill_from_history(order_id)
            status = r.json().get("status")
            if status in ("REJECTED", "CANCELLED"):
                raise BrokerError(f"order {order_id} {status}")
        await self.client.delete(f"/api/v0/equity/orders/{order_id}")
        raise BrokerError(f"order {order_id} not filled within {self.FILL_TIMEOUT_S}s; cancel requested")

    async def _fill_from_history(self, order_id: int) -> Fill:
        for attempt in range(5):  # history can lag the pending list by a moment
            page = await self._json("/equity/history/orders", ticker=self.ticker, limit=10)
            for item in page.get("items", []):
                if item["order"]["id"] == order_id:
                    if item["order"]["status"] != "FILLED" or not item.get("fill"):
                        raise BrokerError(f"order {order_id} ended {item['order']['status']}")
                    f = item["fill"]
                    qty = abs(float(f["quantity"]))
                    return Fill(price=float(f["price"]), quantity=qty, value=float(f["price"]) * qty,
                                order_id=str(order_id))
            await asyncio.sleep(3 * (attempt + 1))
        raise BrokerError(f"order {order_id} left the pending list but no fill found in history; check the app")

    async def buy(self, fraction: float, ref_price: float) -> Fill:
        cash = await self.available_cash()
        budget = min(cash * fraction, self.max_order_value)
        price = await self._estimate_price_eur()
        step = 10 ** -self.qty_decimals
        quantity = math.floor(budget * 0.97 / price / step) * step  # 3% headroom for quote drift
        quantity = round(quantity, self.qty_decimals)
        if quantity <= 0:
            raise BrokerError(f"budget €{budget:.2f} buys 0 units at ~€{price:.2f}")
        log.info("sizing: cash €%.2f × %.2f capped at €%.2f → %s units @ ~€%.2f",
                 cash, fraction, self.max_order_value, quantity, price)
        fill = await self._market_order(quantity)
        if fill.value > self.max_order_value * 1.05:
            log.error("fill value €%.2f exceeded the €%.2f cap", fill.value, self.max_order_value)
        fill.account_fraction = fill.value / cash
        return fill

    # ---- broker-side stop loss ---------------------------------------------
    async def place_stop(self, quantity: float, entry_price: float, stop_pct: float) -> str | None:
        """Sell-stop held by Trading 212, so the stop loss works even if this machine is asleep or offline.

        Returns the order id, or None if it could not be placed (the local stop still applies).
        """
        stop_price = round(entry_price * (1 - stop_pct / 100), 2)
        await asyncio.sleep(2)  # stop endpoint allows 1 req / 2s
        # Not idempotent: one attempt only.
        r = await self.client.post("/api/v0/equity/orders/stop", json={
            "ticker": self.ticker, "quantity": -quantity, "stopPrice": stop_price,
            "timeValidity": "GOOD_TILL_CANCEL"})
        if r.status_code >= 400:
            log.error("STOP ORDER REJECTED (%s): %s; position is protected only while the bot runs",
                      r.status_code, r.text[:300])
            return None
        order_id = str(r.json()["id"])
        log.warning("LIVE STOP placed: sell %s %s if price ≤ €%.2f (id %s)", quantity, self.ticker, stop_price, order_id)
        return order_id

    async def cancel_stop(self, order_id: str) -> None:
        """Cancel the broker stop and wait until it is no longer pending."""
        r = await self.client.delete(f"/api/v0/equity/orders/{order_id}")
        if r.status_code >= 400 and r.status_code != 404:  # 404: already gone (filled or cancelled)
            raise BrokerError(f"could not cancel stop {order_id} ({r.status_code}): {r.text[:200]}")
        for _ in range(10):
            g = await self._get(f"/equity/orders/{order_id}")
            if g.status_code == 404 or g.json().get("status") == "CANCELLED":
                log.info("stop %s cancelled", order_id)
                return
            await asyncio.sleep(1)
        raise BrokerError(f"stop {order_id} still pending after cancel; not selling to avoid a double sell")

    async def stop_fill(self, order_id: str) -> Fill | None:
        """The fill of a broker stop that has triggered, or None if it never filled."""
        page = await self._json("/equity/history/orders", ticker=self.ticker, limit=20)
        for item in page.get("items", []):
            if str(item["order"]["id"]) == str(order_id) and item["order"]["status"] == "FILLED" and item.get("fill"):
                f = item["fill"]
                qty = abs(float(f["quantity"]))
                return Fill(price=float(f["price"]), quantity=qty, value=float(f["price"]) * qty, order_id=str(order_id))
        return None

    async def sell(self, quantity: float, ref_price: float, stop_order_id: str | None = None) -> Fill:
        if stop_order_id:
            await self.cancel_stop(stop_order_id)  # otherwise the stop and this sell could both fill
        pos = await self.position()
        if pos is None:
            raise BrokerError(f"no {self.ticker} position to sell (the broker stop may have just filled)")
        quantity = min(quantity, float(pos["quantityAvailableForTrading"]))  # never sell shares the bot didn't buy
        return await self._market_order(-quantity)

    async def mark_price(self, pos: dict, ref_price: float) -> float | None:
        """Current fund price, or None if the position vanished (broker stop filled, or sold by hand)."""
        live = await self.position()
        return float(live["currentPrice"]) if live else None


def make_broker() -> PaperBroker | Trading212Broker:
    mode = os.environ.get("HERMES_TRADING_MODE", "paper")
    if mode == "paper":
        return PaperBroker()
    if mode != "live":
        raise SystemExit(f"Unknown HERMES_TRADING_MODE={mode!r}")
    if os.environ.get("HERMES_TRADING_I_ACCEPT_RISK", "").lower() != "true":
        raise SystemExit("HERMES_TRADING_MODE=live but HERMES_TRADING_I_ACCEPT_RISK is not true. Refusing to start.")
    broker = os.environ.get("BROKER", "trading212")
    if broker != "trading212":
        raise SystemExit(f"Unknown BROKER={broker!r}")
    return Trading212Broker()
