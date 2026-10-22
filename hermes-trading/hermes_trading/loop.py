"""The 24/7 reliability loop: fetch → evaluate strategy → paper trade → log → heartbeat."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from functools import partial
from zoneinfo import ZoneInfo

import numpy as np

from . import state
from .adapters import SchemaError, fetch_with_retry
from .broker import BrokerError, make_broker
from .daily import daily_context, passes as daily_passes
from .patterns import bullish_signal
from .adapters import macro, news, onchain, price

log = logging.getLogger("loop")
CIRCUIT_BREAK_AFTER = 5
RSI_WINDOW = 100  # RSI always uses the latest 100 candles, whatever else is fetched
BREAKER_COOLDOWN_S = 300


def rsi(closes: list[float], period: int = 14) -> float:
    deltas = np.diff(np.asarray(closes, dtype=float))
    gains, losses = np.clip(deltas, 0, None), np.clip(-deltas, 0, None)
    avg_gain, avg_loss = gains[:period].mean(), losses[:period].mean()
    for g, l in zip(gains[period:], losses[period:]):  # Wilder smoothing
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0
    return float(100 - 100 / (1 + avg_gain / avg_loss))


def ema(closes: list[float], period: int) -> float:
    """Exponential moving average of the closes, seeded with a simple average of the first `period`."""
    values = np.asarray(closes, dtype=float)
    alpha = 2 / (period + 1)
    avg = values[:period].mean()
    for v in values[period:]:
        avg = alpha * v + (1 - alpha) * avg
    return float(avg)


def trend(closes: list[float], strategy: dict) -> dict | None:
    """EMA trend filter state, or None when the filter is off."""
    f = strategy.get("trend_filter") or {}
    if not f.get("enabled"):
        return None
    fast, slow = ema(closes, int(f["fast"])), ema(closes, int(f["slow"]))
    return {"ema_fast": round(fast, 2), "ema_slow": round(slow, 2), "up": fast > slow}


def candles_needed(strategy: dict) -> int:
    """Enough history for RSI (also over past closed candles) and for the slow EMA to settle."""
    base = RSI_WINDOW + 10
    f = strategy.get("trend_filter") or {}
    return max(base, int(f.get("slow", 0)) + 50) if f.get("enabled") else base


def entry_signal(price_data: dict, strategy: dict, r: float) -> str | None:
    """Why to buy now, or None. Mirrors backtest.simulate() so live and backtest agree.

    Without confirmation: RSI (on the forming candle) below the threshold.
    With confirmation: RSI dipped below the threshold on one of the last `dip_lookback`+1 closed
    candles, and a listed bullish pattern just completed on the last closed candles.
    """
    if strategy["entry"]["indicator"] != "rsi":
        return None
    threshold = strategy["entry"]["threshold"]
    conf = strategy.get("confirm") or {}
    if not conf.get("enabled"):
        return f"rsi {r:.1f}" if r < threshold else None

    closed = price_data["candles"][:-1]  # the last candle is still forming
    closes = [k["c"] for k in closed]
    lookback = int(conf.get("dip_lookback", 4))
    dip = min(rsi(closes[len(closes) - j - RSI_WINDOW:len(closes) - j]) for j in range(lookback + 1))
    if dip >= threshold:
        return None
    name = bullish_signal(closed[-3:], conf["patterns"])
    return f"{name} after RSI dip to {dip:.1f}" if name else None


async def gather_data(asset: str, timeframe: str, limit: int = 100) -> dict:
    """Price is mandatory; the other adapters are context and may be missing."""
    data = {"price": await fetch_with_retry("price", partial(price.fetch, asset, timeframe, limit),
                                            price.SCHEMA_VERSION)}
    for name, mod in (("onchain", onchain), ("news", news), ("macro", macro)):
        try:
            data[name] = await fetch_with_retry(name, mod.fetch, mod.SCHEMA_VERSION)
        except SchemaError:
            raise
        except Exception as exc:
            log.warning("%s unavailable this tick: %s", name, exc)
            data[name] = None
    return data


async def step(asset: str, data: dict, strategy: dict, broker) -> str | None:
    """Evaluate one tick against the strategy. Signals come from BTC; prices and P&L come from the broker's fills.

    Returns 'open', 'close' or None.
    """
    btc = data["price"]["last"]
    r = rsi(data["price"]["closes"][-RSI_WINDOW:])
    pos = state.read_json(state.POSITION)
    context = {k: data[k] for k in ("onchain", "news", "macro")}

    if not broker.market_open():
        return None  # never queue an order for the next session; the signal may be gone by then

    if pos is None:
        tr = trend(data["price"]["closes"], strategy)
        signal = entry_signal(data["price"], strategy, r)
        if signal:
            if tr is not None and not tr["up"]:
                log.info("RSI %.1f says buy, but trend is down (EMA %s < %s): skipping", r,
                         tr["ema_fast"], tr["ema_slow"])
                return None
            df = strategy.get("daily_filter") or {}
            day = None
            if df.get("enabled"):
                try:
                    day = await daily_context(asset, int(df["sma"]))
                except Exception as exc:
                    log.warning("%s, but daily data unavailable (%s): skipping, not buying blind", signal, exc)
                    return None
                if not daily_passes(day, df):
                    log.info("%s, but daily close is %+.2f%% vs SMA%s (allowed %s%%..%s%%): skipping", signal,
                             day["vs_sma"] * 100, df["sma"], df["min_pct"], df["max_pct"])
                    return None
            try:
                fill = await broker.buy(strategy["position_size_r"], btc)
            except BrokerError as exc:
                log.error("BUY failed, staying flat: %s", exc)
                return None
            pos = {"asset": asset, "broker": broker.name, "entry_price": fill.price, "quantity": fill.quantity,
                   "entry_value": fill.value, "size": fill.account_fraction, "order_id": fill.order_id,
                   "entry_btc": btc, "entry_ts": state.now_iso(), "rsi": r, "strategy_version": strategy["version"],
                   "trend": tr, "signal": signal, "daily": day, "stop_order_id": None}
            state.write_json(state.POSITION, pos)  # save before placing the stop: the buy is real either way
            pos["stop_order_id"] = await broker.place_stop(fill.quantity, fill.price, strategy["stop_loss_pct"])
            state.write_json(state.POSITION, pos)
            state.append_jsonl(state.TRADES, {"event": "open", "ts": pos["entry_ts"], **pos, "context": context})
            log.info("%s OPEN  %s @ %.4f x %s (%s, strategy v%s)", broker.name.upper(), asset,
                     fill.price, fill.quantity, signal, strategy["version"])
            return "open"
        return None

    mark = await broker.mark_price(pos, btc)
    if mark is None:
        stop_fill = await broker.stop_fill(pos["stop_order_id"]) if pos.get("stop_order_id") else None
        if stop_fill:  # the broker-side stop loss fired: a real, scoreable trade
            return close_position(asset, pos, stop_fill, "broker_stop", broker, context)
        log.error("Position is gone at the broker (sold manually?). Dropping local position without scoring it.")
        state.append_jsonl(state.TRADES, {"event": "orphaned", "ts": state.now_iso(), **pos})
        state.POSITION.unlink(missing_ok=True)
        return None

    ret = mark / pos["entry_price"] - 1
    reason = None
    if ret <= -strategy["stop_loss_pct"] / 100:
        reason = "stop_loss"
    elif r > strategy.get("exit", {}).get("rsi_above", 55):
        reason = "rsi_exit"
    if reason is None:
        return None

    try:
        fill = await broker.sell(pos["quantity"], btc, pos.get("stop_order_id"))
    except BrokerError as exc:
        log.error("SELL failed, still holding: %s", exc)
        return None
    return close_position(asset, pos, fill, reason, broker, context)


def close_position(asset: str, pos: dict, fill, reason: str, broker, context: dict) -> str:
    ret = fill.price / pos["entry_price"] - 1
    record = {"event": "close", "ts": state.now_iso(), "asset": asset, "broker": broker.name,
              "entry_ts": pos["entry_ts"], "entry_price": pos["entry_price"], "exit_price": fill.price,
              "quantity": fill.quantity, "order_id": fill.order_id, "return_pct": ret * 100,
              "pnl_frac": ret * pos["size"], "reason": reason, "strategy_version": pos["strategy_version"],
              "context": context}
    state.append_jsonl(state.TRADES, record)
    state.POSITION.unlink(missing_ok=True)
    log.info("%s CLOSE %s @ %.4f  %+.2f%% (%s)", broker.name.upper(), asset, fill.price, ret * 100, reason)
    return "close"


async def run_loop(asset: str, interval: float = 60, once: bool = False) -> None:
    broker = make_broker()  # exits unless paper, or live with both risk flags set
    if broker.name != "paper":
        log.warning("LIVE TRADING via %s on %s · max €%.2f per order", broker.name, broker.ticker,
                    broker.max_order_value)
    try:
        await _tick_forever(asset, interval, once, broker)
    finally:
        await price.close_all()  # exchange clients are long-lived; close them cleanly on exit


LEARN_STATE = state.STATE_DIR / "learn_state.json"
LEARN_AFTER = (18, 0)  # Berlin time, after the Xetra close
_learn_task: asyncio.Task | None = None


def _run_learner(mode: str) -> None:
    from .learn import learn
    try:
        learn(apply=(mode == "apply"))
    except Exception as exc:  # never let the learner take the trading loop down
        log.error("learner failed, strategy unchanged: %r", exc)


def maybe_learn(once: bool) -> None:
    """Once per day after the close, run the pattern learner in a background thread."""
    global _learn_task
    mode = os.environ.get("AUTO_LEARN", "propose")  # off | propose | apply (apply works in paper mode only)
    if once or mode == "off" or (_learn_task is not None and not _learn_task.done()):
        return
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    today = now.date().isoformat()
    if (now.hour, now.minute) < LEARN_AFTER or (state.read_json(LEARN_STATE, {}) or {}).get("last") == today:
        return
    state.write_json(LEARN_STATE, {"last": today, "mode": mode})
    log.info("running the pattern learner (%s) in the background", mode)
    _learn_task = asyncio.create_task(asyncio.to_thread(_run_learner, mode))


async def _tick_forever(asset: str, interval: float, once: bool, broker) -> None:
    goal = state.load_yaml(state.GOAL)
    failures = 0
    closes_since_reflect = 0
    while True:
        strategy = state.load_yaml(state.STRATEGY)  # re-read every tick so reflection edits apply live
        try:
            data = await gather_data(asset, strategy.get("timeframe", "15m"), candles_needed(strategy))
            event = await step(asset, data, strategy, broker)
            failures = 0
            if event == "close":
                closes_since_reflect += 1
                mode = os.environ.get("AUTO_REFLECT", "fallback")  # fallback | hermes | off
                if closes_since_reflect >= goal["reflection_every"] and mode != "off":
                    from .reflect import reflect_auto
                    try:  # in a thread: a Hermes call can take minutes and must not stall the ticks
                        await asyncio.to_thread(reflect_auto, mode)
                    except Exception as exc:
                        log.error("reflection failed, strategy unchanged: %s", exc)
                    closes_since_reflect = 0
            maybe_learn(once)
            tick_rsi = round(rsi(data["price"]["closes"][-RSI_WINDOW:]), 2)
            state.write_json(state.HEARTBEAT, {"ts": state.now_iso(), "ok": True, "last": data["price"]["last"],
                                               "rsi": tick_rsi, "strategy_version": strategy["version"]})
            pos = state.read_json(state.POSITION)
            tr = trend(data["price"]["closes"], strategy)
            log.info("tick · BTC %.2f · RSI %.1f (buy < %s) · %s · %s · v%s", data["price"]["last"], tick_rsi,
                     strategy["entry"]["threshold"],
                     "trend off" if tr is None else ("trend UP" if tr["up"] else "trend down"),
                     f"holding since {pos['entry_ts']}" if pos else "flat", strategy["version"])
        except SchemaError as exc:
            log.error("Schema mismatch, halting: %s", exc)
            raise
        except Exception as exc:
            failures += 1
            log.error("tick failed (%d/%d): %s", failures, CIRCUIT_BREAK_AFTER, exc)
            state.write_json(state.HEARTBEAT, {"ts": state.now_iso(), "ok": False, "error": str(exc)})
            if failures >= CIRCUIT_BREAK_AFTER:
                # Locally nothing restarts us (unlike Railway), so cool down instead of exiting.
                log.error("Circuit breaker tripped after %d consecutive failures; pausing %ds before retrying.",
                          failures, BREAKER_COOLDOWN_S)
                if once:
                    raise SystemExit(1)
                await asyncio.sleep(BREAKER_COOLDOWN_S)
                failures = 0
                continue
        if once:
            return
        await asyncio.sleep(interval)
