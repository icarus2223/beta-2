"""Walk-forward backtest of strategy combinations on BTC/USDT 15m candles, simulating the live setup.

    uv run python -m hermes_trading.backtest --days 90

Realism rules (matching the live bot):
- Acts only inside Xetra hours (Mon-Fri 09:05-17:25 Berlin); BTC moves overnight still count.
- Stop loss is a broker stop: inside hours it fills at the stop, or at the open if the price gapped through it.
- Entries fill at the close of the signal candle; every round trip pays SPREAD.
- Candle patterns are evaluated on closed candles only.
- Combinations are ranked on the first 2/3 of the data (train) and judged on the last 1/3 (test).
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import ccxt
import numpy as np

from .loop import ema, rsi
from .patterns import PATTERNS
from .score import metrics, score

TZ = ZoneInfo("Europe/Berlin")
SPREAD = 0.002          # assumed BTIC round-trip spread (0.1% per side)
SIZE = 0.5              # position_size_r
WARMUP = 250            # candles needed before the first decision
CACHE = Path(__file__).resolve().parent.parent / "data" / "btc_15m.json"

FAMILIES = {
    "none": [],
    "any_bullish": list(PATTERNS),
    "two_candle": ["bullish_engulfing", "piercing_line", "bullish_harami"],
    "wick": ["hammer", "dragonfly_doji"],
    "three_candle": ["morning_star", "three_inside_up", "three_outside_up", "three_white_soldiers"],
}


def load_candles(days: int) -> list[dict]:
    need = days * 96 + WARMUP
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if len(cached) >= need and (datetime.now().timestamp() * 1000 - cached[-1]["t"]) < 3600_000:
            return cached[-need:]
    ex = ccxt.binance({"options": {"fetchMarkets": ["spot"]}})
    since, rows = ex.milliseconds() - need * 15 * 60 * 1000, []
    while len(rows) < need:
        batch = ex.fetch_ohlcv("BTC/USDT", "15m", since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 1
    candles = [{"t": r[0], "o": r[1], "h": r[2], "l": r[3], "c": r[4]} for r in rows[:-1]]  # drop the forming candle
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(candles))
    return candles[-need:]


def precompute(candles: list[dict]) -> dict:
    closes = [k["c"] for k in candles]
    n = len(candles)
    out = {"rsi": np.full(n, np.nan), "trend_up": np.zeros(n, bool), "in_hours": np.zeros(n, bool),
           "pattern": {f: np.zeros(n, bool) for f in FAMILIES}, "ts": []}
    for i in range(n):
        close_time = datetime.fromtimestamp((candles[i]["t"] + 15 * 60 * 1000) / 1000, TZ)
        out["ts"].append(close_time.isoformat())
        out["in_hours"][i] = close_time.weekday() < 5 and (9, 5) <= (close_time.hour, close_time.minute) <= (17, 25)
        if i < WARMUP:
            continue
        out["rsi"][i] = rsi(closes[i - 99:i + 1])
        window = closes[i - WARMUP + 1:i + 1]
        out["trend_up"][i] = ema(window, 50) > ema(window, 200)
        for fam, names in FAMILIES.items():
            out["pattern"][fam][i] = any(PATTERNS[p](candles[i - 2:i + 1]) for p in names)
    return out


def simulate(candles, pre, lo, hi, p) -> list[dict]:
    trades, pos, recent_dip = [], None, -10**9
    for i in range(max(lo, WARMUP), hi):
        k, r = candles[i], pre["rsi"][i]
        if r < p["entry"]:
            recent_dip = i
        if not pre["in_hours"][i]:
            continue
        if pos is not None:
            stop = pos["entry"] * (1 - p["stop"] / 100)
            exit_px, reason = None, None
            if k["l"] <= stop:
                exit_px, reason = min(k["o"], stop), "broker_stop"  # gap through the stop fills at the open
            elif r > p["exit"]:
                exit_px, reason = k["c"], "rsi_exit"
            if exit_px is not None:
                ret = exit_px / pos["entry"] - 1 - SPREAD
                trades.append({"ts": pre["ts"][i], "return_pct": ret * 100, "pnl_frac": ret * SIZE, "reason": reason})
                pos = None
            continue
        dip_ok = (r < p["entry"]) if p["confirm"] == "none" else (i - recent_dip <= 4)
        if not dip_ok or (p["trend"] and not pre["trend_up"][i]):
            continue
        if p["confirm"] != "none" and not pre["pattern"][p["confirm"]][i]:
            continue
        pos = {"entry": k["c"]}
    return trades


def summarize(trades: list[dict], goal: dict) -> dict:
    m = metrics(trades)
    return {"trades": m["n"], "wins": sum(t["return_pct"] > 0 for t in trades), "total": m["return"],
            "max_dd": m["max_drawdown"], "score": score(trades, goal) if trades else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--spread", type=float, default=SPREAD, help="round-trip cost, e.g. 0.002 = 0.2%%")
    args = ap.parse_args()
    globals()["SPREAD"] = args.spread

    from . import state
    goal = state.load_yaml(state.GOAL)
    candles = load_candles(args.days)
    pre = precompute(candles)
    n = len(candles)
    split = WARMUP + (n - WARMUP) * 2 // 3
    print(f"{len(candles) - WARMUP} candles · BTC {candles[WARMUP]['c']:.0f} → {candles[-1]['c']:.0f} · "
          f"train {pre['ts'][WARMUP][:10]}…{pre['ts'][split][:10]} · test …{pre['ts'][-1][:10]} · spread {SPREAD:.1%}")

    grid = [dict(entry=e, exit=x, stop=s, trend=t, confirm=c) for e, x, s, t, c in itertools.product(
        (25, 30, 35), (50, 55, 65), (1.5, 2.0, 3.0), (False, True), FAMILIES)]
    rows = []
    for p in grid:
        tr, te = simulate(candles, pre, WARMUP, split, p), simulate(candles, pre, split, n, p)
        rows.append((p, summarize(tr, goal), summarize(te, goal)))

    def label(p):
        return (f"RSI<{p['entry']} exit>{p['exit']} stop{p['stop']}% "
                f"{'EMA50/200' if p['trend'] else 'no-trend'} {p['confirm']}")

    def line(p, a, b):
        return (f"{label(p):48} | train {a['trades']:3d}tr {a['total']:+7.2%} dd{a['max_dd']:5.1%} s{a['score']:+.2f}"
                f" | test {b['trades']:3d}tr {b['total']:+7.2%} dd{b['max_dd']:5.1%} s{b['score']:+.2f}")

    current = dict(entry=30, exit=55, stop=2.0, trend=True, confirm="none")
    print("\nCURRENT LIVE (v02):")
    print(line(current, *[summarize(simulate(candles, pre, a, b, current), goal) for a, b in ((WARMUP, split), (split, n))]))
    eligible = [r for r in rows if r[1]["trades"] >= 6]  # too few trades = luck, not evidence
    eligible.sort(key=lambda r: r[1]["score"], reverse=True)
    print(f"\nTOP {args.top} ON TRAIN (≥6 trades), with their unseen TEST result:")
    for p, a, b in eligible[:args.top]:
        print(line(p, a, b))
    out = Path(__file__).resolve().parent.parent / "data" / "backtest_results.json"
    out.write_text(json.dumps([{"params": p, "train": a, "test": b} for p, a, b in rows], indent=1))
    print(f"\nAll {len(rows)} combinations saved to {out.relative_to(out.parent.parent)}")


if __name__ == "__main__":
    main()
