"""Pattern learner: find ONE change to strategy.yaml that would have worked over the last year.

    uv run python -m hermes_trading.learn            # report + proposal, changes nothing
    uv run python -m hermes_trading.learn --apply    # also apply the winner (paper mode only)

Method (the same checks done by hand in Oct 2026):
- Replay the CURRENT strategy over 365 days of 15m BTC candles, Xetra hours, 0.2% costs.
- Try one-variable changes (scientific method: one change per experiment).
- Score each per quarter, with double costs, and next to its neighbouring values.
- accept_candidate() decides what counts as learned; the best accepted candidate is proposed.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import json
import os
import time
from pathlib import Path

import ccxt
import numpy as np

from . import backtest as B
from . import state

SMAS = (20, 50, 100, 200)
MIN_TRADES = 30    # fewer trades in a year is too little evidence
MIN_GAIN = 0.01    # must beat the current strategy by at least 1 percentage point over 365 days
DAILY_CACHE = B.CACHE.parent / "btc_1d.json"
REPORT = B.CACHE.parent / "learn_report.json"
PROPOSALS = state.STATE_DIR / "proposals.jsonl"


# ---- data -------------------------------------------------------------------
def load_daily() -> list[dict]:
    if DAILY_CACHE.exists() and time.time() - DAILY_CACHE.stat().st_mtime < 6 * 3600:
        return json.loads(DAILY_CACHE.read_text())
    rows = ccxt.binance({"options": {"fetchMarkets": ["spot"]}}).fetch_ohlcv("BTC/USDT", "1d", limit=1000)
    days = [{"t": r[0], "c": r[4]} for r in rows[:-1]]  # drop today's forming bar
    closes = [k["c"] for k in days]
    for i, k in enumerate(days):
        for n in SMAS:
            k[f"sma{n}"] = float(np.mean(closes[i - n + 1:i + 1])) if i >= n - 1 else None
    DAILY_CACHE.parent.mkdir(exist_ok=True)
    DAILY_CACHE.write_text(json.dumps(days))
    return days


def prepare(days_back: int = 365) -> dict:
    candles = B.load_candles(days_back)
    pre = B.precompute(candles)
    days = load_daily()
    ends = [k["t"] + 86_400_000 for k in days]
    # index of the last fully closed day at the close of each 15m candle
    pre["day"] = [bisect.bisect_right(ends, k["t"] + 900_000) - 1 for k in candles]
    n = len(candles)
    quarters = [B.WARMUP + (n - B.WARMUP) * j // 4 for j in range(5)]
    return {"candles": candles, "pre": pre, "days": days, "quarters": quarters}


# ---- replay any strategy.yaml ------------------------------------------------
def family_of(patterns: list[str]) -> str:
    for name, members in B.FAMILIES.items():
        if sorted(members) == sorted(patterns):
            return name
    raise ValueError(f"pattern list {patterns} is not one of the backtest families")


def simulate(data: dict, strat: dict, lo: int, hi: int, spread: float = 0.002) -> list[float]:
    """Account-level P&L per closed trade, mirroring loop.step() decisions."""
    candles, pre, days = data["candles"], data["pre"], data["days"]
    entry, exit_rsi = strat["entry"]["threshold"], strat["exit"]["rsi_above"]
    stop, size = strat["stop_loss_pct"], strat["position_size_r"]
    trend_on = (strat.get("trend_filter") or {}).get("enabled", False)
    conf = strat.get("confirm") or {}
    fam = family_of(conf["patterns"]) if conf.get("enabled") else None
    lookback = int(conf.get("dip_lookback", 4))
    df = strat.get("daily_filter") or {}
    out, pos, recent_dip = [], None, -10**9
    for i in range(max(lo, B.WARMUP), hi):
        k, r = candles[i], pre["rsi"][i]
        if r < entry:
            recent_dip = i
        if not pre["in_hours"][i]:
            continue
        if pos is not None:
            px = None
            if k["l"] <= pos * (1 - stop / 100):
                px = min(k["o"], pos * (1 - stop / 100))
            elif r > exit_rsi:
                px = k["c"]
            if px is not None:
                out.append((px / pos - 1 - spread) * size)
                pos = None
            continue
        if fam is None:
            if r >= entry:
                continue
        elif i - recent_dip > lookback or not pre["pattern"][fam][i]:
            continue
        if trend_on and not pre["trend_up"][i]:
            continue
        if df.get("enabled"):
            d = days[pre["day"][i]]
            sma = d.get(f"sma{df['sma']}")
            if sma is None or not df["min_pct"] / 100 <= d["c"] / sma - 1 < df["max_pct"] / 100:
                continue
        pos = k["c"]
    return out


def total(pnl: list[float]) -> float:
    return float(np.prod([1 + v for v in pnl]) - 1) if pnl else 0.0


def evaluate(data: dict, strat: dict) -> dict:
    q = data["quarters"]
    per_q = [simulate(data, strat, q[j], q[j + 1]) for j in range(4)]
    all_t = [v for t in per_q for v in t]
    double = simulate(data, strat, q[0], q[4], spread=0.004)
    return {"trades": len(all_t), "total": total(all_t), "quarters": [total(t) for t in per_q],
            "quarter_trades": [len(t) for t in per_q], "total_2x_cost": total(double)}


# ---- candidates: one change each ---------------------------------------------
def candidates(strat: dict) -> list[dict]:
    """Each candidate = (group, label, changed strategy). Group = what neighbours are compared within."""
    out = []

    def add(group, label, mutate):
        s = copy.deepcopy(strat)
        mutate(s)
        if s != strat:
            out.append({"group": group, "label": label, "strategy": s})

    for v in (25, 30, 35, 40, 45):
        add("entry.threshold", f"buy RSI<{v}", lambda s, v=v: s["entry"].__setitem__("threshold", v))
    for v in (60, 65, 70):
        add("exit.rsi_above", f"sell RSI>{v}", lambda s, v=v: s["exit"].__setitem__("rsi_above", v))
    for v in (2.0, 2.5, 3.0):
        add("stop_loss_pct", f"stop {v}%", lambda s, v=v: s.__setitem__("stop_loss_pct", v))
    add("daily_filter", "daily filter off",
        lambda s: s.setdefault("daily_filter", {}).__setitem__("enabled", False))
    for sma in (20, 50, 100):
        for lo, hi in ((0, 1), (0, 2), (0, 3), (-1, 3)):
            add(f"daily_filter.sma{sma}", f"daily {lo:+d}..{hi:+d}% vs SMA{sma}",
                lambda s, sma=sma, lo=lo, hi=hi: s.__setitem__(
                    "daily_filter", {"enabled": True, "sma": sma, "min_pct": float(lo), "max_pct": float(hi)}))
    add("trend_filter", "EMA50/200 trend filter " + ("off" if (strat.get("trend_filter") or {}).get("enabled") else "on"),
        lambda s: s.setdefault("trend_filter", {"indicator": "ema", "fast": 50, "slow": 200}).__setitem__(
            "enabled", not (strat.get("trend_filter") or {}).get("enabled", False)))
    for fam in ("none", "two_candle", "wick", "three_candle"):
        if fam == "none":
            add("confirm", "no pattern confirmation",
                lambda s: s.setdefault("confirm", {}).__setitem__("enabled", False))
        else:
            add("confirm", f"confirm with {fam}", lambda s, fam=fam: s.__setitem__(
                "confirm", {"enabled": True, "patterns": B.FAMILIES[fam], "dip_lookback": 4}))
    return out


def accept_candidate(c: dict, base: dict) -> bool:
    """Decide whether a candidate is a real improvement or just luck.

    c and base are evaluate() results plus, for c, "neighbours": the evaluate() results of the
    other candidates in the same group (e.g. other bands of the same SMA, other buy levels).
    Keys: trades, total, quarters (list of 4 returns), quarter_trades, total_2x_cost.
    """
    if c["trades"] < MIN_TRADES:                                  # enough evidence
        return False
    if c["total"] < base["total"] + MIN_GAIN:                     # clearly better than now
        return False
    if sum(q > 0 for q in c["quarters"]) < 3:                     # consistent: >= 3 of 4 quarters positive
        return False
    if c["total_2x_cost"] <= 0:                                   # survives double costs
        return False
    neighbours = c.get("neighbours") or []
    if neighbours and sum(n["total"] > base["total"] for n in neighbours) * 2 < len(neighbours):
        return False                                              # not a lucky number: most neighbours agree
    return True


# ---- run -----------------------------------------------------------------------
def learn(apply: bool = False, verbose: bool = True) -> dict | None:
    strat = state.load_yaml(state.STRATEGY)
    data = prepare()
    base = evaluate(data, strat)
    cands = candidates(strat)
    for c in cands:
        c["result"] = evaluate(data, c["strategy"])
    for c in cands:
        c["result"]["neighbours"] = [o["result"] for o in cands if o["group"] == c["group"] and o is not c]
        c["accepted"] = accept_candidate(c["result"], base)

    if verbose:
        print(f"current v{strat['version']}: {base['trades']} trades, {base['total']:+.2%} over 365d, "
              f"quarters {' '.join(f'{x:+.1%}' for x in base['quarters'])}")
        for c in sorted(cands, key=lambda c: -c["result"]["total"])[:10]:
            r = c["result"]
            print(f"  {'✓' if c['accepted'] else ' '} {c['label']:32} {r['trades']:4d}tr {r['total']:+7.2%} "
                  f"(2x cost {r['total_2x_cost']:+6.2%}) quarters {' '.join(f'{x:+.1%}' for x in r['quarters'])}")

    accepted = sorted((c for c in cands if c["accepted"]), key=lambda c: -c["result"]["total"])
    REPORT.write_text(json.dumps({"ts": state.now_iso(), "version": strat["version"], "base": base,
                                  "candidates": [{k: v for k, v in c.items() if k != "strategy"} | {
                                      "result": {k: v for k, v in c["result"].items() if k != "neighbours"}}
                                                 for c in cands]}, indent=1, default=str))
    if not accepted:
        if verbose:
            print("No change passed the acceptance rule: keeping the current strategy.")
        return None

    best = accepted[0]
    proposal = {"ts": state.now_iso(), "from_version": strat["version"], "change": best["label"],
                "base_total": base["total"], "new_total": best["result"]["total"],
                "quarters": best["result"]["quarters"], "applied": False}
    if apply:
        if os.environ.get("HERMES_TRADING_MODE", "paper") != "paper":
            print("Live mode: proposing only, never auto-applying.")
        else:
            new = state.archive_and_bump(strat)
            new.update({k: v for k, v in best["strategy"].items() if k not in ("version", "lessons")})
            new.setdefault("lessons", []).append(
                f"Learner ({state.now_iso()[:10]}): {best['label']} — 365d {base['total']:+.2%} -> "
                f"{best['result']['total']:+.2%}, quarters {[round(x, 4) for x in best['result']['quarters']]}.")
            state.write_yaml(state.STRATEGY, new)
            state.append_jsonl(state.HYPOTHESES, {"ts": state.now_iso(), "source": "learner",
                                                  "from_version": strat["version"], "to_version": new["version"],
                                                  "variable": best["group"], "hypothesis": best["label"]})
            proposal.update(applied=True, to_version=new["version"])
    state.append_jsonl(PROPOSALS, proposal)
    if verbose:
        print(f"PROPOSAL: {best['label']} ({base['total']:+.2%} -> {best['result']['total']:+.2%})"
              + (f" — applied as v{proposal['to_version']}" if proposal["applied"] else " — not applied"))
    return proposal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="apply the winning change (paper mode only)")
    args = ap.parse_args()
    learn(apply=args.apply)


if __name__ == "__main__":
    main()
