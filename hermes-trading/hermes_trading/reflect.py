"""Reflection cycle: look at outcomes, change exactly ONE strategy variable.

    uv run python -m hermes_trading.reflect --fallback   # deterministic rule
    uv run python -m hermes_trading.reflect --hermes     # ask Hermes Agent
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from . import state
from .score import metrics, score

LOOKBACK = 25

# The only variables a reflection may change: (min, max, largest single step).
# Enforced in apply_change(), so neither the fallback rule nor Hermes can walk a parameter somewhere absurd.
BOUNDS: dict[str, tuple[float, float, float]] = {
    "entry.threshold": (15, 45, 3),
    # Floors from the 180d backtest (Oct 2026): stops tighter than 2% and exits below RSI 60
    # lost money in both the train and the unseen test period, so reflection may not go there.
    "exit.rsi_above": (60, 70, 5),
    "stop_loss_pct": (2.0, 4.0, 0.5),
    "position_size_r": (0.1, 0.5, 0.1),
    "trend_filter.fast": (20, 100, 10),   # must stay a *slow* filter: fast EMAs cancel the RSI dip itself
    "trend_filter.slow": (100, 300, 25),
}
NEVER_INCREASE = {"position_size_r"}  # reflection may shrink risk, never grow it

STOP_LOSS_MIN = BOUNDS["stop_loss_pct"][0]
ENTRY_MIN, ENTRY_MAX = BOUNDS["entry.threshold"][:2]


class HermesError(RuntimeError):
    """Hermes could not produce a usable hypothesis. Callers fall back to the deterministic rule."""


def check_change(variable: str, old_value: float, new_value: Any) -> None:
    if variable not in BOUNDS:
        raise ValueError(f"{variable} is not an allowed variable ({', '.join(BOUNDS)})")
    if not isinstance(new_value, (int, float)) or isinstance(new_value, bool):
        raise ValueError(f"new value for {variable} must be a number, got {new_value!r}")
    lo, hi, max_step = BOUNDS[variable]
    if not lo <= new_value <= hi:
        raise ValueError(f"{variable}={new_value} is outside the allowed range {lo}–{hi}")
    if abs(new_value - old_value) > max_step + 1e-9:
        raise ValueError(f"{variable} {old_value} → {new_value} is a bigger step than {max_step}")
    if variable in NEVER_INCREASE and new_value > old_value:
        raise ValueError(f"{variable} may only decrease ({old_value} → {new_value} refused)")


def get_path(d: dict, dotted: str) -> Any:
    for key in dotted.split("."):
        d = d[key]
    return d


def set_path(d: dict, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    for key in parents:
        d = d[key]
    d[leaf] = value


def apply_change(variable: str, new_value: Any, hypothesis: str, source: str,
                 confidence: float | None = None) -> dict:
    """The one place strategy.yaml gets rewritten. Enforces the one-variable rule and BOUNDS."""
    strategy = state.load_yaml(state.STRATEGY)
    old_value = get_path(strategy, variable)  # KeyError if the variable doesn't exist
    if not isinstance(old_value, (int, float)) or isinstance(old_value, bool):
        raise ValueError(f"{variable} is not a numeric parameter; refusing to change it")
    if new_value == old_value:
        raise ValueError(f"{variable} already equals {new_value}; no experiment to run")
    check_change(variable, old_value, new_value)

    trades = state.closed_trades()[-LOOKBACK:]
    goal = state.load_yaml(state.GOAL)
    candidate = json.loads(json.dumps(strategy))
    set_path(candidate, variable, new_value)
    f = candidate.get("trend_filter") or {}
    if f.get("enabled") and f["fast"] >= f["slow"]:
        raise ValueError(f"trend_filter.fast ({f['fast']}) must stay below trend_filter.slow ({f['slow']})")

    bumped = state.archive_and_bump(strategy)
    set_path(bumped, variable, new_value)
    state.write_yaml(state.STRATEGY, bumped)

    record = {"ts": state.now_iso(), "source": source, "from_version": strategy["version"],
              "to_version": bumped["version"], "variable": variable, "old": old_value, "new": new_value,
              "hypothesis": hypothesis, "confidence": confidence,
              "score_before": round(score(trades, goal), 4), "metrics_before": metrics(trades)}
    state.append_jsonl(state.HYPOTHESES, record)
    print(f"v{record['from_version']} → v{record['to_version']}: {variable} {old_value} → {new_value}")
    print(f"  why: {hypothesis}")
    return record


def choose_fallback_change(m: dict, goal: dict, strategy: dict) -> tuple[str, Any, str] | None:
    """Pick ONE variable to change from the metrics alone (no LLM).

    m:        output of score.metrics() — keys n, return, return_30d, max_drawdown, sharpe
    goal:     goal.yaml (target_return_30d, max_drawdown, min_sharpe, ...)
    strategy: current strategy.yaml

    Return (dotted_variable_path, new_value, one_sentence_reason), or None to skip this cycle.
    """
    if m["n"] < goal["reflection_every"]:
        return None  # too few trades to tell signal from noise

    stop = strategy["stop_loss_pct"]
    entry = strategy["entry"]["threshold"]

    # Candidates in priority order: protect capital first, then chase return, then quality.
    # A candidate already at its bound is skipped so the next problem still gets worked on.
    candidates = []
    if m["max_drawdown"] > goal["max_drawdown"]:
        candidates.append(("stop_loss_pct", round(max(STOP_LOSS_MIN, stop - 0.2), 2),
                           f"drawdown {m['max_drawdown']:.1%} > {goal['max_drawdown']:.0%} limit; tighter stop should cut losers sooner"))
    if m["return_30d"] < goal["target_return_30d"]:
        candidates.append(("entry.threshold", min(ENTRY_MAX, entry + 2),
                           f"30d return {m['return_30d']:+.1%} < {goal['target_return_30d']:+.0%} target; a looser entry should take more trades"))
    if m["sharpe"] < goal["min_sharpe"]:
        candidates.append(("entry.threshold", max(ENTRY_MIN, entry - 2),
                           f"Sharpe {m['sharpe']:.2f} < {goal['min_sharpe']}; a stricter entry should take fewer, cleaner trades"))

    current = {"stop_loss_pct": stop, "entry.threshold": entry}
    for variable, new_value, reason in candidates:
        if new_value != current[variable]:
            return variable, new_value, reason
    return None


def reflect_fallback() -> dict | None:
    trades = state.closed_trades()[-LOOKBACK:]
    goal = state.load_yaml(state.GOAL)
    strategy = state.load_yaml(state.STRATEGY)
    choice = choose_fallback_change(metrics(trades), goal, strategy)
    if choice is None:
        print("Fallback reflection: on target, no change this cycle.")
        return None
    variable, new_value, reason = choice
    return apply_change(variable, new_value, reason, source="fallback")


HERMES_PROMPT = """You are a cautious quantitative strategist reviewing a live trading bot that runs
real money on a small Trading 212 account. The bot reads BTC/USDT's 15-minute RSI
around the clock, but it can only trade the Invesco Physical Bitcoin ETP (BTIC)
during Xetra hours (09:05-17:25 Berlin, weekdays). When trend_filter is enabled, it
only buys the RSI dip while the fast EMA is above the slow EMA (an uptrend); buys
skipped in a downtrend do not appear as trades. When confirm is enabled, an RSI dip
alone is not enough: one of the listed bullish candle patterns must complete within
dip_lookback candles of the dip. When daily_filter is enabled, it only buys while
yesterday's daily close is between min_pct and max_pct above its daily SMA (dips that
land on the 50-day average tested best over a year). Each buy is protected by a stop
order held at Trading 212; trades closed by that stop are logged as "broker_stop".
Overnight and weekend moves can gap past the stop, so real losses can exceed
stop_loss_pct.

Your job: propose ONE controlled experiment that is most likely to raise the
strategy's score against the goal below, or decide that no change is justified yet.

GOAL:
{goal}

CURRENT STRATEGY (strategy.yaml):
{strategy}

METRICS over the last {n} closed trades:
{metrics}

LAST {n} CLOSED TRADES (oldest first):
{trades}

PREVIOUS HYPOTHESES, with score_before (most recent last):
{hypotheses}

How to decide:
1. First check whether the previous hypothesis helped: compare its score_before
   with the current metrics. If it clearly hurt, consider reverting that variable.
2. Priority order: protect capital (drawdown), then return, then Sharpe.
3. With fewer than {min_trades} trades, or results driven by a single outlier trade,
   prefer no change.

Rules:
- Change exactly ONE numeric variable, as a dotted path (e.g. entry.threshold).
- Allowed variables, ranges and largest single step:
{bounds}
- Never increase: {never_increase}.
- Do not edit files, run commands or place orders. Only reply.

Reply format: up to 3 one-line hypotheses, each naming one variable and the
expected effect. Then, on the final line, exactly one JSON object and nothing after it:
{{"variable": "<dotted.path>", "new_value": <number>, "hypothesis": "<one sentence>", "confidence": <0-1>}}
or, if no change is justified:
{{"variable": null, "hypothesis": "<why not>"}}
"""


def build_hermes_prompt() -> str:
    trades = state.closed_trades()[-LOOKBACK:]
    slim = [{k: t.get(k) for k in ("ts", "return_pct", "reason", "strategy_version")} for t in trades]
    hypotheses = [{k: h.get(k) for k in ("ts", "source", "from_version", "to_version", "variable", "old", "new",
                                         "hypothesis", "confidence", "score_before")}
                  for h in state.read_jsonl(state.HYPOTHESES)[-5:]]
    return HERMES_PROMPT.format(
        goal=state.GOAL.read_text(), strategy=state.STRATEGY.read_text(), n=len(trades),
        metrics=json.dumps(metrics(trades), indent=2),
        trades="\n".join(json.dumps(t) for t in slim) or "(none yet)",
        hypotheses="\n".join(json.dumps(h) for h in hypotheses) or "(none yet)",
        min_trades=state.load_yaml(state.GOAL)["reflection_every"],
        bounds="\n".join(f"    {v}: {lo}-{hi}, step at most {step}" for v, (lo, hi, step) in BOUNDS.items()),
        never_increase=", ".join(sorted(NEVER_INCREASE)),
    )


def parse_hermes_reply(text: str) -> dict:
    """The last JSON object in the reply that has a "variable" key."""
    for candidate in reversed(re.findall(r"\{[^{}]*\"variable\"[^{}]*\}", text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise HermesError(f"no hypothesis JSON in Hermes reply: {text[-500:]!r}")


SECRET_ENV_PREFIXES = ("T212_", "EXCHANGE_API_", "GLASSNODE_", "NEWS_API_", "HERMES_TRADING_")


def hermes_env() -> dict[str, str]:
    """This process's environment minus every trading credential and live-mode switch."""
    return {k: v for k, v in os.environ.items() if not k.startswith(SECRET_ENV_PREFIXES)}


def reflect_hermes() -> dict | None:
    if not shutil.which("hermes"):
        raise HermesError("`hermes` not on PATH")
    cmd = ["hermes", "--ignore-rules", "-z", build_hermes_prompt()]
    if os.environ.get("HERMES_REFLECT_TOOLSETS"):
        cmd[1:1] = ["-t", os.environ["HERMES_REFLECT_TOOLSETS"]]
    # Empty working folder and no trading secrets: Hermes sees only the prompt, not this project's
    # files or the broker/exchange keys that .env loaded into this process's environment.
    with tempfile.TemporaryDirectory() as empty:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=empty, env=hermes_env())
        except subprocess.TimeoutExpired as exc:
            raise HermesError("hermes took longer than 10 minutes") from exc
    if out.returncode != 0:
        raise HermesError(f"hermes exited {out.returncode}: {out.stderr[-500:]}")

    choice = parse_hermes_reply(out.stdout)
    if choice.get("variable") is None:
        print(f"Hermes: no change this cycle. {choice.get('hypothesis', '')}")
        state.append_jsonl(state.HYPOTHESES, {"ts": state.now_iso(), "source": "hermes", "variable": None,
                                              "hypothesis": choice.get("hypothesis")})
        return None
    try:
        return apply_change(choice["variable"], choice["new_value"], choice["hypothesis"], source="hermes",
                            confidence=choice.get("confidence"))
    except (KeyError, ValueError) as exc:
        raise HermesError(f"Hermes proposed an invalid change {choice}: {exc}") from exc


def reflect_auto(mode: str) -> dict | None:
    """Used by the loop. 'hermes' asks Hermes and falls back to the fixed rule if anything goes wrong."""
    if mode == "hermes":
        try:
            return reflect_hermes()
        except HermesError as exc:
            print(f"Hermes reflection failed, using the fallback rule instead: {exc}")
    return reflect_fallback()


def main() -> None:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fallback", action="store_true")
    mode.add_argument("--hermes", action="store_true")
    mode.add_argument("--show-prompt", action="store_true", help="print the Hermes prompt without calling it")
    args = p.parse_args()
    if args.show_prompt:
        print(build_hermes_prompt())
    elif args.hermes:
        reflect_hermes()
    else:
        reflect_fallback()


if __name__ == "__main__":
    main()
