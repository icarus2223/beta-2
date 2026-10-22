"""Bullish reversal candlestick patterns (the "buy" side of the desk-mat cheat sheet).

Each detector looks at the most recent candles and returns True if the pattern just completed.
A candle is a dict with o, h, l, c (open, high, low, close).
"""
from __future__ import annotations

from typing import Callable


def _body(k: dict) -> float:
    return abs(k["c"] - k["o"])


def _range(k: dict) -> float:
    return max(k["h"] - k["l"], 1e-12)


def _green(k: dict) -> bool:
    return k["c"] > k["o"]


def _red(k: dict) -> bool:
    return k["c"] < k["o"]


def hammer(c: list[dict]) -> bool:
    """Small body near the top, lower wick at least twice the body: sellers pushed down, buyers pushed back."""
    k = c[-1]
    body, rng = _body(k), _range(k)
    lower = min(k["o"], k["c"]) - k["l"]
    upper = k["h"] - max(k["o"], k["c"])
    return body <= 0.35 * rng and lower >= 2 * max(body, 1e-12) and upper <= 0.15 * rng


def dragonfly_doji(c: list[dict]) -> bool:
    """Open ≈ close at the top of the range with a long lower wick."""
    k = c[-1]
    rng = _range(k)
    lower = min(k["o"], k["c"]) - k["l"]
    return _body(k) <= 0.1 * rng and lower >= 0.7 * rng


def bullish_engulfing(c: list[dict]) -> bool:
    """A green candle whose body fully covers the previous red candle's body."""
    a, b = c[-2], c[-1]
    return _red(a) and _green(b) and b["o"] <= a["c"] and b["c"] >= a["o"] and _body(b) > _body(a)


def bullish_harami(c: list[dict]) -> bool:
    """A small green body inside the previous large red body: selling is losing force."""
    a, b = c[-2], c[-1]
    return (_red(a) and _green(b) and b["o"] >= a["c"] and b["c"] <= a["o"]
            and _body(b) <= 0.6 * _body(a) and _body(a) >= 0.5 * _range(a))


def piercing_line(c: list[dict]) -> bool:
    """After a red candle, a green candle opens at/below its close and closes above its midpoint."""
    a, b = c[-2], c[-1]
    mid = (a["o"] + a["c"]) / 2
    return _red(a) and _green(b) and b["o"] <= a["c"] and mid < b["c"] < a["o"]


def morning_star(c: list[dict]) -> bool:
    """Big red, small 'star', then a green candle closing above the middle of the first."""
    a, s, b = c[-3], c[-2], c[-1]
    return (_red(a) and _body(a) >= 0.5 * _range(a) and _body(s) <= 0.5 * _body(a)
            and _green(b) and b["c"] > (a["o"] + a["c"]) / 2)


def three_inside_up(c: list[dict]) -> bool:
    """Bullish harami, confirmed by a third candle closing above the first candle's open."""
    return bullish_harami(c[:-1]) and _green(c[-1]) and c[-1]["c"] > c[-3]["o"]


def three_outside_up(c: list[dict]) -> bool:
    """Bullish engulfing, confirmed by a third green candle closing higher."""
    return bullish_engulfing(c[:-1]) and _green(c[-1]) and c[-1]["c"] > c[-2]["c"]


def three_white_soldiers(c: list[dict]) -> bool:
    """Three rising green candles, each closing near its high."""
    last = c[-3:]
    return (all(_green(k) and (k["h"] - k["c"]) <= 0.3 * _range(k) for k in last)
            and last[0]["c"] < last[1]["c"] < last[2]["c"])


PATTERNS: dict[str, Callable[[list[dict]], bool]] = {
    "hammer": hammer,
    "dragonfly_doji": dragonfly_doji,
    "bullish_engulfing": bullish_engulfing,
    "bullish_harami": bullish_harami,
    "piercing_line": piercing_line,
    "morning_star": morning_star,
    "three_inside_up": three_inside_up,
    "three_outside_up": three_outside_up,
    "three_white_soldiers": three_white_soldiers,
}


def bullish_signal(candles: list[dict], names: list[str]) -> str | None:
    """Name of the first listed pattern that just completed on the latest candles, or None."""
    if len(candles) < 3:
        return None
    for name in names:
        if PATTERNS[name](candles):
            return name
    return None
