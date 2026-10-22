"""Score closed trades against goal.yaml. Output is a single number in [-1, +1]."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


def _clip(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


def metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
    """Equity-curve metrics from closed trades (each has `pnl_frac` and `ts`)."""
    if not trades:
        return {"n": 0, "return": 0.0, "return_30d": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    pnl = np.array([t["pnl_frac"] for t in trades], dtype=float)
    equity = np.cumprod(1 + pnl)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    max_dd = float(np.max(1 - equity / peak))
    total = float(equity[-1] - 1)

    first = datetime.fromisoformat(trades[0]["ts"])
    last = datetime.fromisoformat(trades[-1]["ts"])
    days = max((last - first).total_seconds() / 86400, 1.0)  # don't extrapolate from minutes
    return_30d = (1 + total) ** (30 / days) - 1

    # Per-trade Sharpe scaled by sqrt(trades per year). Crude but monotonic, which is what reflection needs.
    sharpe = 0.0
    if len(pnl) > 1 and pnl.std(ddof=1) > 0:
        trades_per_year = len(pnl) / days * 365
        sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(trades_per_year))

    return {"n": len(trades), "return": total, "return_30d": float(return_30d), "max_drawdown": max_dd, "sharpe": sharpe}


def score(trades: list[dict[str, Any]], goal: dict[str, Any]) -> float:
    m = metrics(trades)
    if m["n"] == 0:
        return 0.0
    ret_part = _clip(m["return_30d"] / goal["target_return_30d"])
    dd_part = _clip(1 - m["max_drawdown"] / goal["max_drawdown"])
    sharpe_part = _clip(m["sharpe"] / goal["min_sharpe"])
    composite = (ret_part + dd_part + sharpe_part) / 3
    if m["return"] < goal.get("failure_below", -1.0):
        composite = composite - 1.0  # steep penalty once we're past the floor
    return _clip(composite)
