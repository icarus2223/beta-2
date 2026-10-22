"""Entrypoint: `uv run python -m hermes_trading.run [--asset BTC/USDT] [--once] [--interval 60]`."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import logging
import os
from pathlib import Path

from rich.logging import RichHandler

from . import state
from .loop import run_loop


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.split(" #", 1)[0]  # allow trailing comments
            os.environ.setdefault(key.strip(), value.strip())


def acquire_lock():
    """Hold an exclusive lock on state/bot.lock for the life of this process.

    A second copy (two bots would each place orders on the same signal) fails to get it and exits.
    The OS releases the lock when the process ends, even after a crash, so it never goes stale.
    """
    path = state.STATE_DIR / "bot.lock"
    handle = open(path, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        raise SystemExit(f"Another hermes-trading bot is already running (pid {owner}). "
                         f"Stop it first (Ctrl+C in its tab, or `kill {owner}`); not starting a second copy.")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle  # keep a reference: closing the file would release the lock


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    lock = acquire_lock()  # noqa: F841  (held until the process exits)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_path=False)])
    for noisy in ("httpx", "httpcore", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    goal = state.load_yaml(state.GOAL)
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default=goal["asset"])
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--once", action="store_true", help="run a single tick and exit (smoke test)")
    args = p.parse_args()

    logging.getLogger("run").info("Booting hermes-trading worker · %s · mode=%s",
                                  args.asset, os.environ.get("HERMES_TRADING_MODE", "paper"))
    asyncio.run(run_loop(args.asset, args.interval, args.once))


if __name__ == "__main__":
    main()
