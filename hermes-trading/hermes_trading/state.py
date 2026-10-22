"""Read/write helpers for everything under state/."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

STATE_DIR = Path(os.environ.get("HERMES_STATE_DIR", Path(__file__).resolve().parent.parent / "state"))
GOAL = STATE_DIR / "goal.yaml"
STRATEGY = STATE_DIR / "strategy.yaml"
TRADES = STATE_DIR / "trades.jsonl"
HYPOTHESES = STATE_DIR / "hypotheses.jsonl"
HISTORY = STATE_DIR / "history"
HEARTBEAT = STATE_DIR / "heartbeat.json"
POSITION = STATE_DIR / "position.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False))
    tmp.replace(path)  # atomic: the loop never reads a half-written strategy


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def closed_trades() -> list[dict[str, Any]]:
    return [t for t in read_jsonl(TRADES) if t.get("event") == "close"]


def archive_and_bump(strategy: dict[str, Any]) -> dict[str, Any]:
    """Save the current strategy to history/ and return a copy with version+1."""
    HISTORY.mkdir(parents=True, exist_ok=True)
    old_version = int(str(strategy.get("version", "0")))
    shutil.copy(STRATEGY, HISTORY / f"v{old_version:04d}.yaml")
    bumped = json.loads(json.dumps(strategy))  # deep copy
    bumped["version"] = f"{old_version + 1:02d}"
    return bumped
