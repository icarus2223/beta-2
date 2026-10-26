"""Check the trading bot on this Mac and on the VPS side by side, and warn about unsafe combinations.

    python3 status.py            # VPS reached via the `hostinger` alias in ~/.ssh/config
    HERMES_VPS=other python3 status.py

Standard library only, so it runs without the project's virtualenv.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

VPS_HOST = os.environ.get("HERMES_VPS", "hostinger")
VPS_DIR = os.environ.get("HERMES_VPS_DIR", "/docker/hermes-agent-zgmz/data/beta-2/hermes-trading")
LOCAL_DIR = str(Path(__file__).resolve().parent)
STALE_AFTER_S = 300  # a healthy bot writes a heartbeat every minute

# Runs on each machine (locally, or piped to `python3 -` over SSH) and prints one JSON object.
PROBE = r'''
import json, os, re, subprocess, sys
from datetime import datetime, timezone
d = sys.argv[1]
out = {"dir_exists": os.path.isdir(d)}
def sh(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return ""
pids = []
for pid in sh(["pgrep", "-f", "--", "-m hermes_trading.run"]).split():  # "--": pattern starts with "-"
    comm = os.path.basename(sh(["ps", "-o", "comm=", "-p", pid]))
    if comm.lower().startswith("python"):  # skip uv/caffeinate/bash wrappers around the same bot
        pids.append(int(pid))
out["bots"] = len(pids)
def read(p):
    try: return open(os.path.join(d, p)).read()
    except OSError: return None
env = read(".env") or ""
m = re.search(r"^HERMES_TRADING_MODE=(\w+)", env, re.M)
out["mode"] = m.group(1) if m else "paper"
k = re.search(r"^T212_API_KEY=(\S*)", env, re.M)
out["t212_key"] = bool(k and k.group(1) and not k.group(1).startswith("#"))
s = read("state/strategy.yaml") or ""
v = re.search(r"^version:\s*'?\"?(\w+)", s, re.M)
out["strategy"] = v.group(1) if v else None
try:
    hb = json.loads(read("state/heartbeat.json") or "{}")
    ts = datetime.fromisoformat(hb["ts"])
    out["heartbeat_age_s"] = int((datetime.now(timezone.utc) - ts).total_seconds())
    out["heartbeat_ok"] = hb.get("ok")
    out["last_price"] = hb.get("last")
    out["rsi"] = hb.get("rsi")
except Exception:
    out["heartbeat_age_s"] = None
out["position"] = os.path.exists(os.path.join(d, "state/position.json"))
trades = read("state/trades.jsonl") or ""
out["closed_trades"] = trades.count('"event": "close"')
out["commit"] = sh(["git", "-c", "safe.directory=*", "-C", d, "rev-parse", "--short", "HEAD"]) or None
print(json.dumps(out))
'''


def probe_local() -> dict:
    r = subprocess.run([sys.executable, "-c", PROBE, LOCAL_DIR], capture_output=True, text=True, timeout=30)
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr.strip()[-200:]}


def probe_vps() -> dict:
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", VPS_HOST, "python3", "-", VPS_DIR],
                           input=PROBE, capture_output=True, text=True, timeout=40)
    except subprocess.TimeoutExpired:
        return {"error": "SSH timed out"}
    if r.returncode != 0:
        return {"error": (r.stderr.strip() or "ssh failed")[-200:]}
    return json.loads(r.stdout)


def age(s: int | None) -> str:
    if s is None:
        return "never"
    if s < 120:
        return f"{s}s ago"
    if s < 7200:
        return f"{s // 60}m ago"
    return f"{s // 3600}h ago" if s < 172800 else f"{s // 86400}d ago"


def row(label: str, mac: str, vps: str) -> None:
    print(f"  {label:<16} {mac:<26} {vps}")


def describe(p: dict) -> dict:
    if "error" in p:
        return {k: "—" for k in ("bot", "mode", "strategy", "heartbeat", "market", "position", "trades", "commit")}
    hb = p.get("heartbeat_age_s")
    alive = p["bots"] > 0 and hb is not None and hb < STALE_AFTER_S
    bot = ("RUNNING" if alive else "running, no heartbeat!") if p["bots"] else "stopped"
    if p["bots"] > 1:
        bot = f"{p['bots']} COPIES!"
    return {
        "bot": bot,
        "mode": p["mode"] + (" (key set)" if p["t212_key"] else ""),
        "strategy": f"v{p['strategy']}",
        "heartbeat": age(hb) + ("" if p.get("heartbeat_ok") in (True, None) else " (error)"),
        "market": f"BTC {p['last_price']:.0f} · RSI {p['rsi']}" if p.get("last_price") else "—",
        "position": "OPEN" if p["position"] else "flat",
        "trades": str(p["closed_trades"]),
        "commit": p["commit"] or "—",
    }


def main() -> int:
    mac, vps = probe_local(), probe_vps()
    m, v = describe(mac), describe(vps)
    print(f"\n  {'':<16} {'MAC':<26} VPS ({VPS_HOST})")
    for key, label in (("bot", "Bot"), ("mode", "Mode"), ("strategy", "Strategy"), ("heartbeat", "Last heartbeat"),
                       ("market", "Market"), ("position", "Position"), ("trades", "Closed trades"),
                       ("commit", "Git commit")):
        row(label, m[key], v[key])

    problems, notes = [], []
    if "error" in vps:
        problems.append(f"Cannot reach the VPS: {vps['error']}")
    if "error" in mac:
        problems.append(f"Local check failed: {mac['error']}")
    if "error" not in mac and "error" not in vps:
        live = [name for name, p in (("Mac", mac), ("VPS", vps)) if p["bots"] and p["mode"] == "live"]
        if len(live) == 2:
            problems.append("TWO LIVE BOTS (Mac + VPS) on the same Trading 212 account: every signal buys twice. "
                            "Stop one now.")
        for name, p in (("Mac", mac), ("VPS", vps)):
            if p["bots"] > 1:
                problems.append(f"{name} runs {p['bots']} bot copies at once.")
            if p["bots"] and (p["heartbeat_age_s"] is None or p["heartbeat_age_s"] >= STALE_AFTER_S):
                problems.append(f"{name} bot process exists but has not written a heartbeat recently (stuck?).")
        if not mac["bots"] and not vps["bots"]:
            problems.append("No bot is running anywhere.")
        if mac["strategy"] != vps["strategy"]:
            notes.append(f"Strategy differs: Mac v{mac['strategy']}, VPS v{vps['strategy']} "
                         "(only the running machine's version matters).")
        if mac["commit"] != vps["commit"]:
            notes.append(f"Code differs: Mac {mac['commit']}, VPS {vps['commit']} (git pull on the VPS to sync).")
        if mac["position"] and vps["position"]:
            problems.append("Both machines think they hold a position.")
    print()
    for p in problems:
        print(f"  ✗ {p}")
    for n in notes:
        print(f"  • {n}")
    if not problems:
        runner = "VPS" if vps.get("bots") else "Mac"
        print(f"  ✓ OK: one bot running ({runner}), no double trading.")
    print()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
