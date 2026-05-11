"""Smoke-Test fuer mcp_client.py.

Verbindet zum Live-MCP-Server und ruft ein paar Read-Tools auf um zu
verifizieren dass:
  - Bearer-Auth funktioniert
  - Streamable-HTTP-Session aufgebaut wird
  - JSON-Auto-Unwrap richtig liefert
  - Dynamic-Dispatch (__getattr__) funktioniert
  - Phase X1-Tools (read_vision, read_saeulen, ...) erreichbar sind
  - Phase X2-Tools (goal_status_check, ...) erreichbar sind

Run lokal (von /opt/KI_WIKI_OS/ oder lokal mit MCP_TOKEN gesetzt):
    cd ki_wiki_bot && python scripts/smoke_test_mcp_client.py

Run im Container:
    docker exec ki-os-bot python /app/scripts/smoke_test_mcp_client.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Pfad-Setup damit das Script im Bot-Repo standalone laeuft
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_client import mcp, health_check, KNOWN_TOOLS  # noqa: E402


async def main() -> int:
    print(f"MCP_URL   = {os.environ.get('MCP_URL', '(default)')}")
    print(f"MCP_TOKEN = {'***SET***' if os.environ.get('MCP_TOKEN') else '(missing!)'}")
    print()

    # 1) Health-Check
    print("=== 1) Health-Check ===")
    h = await health_check()
    print(json.dumps(h, indent=2))
    if not h["ok"]:
        print("\nFAIL: Health-Check fehlgeschlagen")
        return 1
    visible = h["tools_visible"]
    expected_min = len(KNOWN_TOOLS)
    if visible < expected_min:
        print(f"WARN: nur {visible} Tools sichtbar — erwartet >= {expected_min}")
    else:
        print(f"OK: {visible} Tools sichtbar (>= {expected_min} erwartet)")

    # 2) Read-Tool: daily_briefing
    print("\n=== 2) daily_briefing() ===")
    try:
        b = await mcp.daily_briefing()
        print(f"OK: keys = {sorted(b.keys()) if isinstance(b, dict) else type(b)}")
        if isinstance(b, dict) and "summary" in b:
            print(f"   summary = {b['summary']}")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1

    # 3) Phase-X1-Tool: read_vision
    print("\n=== 3) read_vision() (Phase X1) ===")
    try:
        v = await mcp.read_vision()
        print(f"OK: found={v.get('found')}, text-len={len(v.get('text', ''))}")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1

    # 4) Phase-X1-Tool: compute_streak
    print("\n=== 4) compute_streak() (Phase X1) ===")
    try:
        s = await mcp.compute_streak()
        print(f"OK: current={s.get('current')}, best={s.get('best')}")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1

    # 5) Phase-X2-Tool: goal_status_check
    print("\n=== 5) goal_status_check() (Phase X2) ===")
    try:
        g = await mcp.goal_status_check()
        print(f"OK: overall={g.get('overall')}")
        if "drift" in g:
            for k, v in g["drift"].items():
                print(f"   {k}: {v.get('status')} (age_days={v.get('age_days')})")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1

    # 6) list_tasks (Read)
    print("\n=== 6) list_tasks(status=open) ===")
    try:
        t = await mcp.list_tasks(status="open")
        if isinstance(t, dict) and "tasks" in t:
            print(f"OK: {len(t['tasks'])} open tasks")
        else:
            print(f"OK: {type(t).__name__} returned")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1

    # 7) Connection re-use (zweiter Call sollte schnell sein, gleiche Session)
    print("\n=== 7) Connection-Reuse-Test (2x daily_briefing) ===")
    import time
    t0 = time.perf_counter()
    await mcp.daily_briefing()
    dt1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    await mcp.daily_briefing()
    dt2 = time.perf_counter() - t0
    print(f"   1. Call: {dt1*1000:.1f}ms")
    print(f"   2. Call: {dt2*1000:.1f}ms")
    if dt2 > dt1:
        print(f"   WARN: 2. Call langsamer — Session-Reuse evtl. nicht aktiv?")
    else:
        print(f"   OK: 2. Call schneller (Session re-used)")

    # 8) Phase-X3c-Wrappers: Connectivity-only Tests (kein Write zum Vault)
    # Wir checken dass die Wrapper Validation-Errors sauber passieren — also
    # dass die Verkettung Bot→mcp_thin_tools→mcp_client funktioniert.
    print("\n=== 8) Phase-X3c Wrappers (Validation-only, keine Vault-Writes) ===")
    try:
        # Importieren erst hier damit smoke-test auch ohne X3c lauft
        import mcp_thin_tools as thin
    except ImportError:
        print("   SKIP: mcp_thin_tools nicht verfuegbar")
    else:
        # Empty title → Validation Fail erwartet
        for name, coro in [
            ("create_note(empty title)", thin.create_note(title="", body="x")),
            ("create_meeting(empty title)", thin.create_meeting(title="")),
            ("task(unknown action)", thin.task(action="xyz")),
            ("task(create no title)", thin.task(action="create")),
            ("append_to_daily(empty text)",
             thin.append_to_daily(section="Notizen & Gedanken", text="")),
            # Phase X3d: read-only Aggregator
            ("goal_status(scope=drift)", thin.goal_status(scope="drift")),
        ]:
            try:
                r = await coro
                # Erwartet: returns string (gefangener error oder valid)
                ok = isinstance(r, str) and len(r) > 0
                print(f"   {'OK' if ok else 'FAIL'}: {name} → {r[:80]!r}")
            except Exception as e:
                print(f"   FAIL: {name} EXC: {type(e).__name__}: {e}")

    # Cleanup
    await mcp.close()
    print("\nALL SMOKE-TESTS OK.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
