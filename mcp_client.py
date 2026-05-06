"""KI-OS Bot — MCP-Client.

Dünner async-Wrapper um die MCP Streamable-HTTP-Verbindung zum Vault-Server
(`wiki-mcp.sima.business`). Handhabt:

  - Bearer-Auth via MCP_TOKEN aus ENV
  - Persistente Session (Re-Connect on demand, kein Reconnect-pro-Call)
  - Auto-JSON-Unwrap der `[{"type":"text","text":"..."}]`-Wrapper
  - Sync-Bridges für nicht-async Code-Pfade (atexit-safe)
  - Dynamic-Tool-Dispatch via __getattr__ (kein Boilerplate für 34 Tools)

Usage (async — Standardpfad im Bot):
    from mcp_client import mcp
    res = await mcp.create_task(title="Test", priority="high")
    res = await mcp.list_tasks(status="open")
    res = await mcp.read_vision()

Usage (sync — für JobQueue-Callbacks/CLI-Scripts):
    from mcp_client import mcp_sync
    res = mcp_sync.create_task(title="Test", priority="high")

Phase X3 vom MCP-Konsolidierungs-Plan: Bot wird thin-client. Alle
Vault-Operationen laufen jetzt durchs MCP statt lokal mit `frontmatter`
+ Filesystem-Calls. Single Source of Truth bleibt der MCP-Server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

log = logging.getLogger("ki-os-bot.mcp_client")

# ─── Config ──────────────────────────────────────────────────────────────────

MCP_URL = os.environ.get("MCP_URL", "https://wiki-mcp.sima.business/mcp/")
MCP_TOKEN = os.environ.get("MCP_TOKEN", "").strip()
MCP_TIMEOUT = float(os.environ.get("MCP_TIMEOUT", "30"))

# Tools die für die häufigen Bot-Operationen gebraucht werden — nur zur
# Doku/Validation. Dynamische Tools die nicht hier stehen, funktionieren
# trotzdem (per __getattr__).
KNOWN_TOOLS = {
    # --- Read ---
    "search_vault", "read_file", "list_files", "list_tasks", "daily_briefing",
    "self_test",
    "read_vision", "read_saeulen", "read_drift", "read_habits", "read_sport",
    "read_books", "read_wins", "compute_streak", "read_reminders",
    "read_yesterday_daily",
    # --- Write ---
    "create_note", "create_task", "create_meeting", "append_to_daily",
    "goal_log", "project_context", "raw_write",
    # --- Edit ---
    "edit_file", "task", "move",
    # --- Delete (2-step) ---
    "request_delete", "confirm_delete",
    # --- Maintain ---
    "vault_lint", "vault_autolink", "vault_maintain",
    "task_reactivate_recurring", "create_daily_skeleton", "goal_status_check",
}


class MCPError(RuntimeError):
    """MCP-Call fehlgeschlagen (Network, Auth, oder vom Server selbst)."""


# ─── Async-Client ────────────────────────────────────────────────────────────


class _MCPAsyncClient:
    """Persistente async MCP-Session. Lazy init beim ersten Call.

    Nicht thread-safe — soll von asyncio-Code aufgerufen werden. Re-Connect
    automatisch wenn die Session brokte (z.B. Server-Restart).
    """

    def __init__(self, url: str = MCP_URL, token: str = MCP_TOKEN, timeout: float = MCP_TIMEOUT):
        self.url = url
        self.token = token
        self.timeout = timeout
        self._session = None  # mcp.ClientSession
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> Any:
        """Stellt sicher dass eine offene Session da ist. Lock-protected.

        Bei Initialize-Failure (z.B. 401 Unauthorized) wird der Stack
        defensive-cleaned, ohne dass die Original-Exception unter einem
        Cleanup-Error verschwindet.
        """
        if self._session is not None:
            return self._session
        async with self._lock:
            if self._session is not None:
                return self._session
            log.info("MCP-Client: connecting %s", self.url)
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            stack = AsyncExitStack()
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            try:
                # streamablehttp_client kommt als async-context — managed durch Stack
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(self.url, headers=headers, timeout=self.timeout)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
            except BaseException as e:
                # Init failed — Stack zuruecksetzen damit kein Half-State bleibt.
                # aclose() kann selbst failen (anyio Cancel-Scope-Issue bei
                # nested-tasks) — wir loggen aber re-raisen die ECHTE Init-Exception.
                try:
                    await stack.aclose()
                except BaseException as cleanup_e:  # noqa: BLE001
                    log.debug("Cleanup nach Init-Fail: %s (ignoriert)", cleanup_e)
                self._stack = None
                self._session = None
                # Re-raise als MCPError mit klarer Diagnose
                if "401" in str(e) or "Unauthorized" in str(e):
                    raise MCPError(
                        f"401 Unauthorized — MCP_TOKEN fehlt/falsch. "
                        f"Pruefe: docker exec ki-os-bot env | grep MCP_TOKEN"
                    ) from e
                raise MCPError(f"Connect-Fail: {type(e).__name__}: {e}") from e
            self._stack = stack
            self._session = session
            log.info("MCP-Client: session ready")
            return self._session

    async def close(self) -> None:
        """Sauber Verbindung schliessen. Idempotent + tolerant gegen
        anyio Cancel-Scope-Errors (bekanntes Issue bei async-nested-context).
        """
        async with self._lock:
            stack = self._stack
            self._stack = None
            self._session = None
            if stack is not None:
                try:
                    await stack.aclose()
                except BaseException as e:  # noqa: BLE001
                    log.debug("MCP-Client close (ignoriert): %s", e)

    async def call(self, tool_name: str, args: dict | None = None) -> Any:
        """Ruft `tool_name` mit `args` auf, returnt geparsten Output.

        Auto-Unwrap:
          - MCP returns content als list[TextContent]. Wir nehmen den ersten
            Text-Block.
          - Wenn der Text valides JSON ist, wird er geparst — sonst raw str.

        Bei Connection-Error: 1× Reconnect-Versuch. Falls dann wieder fail,
        wird MCPError geraised.
        """
        args = args or {}
        try:
            session = await self._ensure_session()
            result = await session.call_tool(tool_name, args)
        except Exception as e:  # noqa: BLE001
            # Vermutlich brokte Session — Reconnect versuchen.
            log.warning("MCP-call %s failed (%s) — reconnecting", tool_name, e)
            await self.close()
            try:
                session = await self._ensure_session()
                result = await session.call_tool(tool_name, args)
            except Exception as e2:  # noqa: BLE001
                raise MCPError(f"{tool_name}: {type(e2).__name__}: {e2}") from e2

        # Server-side Tool-Error
        if getattr(result, "isError", False):
            content = _extract_text(result)
            raise MCPError(f"{tool_name}: {content}")

        return _unwrap_content(result)

    # Dynamic-Dispatch: mcp.create_task(...) ruft call("create_task", {...})
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        async def _dispatch(**kwargs):
            return await self.call(name, kwargs)

        # Hint wenn unbekanntes Tool — hilft bei typos
        if name not in KNOWN_TOOLS:
            log.debug("MCP-Client: tool %r nicht in KNOWN_TOOLS (typo? oder neu?)", name)
        _dispatch.__name__ = f"mcp_{name}"
        return _dispatch


def _extract_text(result: Any) -> str:
    """Holt den ersten TextContent.text aus einem CallToolResult."""
    content = getattr(result, "content", None) or []
    for c in content:
        text = getattr(c, "text", None)
        if text is not None:
            return text
    return ""


def _unwrap_content(result: Any) -> Any:
    """JSON-Auto-Unwrap. Falls der Server ein Tool-Result als JSON-String
    zurückgibt (typisch für FastMCP wenn Tool ein dict returnt), parsen
    wir das hier zurück. Sonst raw-str.
    """
    text = _extract_text(result)
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


# ─── Sync-Bridge ─────────────────────────────────────────────────────────────


class _MCPSyncBridge:
    """Sync-Wrapper für nicht-async Aufrufer (CLI-scripts, alte Code-Pfade).

    Macht für jeden Call ein neues asyncio.run() — d.h. NICHT sharing-fähig
    mit dem persistenten async-Client. Performance-akzeptabel für einzelne
    Calls (Setup ~50ms + HTTP ~100ms), aber für hot-paths bitte den async-
    Client direkt nutzen.

    DARF NICHT aus laufendem asyncio-Loop heraus aufgerufen werden —
    asyncio.run() würde dann RuntimeError raisen. Aus async-Code immer
    `await mcp.call(...)` verwenden.
    """

    def call(self, tool_name: str, args: dict | None = None) -> Any:
        async def _run():
            client = _MCPAsyncClient()
            try:
                return await client.call(tool_name, args)
            finally:
                await client.close()
        return asyncio.run(_run())

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def _dispatch(**kwargs):
            return self.call(name, kwargs)
        _dispatch.__name__ = f"mcp_sync_{name}"
        return _dispatch


# ─── Singletons ──────────────────────────────────────────────────────────────

# Default-Instances für den Bot. Import-once-use-everywhere.
mcp = _MCPAsyncClient()
mcp_sync = _MCPSyncBridge()


# ─── Health/Boot-Check ──────────────────────────────────────────────────────


async def health_check() -> dict[str, Any]:
    """Kurzer Verbindungstest beim Bot-Start. Loggt Tool-Count + maintain-Status.

    Returns: {"ok": bool, "tools_visible": int, "error"?: str}
    """
    try:
        session = await mcp._ensure_session()
        tools = await session.list_tools()
        n = len(tools.tools)
        log.info("MCP-Client: %d tools sichtbar via %s", n, MCP_URL)
        return {"ok": True, "tools_visible": n, "url": MCP_URL}
    except Exception as e:  # noqa: BLE001
        log.error("MCP-Client health-check failed: %s", e)
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "url": MCP_URL}


__all__ = ["mcp", "mcp_sync", "MCPError", "health_check", "KNOWN_TOOLS"]
