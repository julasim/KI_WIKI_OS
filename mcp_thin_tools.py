"""KI-OS Bot — MCP-Thin-Tool-Wrappers.

Phase X3 (Bot wird thin client): jede LLM-exposed Tool-Funktion ruft jetzt
das MCP statt lokal mit `frontmatter` + Filesystem zu arbeiten. Single
Source of Truth bleibt der MCP-Server.

Jeder Wrapper ist:
  - **async** (mcp_client ist async, wir awaiten direkt — der angepasste
    Tool-Dispatch in `llm_loop` erkennt das via iscoroutinefunction).
  - **format-kompatibel**: gibt einen LLM-tauglichen STRING zurück, exakt
    so wie die alten Bot-internen Tools — der LLM weiss aus seinem
    System-Prompt-Vertrag was er erwartet.
  - **fehlertolerant**: MCPError → freundliche Fehlermeldung statt Crash.

Dieses File ersetzt nach und nach die lokalen Implementierungen in
`ki_wiki_bot.py`. Sobald X3b/c/d komplett ist, kann der lokale Code
gelöscht werden (Phase X3e).
"""

from __future__ import annotations

import logging
from typing import Any

from mcp_client import MCPError, mcp

log = logging.getLogger("ki-os-bot.mcp_thin_tools")


# ─── Helper: Result-Format ───────────────────────────────────────────────────


def _err_str(tool: str, e: Exception) -> str:
    """Formatiert MCP-Errors für den LLM. Gleiches Format wie alte Tools
    (mit "Fehler:" am Anfang damit consecutive_failures-Heuristik greift).
    """
    return f"Fehler bei {tool}: {type(e).__name__}: {e}"


# ─── Read-Tools (Phase X3b) ──────────────────────────────────────────────────


async def search_vault(query: str, limit: int = 5) -> str:
    """Volltext-Suche via MCP `search_vault`.

    Format-Kompatibilität mit alter Bot-Implementierung:
      "Suche '<q>' — N Treffer (top L):
       - [[id]] · `type` · score X · `path`"
    """
    try:
        res = await mcp.search_vault(query=query, max_results=limit)
    except MCPError as e:
        return _err_str("search_vault", e)

    if not isinstance(res, dict):
        return f"Unerwartetes Result-Format: {type(res).__name__}"

    hits = res.get("hits") or []
    total = res.get("total", len(hits))
    if not hits:
        return f"Keine Treffer für: {query}"

    lines = [f"Suche '{query}' — {total} Treffer (top {limit}):"]
    for hit in hits[:limit]:
        # MCP search_vault liefert {path, line, match, context}
        # Bot-Erwartung war {id, type, score, path} — wir nähern an
        path = hit.get("path") or "?"
        line_no = hit.get("line", "?")
        match_text = (hit.get("match") or "")[:120]
        lines.append(f"- `{path}` (L{line_no}): {match_text}")
    return "\n".join(lines)


async def read_file(rel_path: str, strip_frontmatter: bool = True) -> str:
    """Read-File via MCP. Capped auf 8KB wie alte Bot-Implementierung.

    strip_frontmatter=True (default): YAML-FM wird entfernt.
    """
    try:
        res = await mcp.read_file(path=rel_path)
    except MCPError as e:
        return _err_str("read_file", e)

    if not isinstance(res, dict):
        return f"Unerwartetes Result-Format: {type(res).__name__}"

    # MCP read_file liefert {path, frontmatter, content}
    content = res.get("content") or ""
    if not strip_frontmatter:
        # Ggf. FM rekonstruieren als Header
        fm = res.get("frontmatter")
        if fm:
            import yaml  # type: ignore
            content = "---\n" + yaml.safe_dump(fm, allow_unicode=True, sort_keys=True) + "---\n" + content
    return content[:8000]


async def list_files(rel_dir: str = "", include_system: bool = False) -> str:
    """Liste .md-Files via MCP `list_files`.

    include_system: ohne diesen Flag werden Templates/Meta/Trash/Archive raus-
    gefiltert (gleiche Heuristik wie alte Bot-Implementierung — Filter passiert
    client-seitig weil MCP keinen include_system-Parameter hat).
    """
    try:
        res = await mcp.list_files(path=rel_dir)
    except MCPError as e:
        return _err_str("list_files", e)

    if not isinstance(res, dict):
        return f"Unerwartetes Result-Format: {type(res).__name__}"

    entries = res.get("entries") or []
    # MCP list_files ist NICHT recursive — wir bauen recursive-Ansicht client-side
    # Für die Bot-Erwartung ist eine flat-Liste ausreichend (LLM sieht's als
    # "ist da was drin"-Hinweis und ruft dann gezielt deeper).

    # System-Filter (Bot-konvention)
    NOISE_DIRS = {"08_Templates", "06_Meta", "07_Tools", ".trash", "99_Archive", ".obsidian"}
    NOISE_FILES = {"CLAUDE.md", "MOC.md", "SCHEMA.md", "PIPELINES.md", "COMMANDS.md"}

    files: list[str] = []
    dirs: list[str] = []
    for e in entries:
        name = e.get("name", "")
        kind = e.get("kind", "")
        if not include_system:
            if name in NOISE_FILES:
                continue
            if name in NOISE_DIRS:
                continue
        if kind == "dir":
            dirs.append(f"{name}/")
        elif kind == "file" and name.endswith(".md"):
            files.append(name)

    base_label = rel_dir or "Vault-Root"
    if not files and not dirs:
        return f"Keine User-Files in {base_label} (System-Files via include_system=true sichtbar)."

    parts = []
    if dirs:
        parts.append(f"Ordner ({len(dirs)}):\n" + "\n".join(f"• `{d}`" for d in sorted(dirs)))
    if files:
        parts.append(f"Files ({len(files)}):\n" + "\n".join(f"• `{f}`" for f in sorted(files)))
    return f"Inhalt von {base_label}:\n\n" + "\n\n".join(parts)


__all__ = ["search_vault", "read_file", "list_files"]
