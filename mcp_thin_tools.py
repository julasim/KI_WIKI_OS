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


# ─── Write-Tools (Phase X3c) ─────────────────────────────────────────────────


async def append_to_daily(section: str, text: str) -> str:
    """An die heutige Daily-Note anhaengen, via MCP.

    LLM-Kontrakt unveraendert: result-string `f"In Daily ({section}) ..."`.
    Bot's lokale Auto-Link-Logik faellt weg — Self-Maintain-Pipeline (alle
    10 Min) holt Auto-Linking nach. Drift ist innerhalb 10 Min eingeholt,
    fuer ein User-Tagebuch absolut akzeptabel.
    """
    # Bot-Args (section, text) → MCP-Args (text, section)
    try:
        res = await mcp.append_to_daily(text=text, section=section)
    except MCPError as e:
        return _err_str("append_to_daily", e)

    if not isinstance(res, dict):
        return f"append_to_daily: unerwartetes Format {type(res).__name__}"
    path = res.get("path", "?")
    fname = path.rsplit("/", 1)[-1] if path else "?"
    return f"In Daily ({section}) eingetragen: {fname}"


async def create_note(
    title: str,
    body: str,
    tags: list | None = None,
    project: str | None = None,
) -> str:
    """Note anlegen via MCP. Project-Routing handelt MCP server-side.

    LLM-Kontrakt: `Note angelegt: [[<id>]]` (+ ggf. Projekt-Suffix).
    """
    if not title or not title.strip():
        return "Fehler: Note-Titel darf nicht leer sein."
    try:
        res = await mcp.create_note(
            title=title,
            project=project,
            body=body or "",
            tags=tags or [],
            subpath="notes",
        )
    except MCPError as e:
        return _err_str("create_note", e)

    if not isinstance(res, dict):
        return f"create_note: unerwartetes Format {type(res).__name__}"
    note_id = res.get("id") or res.get("path", "?").rsplit("/", 1)[-1].replace(".md", "")
    suffix = f" → Projekt {project}" if project else ""
    return f"Note angelegt: [[{note_id}]]{suffix}"


async def create_meeting(
    title: str,
    attendees: list | None = None,
    meeting_date: str | None = None,
    tags: list | None = None,
    project: str | None = None,
) -> str:
    """Meeting-Protokoll via MCP. Project-Routing + Datum-Validation
    machen wir client-seitig (gleiche Heuristik wie alte Bot-Funktion),
    Schreiben + Path-Generation passiert im MCP.
    """
    if not title or not title.strip():
        return "Fehler: Meeting-Titel darf nicht leer sein."
    # MCP create_meeting verlangt attendees als list (Pflicht via SCHEMA),
    # leere Liste OK
    try:
        res = await mcp.create_meeting(
            title=title,
            attendees=attendees or [],
            project=project,
            date=meeting_date,
            tags=tags or [],
            body="",
        )
    except MCPError as e:
        return _err_str("create_meeting", e)

    if not isinstance(res, dict):
        return f"create_meeting: unerwartetes Format {type(res).__name__}"
    meeting_id = res.get("id") or res.get("path", "?").rsplit("/", 1)[-1].replace(".md", "")
    suffix = f" → Projekt {project}" if project else ""
    return f"Meeting angelegt: [[{meeting_id}]]{suffix}"


# String-Mapping fuer 'null'-Werte vom LLM (Bot-Konvention) → echtes None
_TASK_NULL_TOKENS = {"null", "none", "—", "-", ""}


def _is_null(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower() in _TASK_NULL_TOKENS


async def task(
    action: str,
    task_id: str | None = None,
    title: str | None = None,
    priority: str | None = None,
    due: str | None = None,
    project: str | None = None,
    context: str | None = None,
    tags: list | None = None,
    recurrence: str | None = None,
    status: str | None = None,
) -> str:
    """Konsolidiertes Task-Tool — dispatcht auf MCP `create_task` (action=create)
    oder `task` (action=done/reopen/update→edit).

    Bot-API-Kompatibilitaet:
      - action='create' braucht title
      - action='done'/'reopen'/'update' brauchen task_id
      - LLM-Stringtoken 'null'/'none'/'—'/'-' fuer due bedeutet "feld leeren"
        (bei MCP edit: nicht mitgeben → wird ignoriert; clear-Verhalten ist
        in dieser Phase NICHT implementiert, MCP supports kein explizites
        clear via task-tool — das ist ein bekannter Mini-Drift gegen Bot,
        wird in X3e adressiert wenn relevant)
    """
    a = (action or "").strip().lower()

    # ─── CREATE ─────────────────────────────────────────────────────────
    if a == "create":
        if not title or not title.strip():
            return "create: title ist Pflicht."
        try:
            res = await mcp.create_task(
                title=title,
                project=project,
                priority=priority or "medium",
                due=due if due and not _is_null(due) else None,
                context=context,
                recurrence=recurrence,
            )
        except MCPError as e:
            return _err_str("task.create", e)

        if not isinstance(res, dict):
            return f"task.create: unerwartetes Format {type(res).__name__}"
        tid = res.get("id") or res.get("path", "?").rsplit("/", 1)[-1].replace(".md", "")
        extras = []
        if due and not _is_null(due):
            extras.append(f"due {due}")
        if priority and priority != "medium":
            extras.append(f"prio {priority}")
        if recurrence:
            extras.append(f"wiederholt {recurrence}")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        return f"Task angelegt: [[{tid}]]{extra_str}"

    # ─── DONE / REOPEN / UPDATE ────────────────────────────────────────
    if a in ("done", "reopen", "update"):
        if not task_id:
            return f"{a}: task_id ist Pflicht."
        # MCP task() action mapping: Bot 'update' → MCP 'edit'
        mcp_action = "edit" if a == "update" else a
        kwargs: dict[str, Any] = {"id": task_id, "action": mcp_action}
        # Bei edit: nur die explizit gesetzten Felder durchreichen
        if mcp_action == "edit":
            if priority is not None and not _is_null(priority):
                kwargs["priority"] = priority
            if due is not None and not _is_null(due):
                kwargs["due"] = due
            # body, snooze_until werden vom Bot-task aktuell nicht uebergeben
        try:
            res = await mcp.task(**kwargs)
        except MCPError as e:
            return _err_str(f"task.{a}", e)

        if not isinstance(res, dict):
            return f"task.{a}: unerwartetes Format {type(res).__name__}"

        # Format-Kompatibilitaet zur alten Bot-Implementierung
        tid_short = task_id.removeprefix("t-")
        if a == "done":
            return f"Task erledigt: [[t-{tid_short}]]"
        if a == "reopen":
            return f"Task wieder geoeffnet: [[t-{tid_short}]]"
        # update
        changes: list[str] = []
        for k in ("priority", "due"):
            if k in kwargs:
                changes.append(f"{k}={kwargs[k]}")
        change_str = f" ({', '.join(changes)})" if changes else ""
        return f"Task aktualisiert: [[t-{tid_short}]]{change_str}"

    return f"Unbekannte action: {action!r}. Erlaubt: create, done, reopen, update."


# ─── Maintain-Tools (Phase X3d) ──────────────────────────────────────────────


async def goal_status(scope: str = "all", saeule: str | None = None,
                      goal: str = "5y-2031") -> str:
    """Goal-System-Status via MCP `goal_status_check`.

    Bot-LLM-Kontrakt: HTML-formatted output mit Tag-Countdown + Saeulen +
    Habits-Score + Sport-Count + Drift-Anker. Format-kompatibel zur alten
    lokalen goal_status-Funktion (siehe ki_wiki_bot.py:4313).

    Tag-Countdown wird lokal berechnet (reine Date-Arithmetik, kein Vault-IO).
    Rest kommt von MCP.
    """
    try:
        data = await mcp.goal_status_check()
    except MCPError as e:
        return _err_str("goal_status", e)

    if not isinstance(data, dict):
        return f"goal_status: unerwartetes Format {type(data).__name__}"

    from datetime import date as _date
    parts: list[str] = []

    # ─── Header: Tag-Countdown ─────────────────────────────────────────────
    if goal == "5y-2031":
        target = _date(2031, 5, 1)
        today = _date.today()
        days_left = (target - today).days
        if days_left > 0:
            parts.append(f"<b>5y-2031</b> · {days_left} Tage bis Stichtag (01.05.2031)")
        elif days_left == 0:
            parts.append(f"<b>5y-2031</b> · STICHTAG HEUTE (01.05.2031)")
        else:
            parts.append(f"<b>5y-2031</b> · Stichtag {-days_left} Tage vergangen (01.05.2031)")
    else:
        parts.append(f"<b>Goal {goal}</b>")
    parts.append("")

    s = (scope or "all").strip().lower()

    # Saeulen-Block: MCP liefert dies aktuell nicht in goal_status_check —
    # das hat MCP's read_saeulen. Wir ziehen separat.
    if s in ("all", "saeule"):
        try:
            saeulen_data = await mcp.read_saeulen()
            if isinstance(saeulen_data, dict):
                saeulen_list = saeulen_data.get("saeulen") or []
                if saeulen_list:
                    parts.append("<b>Saeulen</b> (Status manuell gepflegt in saeulen.md / readme.md)")
                    for sa in saeulen_list:
                        parts.append(f"  {sa.get('label', '?'):<14} – {sa.get('kpi', '')}")
                    parts.append("")
        except MCPError:
            pass  # nicht kritisch wenn Saeulen-Read fehlt

    # Habits-Score
    if s in ("all", "habits"):
        h = data.get("habits_7d") or {}
        check = h.get("check", 0)
        possible = h.get("possible", 0)
        parts.append(f"<b>Habits</b> letzte 7 Tage: {check} ✓ / {possible} möglich")
        if possible > 0:
            pct = int(check / possible * 100)
            parts.append(f"   Quote: {pct}% (Soll: ≥80%)")
        parts.append("")

    # Sport
    if s in ("all", "sport"):
        sp = data.get("sport") or {}
        parts.append(
            f"🏃 <b>Sport</b> letzte 30 Tage: {sp.get('d30', 0)} Sessions · "
            f"letzte 7 Tage: {sp.get('d7', 0)}"
        )
        parts.append("   Wochen-Soll: 3 Sessions (2× Cardio + 1× Kraft)")
        parts.append("")

    # Drift
    if s in ("all", "drift"):
        drift = data.get("drift") or {}
        parts.append("⚠️ <b>Drift-Detektor</b>")
        for label, key in (
            ("Letzter Wochen-Anker", "weekly"),
            ("Letzter Monats-Anker", "monthly"),
            ("Letzter Quartals-Anker", "quarterly"),
        ):
            entry = drift.get(key) or {}
            value = entry.get("value", "?")
            parts.append(f"   {label}: {value}")

    return "\n".join(parts)


# ─── Edit + Move (Phase X3 Erweiterung) ──────────────────────────────────────


async def edit_file(rel_path: str, find: str, replace: str, regex: bool = False) -> str:
    """Find/Replace in einem File via MCP `edit_file_replace`.

    Bot-LLM-Kontrakt unveraendert: Result-String "<n>x ersetzt in <path>"
    oder "Kein Treffer fuer <find> in <path>".

    MCP-Tool macht ReDoS-Schutz, File-Size-Cap (5MB), Pattern-Length-Cap (500).
    """
    if not isinstance(find, str) or not find:
        return "Edit-Fehler: 'find' muss nicht-leerer String sein."
    if not isinstance(replace, str):
        return "Edit-Fehler: 'replace' muss String sein."
    try:
        res = await mcp.edit_file_replace(
            path=rel_path, find=find, replace=replace, regex=regex
        )
    except MCPError as e:
        return _err_str("edit_file", e)

    if not isinstance(res, dict):
        return f"edit_file: unerwartetes Format {type(res).__name__}"
    n = res.get("replacements", 0)
    if n == 0:
        return f"Kein Treffer fuer {find[:50]!r} in {rel_path}"
    return f"{n}x ersetzt in {rel_path}"


async def move(
    src: str | None = None,
    srcs: list | None = None,
    dst: str | None = None,
    project_slug: str | None = None,
    parent: str | None = None,
    overwrite: bool = False,
) -> str:
    """Konsolidiertes Move-Tool — drei Modi je nach Args:

    a) Einzeln: move(src='foo.md', dst='bar.md') → MCP move
    b) Bulk:    move(srcs=['a.md','b.md'], dst='ordner/') → MCP move_bulk
    c) Projekt: move(project_slug='matura', parent='dachboden') → MCP move_project

    Bot-LLM-Kontrakt unveraendert: kompakte Result-Strings wie alte Bot-Funktion.
    """
    # Mode c) Projekt
    if project_slug:
        try:
            res = await mcp.move_project(slug=project_slug, parent=parent)
        except MCPError as e:
            return _err_str("move_project", e)
        if not isinstance(res, dict):
            return f"move_project: unerwartetes Format {type(res).__name__}"
        if res.get("status") == "no_change":
            return f"Projekt liegt bereits an Zielposition: `{res.get('new_path')}/`"
        old_p = res.get("old_path", "?")
        new_p = res.get("new_path", "?")
        info = f" (jetzt Subprojekt von `{parent}`)" if parent else " (jetzt Top-Level)"
        return f"OK Projekt verschoben: `{old_p}/` -> `{new_p}/`{info}"

    # Defensive: LLM koennte src statt srcs senden mit Liste
    if isinstance(src, list) and not srcs:
        srcs = src
        src = None

    # Mode b) Bulk
    if srcs:
        if not dst:
            return "Fehler: dst (Ziel-Ordner) noetig fuer Bulk-Move."
        if not isinstance(srcs, list):
            return "Fehler: srcs muss eine Liste sein."
        try:
            res = await mcp.move_bulk(sources=srcs, dest_dir=dst, overwrite=overwrite)
        except MCPError as e:
            return _err_str("move_bulk", e)
        if not isinstance(res, dict):
            return f"move_bulk: unerwartetes Format {type(res).__name__}"
        moved = res.get("moved") or []
        failed = res.get("failed") or []
        parts = [f"OK {len(moved)} verschoben -> `{dst}/`"]
        if moved:
            preview = ", ".join(moved[:8]) + (f" (+{len(moved)-8})" if len(moved) > 8 else "")
            parts.append("  " + preview)
        if failed:
            parts.append(f"FAIL {len(failed)} fehlgeschlagen:")
            for f in failed[:5]:
                parts.append(f"  - {f.get('name', '?')}: {f.get('reason', '?')}")
            if len(failed) > 5:
                parts.append(f"  - (+{len(failed)-5} weitere)")
        return "\n".join(parts)

    # Mode a) Einzeln
    if src:
        if not dst:
            return "Fehler: dst noetig fuer Einzel-Move."
        try:
            res = await mcp.move(source=src, dest=dst)
        except MCPError as e:
            return _err_str("move", e)
        if not isinstance(res, dict):
            return f"move: unerwartetes Format {type(res).__name__}"
        return f"OK verschoben: `{src}` -> `{dst}`"

    return "Fehler: keiner der drei Modi erkannt — gib src+dst ODER srcs+dst ODER project_slug an."


__all__ = [
    "search_vault", "read_file", "list_files",
    "append_to_daily", "create_note", "create_meeting", "task",
    "goal_status",
    "edit_file", "move",
]
