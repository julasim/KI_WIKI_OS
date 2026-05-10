#!/usr/bin/env python3
"""
ki_wiki_bot.py — Telegram-Bot fuer KI_WIKI_Vault (Bot v2 — thin client).

Vier Layer:
  • Telegram (Polling, Auth, Commands, HTML-Renderer)
  • Whisper (Voice → Text, lokal)
  • LLM-API (Anthropic Claude / OpenAI-kompat, llm_loop, History)
  • MCP-Connection (alle Vault-Operationen via mcp_thin_tools)

ENV:
  TG_TOKEN, ALLOWED_USER_ID, LLM_API_KEY,
  VAULT_PATH (default /vault — nur fuer Bot-Memory in 06_Meta/bot-memory/),
  MCP_TOKEN, MCP_URL (default https://wiki-mcp.sima.business/mcp/),
  LLM_MODEL, WHISPER_MODEL/DEVICE/LANG.
"""

import os
import re
import json
import time
import asyncio
import logging
import tempfile
import threading  # fuer _REMINDERS_LOCK
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import frontmatter
from openai import OpenAI
from faster_whisper import WhisperModel
from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# Phase X3: alle LLM-exposed Vault-Operationen laufen via MCP-Server.
# Bei import-fail bricht Bot beim Boot ab — bewusst hart, weil ohne MCP
# 80% der LLM-Tools nicht funktionieren wuerden (kein silent-degradation).
import mcp_thin_tools
import mcp_client  # noqa: F401  — ensure session-singleton importiert

# ============================================================================
# Config
# ============================================================================

VAULT = Path(os.environ.get("VAULT_PATH", "/vault"))
# 0 = Setup-Modus: Bot meldet beim ersten Kontakt die User-ID,
#                  user editiert .env und startet neu.
ALLOWED_USER_ID = int(os.environ.get("ALLOWED_USER_ID", "0") or "0")
TG_TOKEN = os.environ["TG_TOKEN"]

# LLM-Provider — beliebige OpenAI-API-kompatible Endpoints:
#   OpenRouter      → https://openrouter.ai/api/v1
#   Ollama Cloud    → https://ollama.com/v1
#   OpenAI direkt   → https://api.openai.com/v1
#   Lokales Ollama  → http://ollama:11434/v1
# OPENROUTER_API_KEY bleibt als Fallback für bestehende Installs.
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY (oder OPENROUTER_API_KEY) muss gesetzt sein.")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.anthropic.com/v1/")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")
# Provider-Detection für Format-Kompatibilität:
# - Anthropic-direkt + OpenRouter→Anthropic: cache_control + content-as-list erlaubt
# - Gemini OpenAI-Compat + reine OpenAI: content muss String sein, kein cache_control
# - Ollama: content muss String sein
_LLM_BASE_LOWER = LLM_BASE_URL.lower()
_LLM_MODEL_LOWER = LLM_MODEL.lower()
USE_ANTHROPIC_CACHE = (
    "anthropic" in _LLM_BASE_LOWER
    or ("openrouter" in _LLM_BASE_LOWER and _LLM_MODEL_LOWER.startswith("anthropic/"))
)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "de")

# Daily-Briefing
try:
    BRIEFING_HOUR = int(os.environ.get("BRIEFING_HOUR", "0") or "0")
except ValueError:
    BRIEFING_HOUR = 0
# SUGGESTION_HOUR entfernt 2026-05-03 — Memory-Vorschläge-Job ist gestrichen,
# Funktion bleibt für manuelle Trigger ('memory N M') verfügbar.
TIMEZONE = ZoneInfo(os.environ.get("TIMEZONE", "Europe/Vienna"))

TG_MAX_MESSAGE = 3800

LIFE = VAULT / "10_Life"
DAILY_DIR = LIFE / "daily"
TASKS_DIR = LIFE / "tasks"
NOTES_DIR = LIFE / "notes"
MEETINGS_DIR = LIFE / "meetings"
PROJECTS_DIR = VAULT / "05_Projects"  # Schema-konform: Projekte sind eigener Top-Level
# AREAS_DIR entfernt 2026-05-03 — Areas-Konzept wurde nie genutzt, area-Feld ist
# tote Logik (kein Folder, kein Schema-Eintrag). project + tags decken alle Use-Cases.
TEMPLATES_DIR = VAULT / "08_Templates"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ki_wiki_bot")


# ─── Log-Sanitization: TG_TOKEN aus HTTPX-INFO-Logs maskieren ───────────────
# python-telegram-bot embeddet den Token in die HTTP-URL. httpx loggt URLs
# bei INFO-Level → Token leakt in Log-Files / Container-Output / Bug-Reports.
# Filter wandelt /bot<TOKEN>/method → /bot[REDACTED]/method.
class _TokenMaskingFilter(logging.Filter):
    _TG_TOKEN_RE = re.compile(r"/bot\d{7,15}:[A-Za-z0-9_\-]{30,}/")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            if "/bot" in msg:
                masked = self._TG_TOKEN_RE.sub("/bot[REDACTED]/", msg)
                if masked != msg:
                    record.msg = masked
                    record.args = ()
        except Exception:
            pass
        return True


# Auf root-Handler hängen → fängt alle Logger inkl. httpx + telegram.ext
for _h in logging.getLogger().handlers:
    _h.addFilter(_TokenMaskingFilter())

# ============================================================================
# LLM client (OpenRouter, OpenAI-API-kompatibel)
# ============================================================================

llm = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/julasim/KI_WIKI_OS",
        "X-Title": "KI Wiki Bot",
    },
)

# ============================================================================
# Whisper (einmal beim Start laden, im Speicher halten)
# ============================================================================

log.info(f"Loading Whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}...")
whisper = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type="int8")
log.info("Whisper loaded.")


# ============================================================================
# Bot-State Helper (NUR fuer 06_Meta/bot-memory/ + reminders.json — KEIN Vault-Content)
# ============================================================================

def today_iso() -> str:
    """Heute als ISO-String in BOT-Lokalzeit (Europe/Vienna o.ä.)."""
    return datetime.now(TIMEZONE).date().isoformat()


def atomic_write(path: Path, content: str) -> None:
    """Atomic-write via tmp+rename. NUR fuer Bot-State-Files (06_Meta/bot-memory/,
    reminders.json, pending-*.json). Vault-Content wird ueber MCP geschrieben.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

# ============================================================================
# Helpers
# ============================================================================

def today_iso() -> str:
    """Heute als ISO-String in BOT-Lokalzeit (Europe/Vienna o.ä.).

    KRITISCH: nutzt TIMEZONE statt date.today(). Container läuft typisch
    UTC — `date.today()` liefert nach 22:00 Wien-Zeit schon morgen.
    Folge wäre: Tasks landen in falscher Daily.
    """
    return datetime.now(TIMEZONE).date().isoformat()


def extract_docx_text(docx_path: Path) -> tuple:
    """Extrahiert Text + Tabellen + Metadaten aus .docx via python-docx.

    Tabellen werden als Markdown-Tabellen gerendert (für Volltext-Suche
    + Lesbarkeit im .md-Wrapper).

    Returns: (text, metadata_dict, paragraph_count)
    metadata_dict hat title, author, subject (alle optional).
    """
    try:
        import docx  # type: ignore  # python-docx
    except ImportError:
        return "(python-docx nicht installiert)", {}, 0

    try:
        doc = docx.Document(str(docx_path))

        # Body: Paragraphs + Tabellen IN REIHENFOLGE der Doc-Struktur.
        # docx hat doc.paragraphs und doc.tables separat — Reihenfolge geht
        # verloren wenn man sie einzeln iteriert. Lösung: über doc.element.body.
        from docx.oxml.ns import qn  # type: ignore
        text_parts = []
        body = doc.element.body
        for child in body.iterchildren():
            tag = child.tag
            if tag == qn("w:p"):  # Paragraph
                # Text-Inhalt sammeln (alle text-runs)
                runs = child.findall(".//" + qn("w:t"))
                line = "".join(r.text or "" for r in runs).strip()
                if line:
                    text_parts.append(line)
            elif tag == qn("w:tbl"):  # Tabelle
                # Markdown-Tabelle bauen
                rows_text = []
                for row in child.findall(qn("w:tr")):
                    cells = []
                    for cell in row.findall(qn("w:tc")):
                        cell_runs = cell.findall(".//" + qn("w:t"))
                        cell_text = "".join(r.text or "" for r in cell_runs).strip()
                        # Pipes in Zellen escapen
                        cells.append(cell_text.replace("|", "\\|").replace("\n", " "))
                    rows_text.append(cells)
                if rows_text:
                    n_cols = max(len(r) for r in rows_text)
                    rows_text = [r + [""] * (n_cols - len(r)) for r in rows_text]
                    md_table = ["| " + " | ".join(rows_text[0]) + " |",
                                "|" + "|".join(["---"] * n_cols) + "|"]
                    for r in rows_text[1:]:
                        md_table.append("| " + " | ".join(r) + " |")
                    text_parts.append("\n" + "\n".join(md_table) + "\n")

        # Core-Metadaten
        cp = doc.core_properties
        meta = {
            "title": (cp.title or "").strip(),
            "author": (cp.author or "").strip(),
            "subject": (cp.subject or "").strip(),
        }
        para_count = sum(1 for c in body.iterchildren() if c.tag == qn("w:p"))

        text = "\n\n".join(text_parts).strip()
        return (text or "(kein Text extrahiert)", meta, para_count)
    except Exception as e:
        log.exception(f"DOCX extract failed for {docx_path}")
        return f"(Extraktions-Fehler: {e})", {}, 0


def extract_pdf_text(pdf_path: Path, max_pages: int = 200) -> tuple:
    """Extrahiert Text + Metadaten aus PDF via pymupdf.

    Returns: (text, metadata_dict, total_pages)
    metadata_dict hat title, author, subject, keywords (alle optional).
    Bei sehr großen PDFs werden nur die ersten max_pages extrahiert.
    """
    try:
        import pymupdf  # type: ignore
    except ImportError:
        return "(pymupdf nicht installiert)", {}, 0

    try:
        doc = pymupdf.open(str(pdf_path))
        total = doc.page_count
        pages_to_read = min(total, max_pages)

        text_parts = []
        for i in range(pages_to_read):
            page_text = doc[i].get_text("text")
            if page_text.strip():
                text_parts.append(f"\n## Seite {i+1}\n\n{page_text.strip()}")

        if total > max_pages:
            text_parts.append(
                f"\n\n_(PDF hat {total} Seiten, nur die ersten {max_pages} extrahiert)_"
            )

        meta_raw = doc.metadata or {}
        meta = {k: (meta_raw.get(k) or "").strip() for k in ("title", "author", "subject", "keywords")}
        doc.close()

        return ("\n".join(text_parts).strip() or "(kein Text extrahiert)", meta, total)
    except Exception as e:
        log.exception(f"PDF extract failed for {pdf_path}")
        return f"(Extraktions-Fehler: {e})", {}, 0


# ============================================================================
# Tool implementations
# ============================================================================

VALID_SECTIONS = {"Heute", "Notizen & Gedanken", "Offen / Einsortieren", "Abends"}

# Auto-Link entfernt (Phase X3 Cleanup) — laeuft jetzt in MCP-Maintain-Pipeline.


# ─── Task-Helpers entfernt (Phase X3 Cleanup) ──────────────────────────
# Lokale Task-Implementierungen (_task_create/done/reopen/update,
# _normalize_priority/recurrence/due, _resolve_task_path, _sync_task_body,
# VALID_PRIORITIES, VALID_TASK_CONTEXTS, VALID_TASK_RECURRENCE, _PRIORITY_MAP,
# _RECURRENCE_MAP, _TASK_CLEAR, lokale task()) wurden entfernt.
# LLM-Tool 'task' laeuft jetzt komplett via mcp_thin_tools.task() → MCP.



# ─── Tagesplanung: Listings + Agenda ────────────────────────────────────────

# Priority-Sortierung + Symbole (zentral — vermeidet Drift zwischen
# _format_task_line und compute_briefing wo vorher unterschiedliche
# Symbol-Maps definiert waren)
_PRIO_ORDER = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
# Symbole — Voll-Farbskala (User-Spec 2026-05-03):
# 🔴 urgent, 🟠 high, 🟡 medium, 🟢 low
# Klare Ampel-Hierarchie für schnelle Visual-Erkennung in Telegram + Dashboard.
PRIO_SYMBOLS = {"urgent": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}


_TODAY_DATA_TTL_SEC = 30


# ─── create_meeting entfernt (Phase X3 Cleanup) ──────────────────────────
# LLM-Tool 'create_meeting' laeuft via mcp_thin_tools.create_meeting() → MCP.


# ─── create_note entfernt (Phase X3 Cleanup) ──────────────────────────
# LLM-Tool 'create_note' laeuft via mcp_thin_tools.create_note() → MCP.


# ─── search_vault entfernt (Phase X3 Cleanup) ──────────────────────────
# LLM-Tool 'search_vault' laeuft via mcp_thin_tools.search_vault() → MCP.


# ─── read_file entfernt (Phase X3 Cleanup) ──────────────────────────
# LLM-Tool 'read_file' laeuft via mcp_thin_tools.read_file() → MCP.


# ─── move_path entfernt (Phase X3 thin-client) ───────────────────
# Funktion 'move_path' laeuft via MCP / mcp_thin_tools.


# ─── move_paths entfernt (Phase X3 thin-client) ───────────────────
# Funktion 'move_paths' laeuft via MCP / mcp_thin_tools.


# ─── move_project entfernt (Phase X3 thin-client) ───────────────────
# Funktion 'move_project' laeuft via MCP / mcp_thin_tools.


# ─── move entfernt (Phase X3 thin-client) ───────────────────
# Funktion 'move' laeuft via MCP / mcp_thin_tools.




# ─── edit_file entfernt (Phase X3 thin-client) ───────────────────
# Funktion 'edit_file' laeuft via MCP / mcp_thin_tools.


CLIP_URL_TIMEOUT = 15  # Sekunden — verhindert dass slowloris-Server den Bot hängen lassen


# ─── Pending Deletions (Multi-File-fähig + Soft/Hard) ───────────────────────
# Two-Step-Delete: request_delete() merkt sich Pfad(e) + Modus,
# confirm_delete() führt aus.
# Modi: 'archive' (Default, sicher) oder 'permanent' (echtes rm).
PENDING_DELETIONS: dict[int, tuple[list[str], float, str]] = {}  # uid → (paths, ts, mode)
DELETE_CONFIRM_TIMEOUT = 300  # Sekunden


# ─── Reminders (persistent + JobQueue) ──────────────────────────────────────
# Reminders überleben Bot-Restart: JSON in 06_Meta/reminders.json.
# Bei Startup werden alle aktiven Reminders neu in die JobQueue eingehängt.
REMINDERS_FILE = VAULT / "06_Meta" / "reminders.json"
# System-Default-Tagebuch-Reminder (am 01.05.2026 angelegt). Stable-ID damit
# reminder_callback ihn auch nach Text-Änderungen sicher erkennt → triggert
# pending_diary-Bypass für direkten Daily-Note-Append.
DIARY_REMINDER_ID = "rem-20260501-110251-676"
ACTIVE_REMINDER_JOBS: dict = {}  # id → telegram.ext.Job
BOT_APP = None  # wird in main() gesetzt — brauchen Zugriff auf job_queue von Tools aus

# Lock gegen Race zwischen reminder_callback (entfernt einmaligen Reminder)
# und cancel_reminder/create_reminder (mutieren ebenfalls reminders.json).
# Last-write-wins ohne Lock kann Mutationen verschlucken.
_REMINDERS_LOCK = threading.Lock()


def _load_reminders() -> list:
    if not REMINDERS_FILE.exists():
        return []
    try:
        return json.loads(REMINDERS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"reminders.json kaputt: {e}")
        return []


def _save_reminders(reminders: list) -> None:
    # atomic_write macht intern tmp+rename — verhindert halbgeschriebene JSON
    atomic_write(
        REMINDERS_FILE,
        json.dumps(reminders, indent=2, ensure_ascii=False),
    )


# Reminder-Recurrence (kein monthly — Reminders sind zeitpunkt-basiert,
# monthly würde unklar bei "31. + Februar"-Edge-Cases). Tasks haben eigene Konstante.
VALID_REMINDER_RECURRENCE = {None, "", "daily", "weekly", "weekdays"}


def create_reminder(when_iso: str, message: str, recurrence: Optional[str] = None) -> str:
    """Setzt eine Erinnerung. Wird zur angegebenen Zeit als Telegram-Nachricht geschickt.

    when_iso: ISO-Datetime YYYY-MM-DDTHH:MM:SS (Lokalzeit Europe/Vienna)
    recurrence: null/leer = einmalig, "daily" = täglich, "weekdays" = Mo-Fr,
                "weekly" = einmal pro Woche (gleicher Wochentag wie der erste Trigger)
    """
    if not message or not message.strip():
        return "Erinnerung-Text fehlt."
    try:
        when_dt = datetime.fromisoformat(when_iso)
    except ValueError:
        return f"Ungültiges Datum/Zeit: {when_iso}. Format: 2026-04-26T15:00:00"

    if when_dt.tzinfo is None:
        when_dt = when_dt.replace(tzinfo=TIMEZONE)

    rec = (recurrence or None) if recurrence else None
    if rec not in VALID_REMINDER_RECURRENCE:
        return f"Ungültige Recurrence '{recurrence}'. Erlaubt: leer / daily / weekly / weekdays"

    # Ein-shot Reminder in der Vergangenheit ablehnen (außer recurring)
    if not rec and when_dt < datetime.now(TIMEZONE):
        return f"Zeitpunkt liegt in der Vergangenheit: {when_dt.isoformat(timespec='minutes')}"

    rid = "rem-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    reminder = {
        "id": rid,
        "fire_at": when_dt.isoformat(timespec="seconds"),
        "message": message.strip(),
        "recurrence": rec,
        "created": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
    }
    with _REMINDERS_LOCK:
        reminders = _load_reminders()
        reminders.append(reminder)
        _save_reminders(reminders)

    if BOT_APP is not None:
        _schedule_reminder(BOT_APP, reminder)

    rec_str = f", wiederholt {rec}" if rec else ""
    when_str = when_dt.strftime("%a %d.%m. %H:%M")
    return f"⏰ Erinnerung gesetzt: {when_str}{rec_str}\n→ \"{message.strip()[:80]}\""


def _schedule_reminder(app, reminder: dict) -> None:
    """Hängt einen Reminder in die JobQueue ein."""
    when_dt = datetime.fromisoformat(reminder["fire_at"])
    if when_dt.tzinfo is None:
        when_dt = when_dt.replace(tzinfo=TIMEZONE)

    rid = reminder["id"]
    rec = reminder.get("recurrence")
    job_data = {"id": rid, "message": reminder["message"]}

    try:
        if rec == "daily":
            job = app.job_queue.run_daily(
                reminder_callback,
                time=when_dt.timetz(),
                data=job_data,
                name=f"reminder-{rid}",
            )
        elif rec == "weekly":
            job = app.job_queue.run_daily(
                reminder_callback,
                time=when_dt.timetz(),
                days=(when_dt.weekday(),),
                data=job_data,
                name=f"reminder-{rid}",
            )
        elif rec == "weekdays":
            job = app.job_queue.run_daily(
                reminder_callback,
                time=when_dt.timetz(),
                days=(0, 1, 2, 3, 4),
                data=job_data,
                name=f"reminder-{rid}",
            )
        else:
            # Einmalig
            if when_dt <= datetime.now(TIMEZONE):
                log.info(f"Reminder {rid} liegt in Vergangenheit, skip.")
                # aus JSON entfernen
                _remove_reminder_from_json(rid)
                return
            job = app.job_queue.run_once(
                reminder_callback,
                when=when_dt,
                data=job_data,
                name=f"reminder-{rid}",
            )
        ACTIVE_REMINDER_JOBS[rid] = job
        log.info(f"Reminder scheduled: {rid} @ {when_dt.isoformat()} rec={rec}")
    except Exception as e:
        log.exception(f"Reminder-Schedule fehlgeschlagen für {rid}")


def _remove_reminder_from_json(rid: str) -> None:
    with _REMINDERS_LOCK:
        reminders = _load_reminders()
        reminders = [r for r in reminders if r["id"] != rid]
        _save_reminders(reminders)


async def reminder_callback(ctx: ContextTypes.DEFAULT_TYPE):
    """Wird von JobQueue ausgelöst wenn ein Reminder fällig wird.

    Spezialfall Tagebuch-Reminder am Sonntag: kombinierter Push mit Anchor-
    Frage (Wochen/Monats/Quartals je nach Tag). Beide pending-States werden
    gesetzt → User-Reply triggert Diary-Append + Anchor-Workflow.
    """
    data = ctx.job.data
    rid = data["id"]
    message = data["message"]
    try:
        # Tagebuch-Spezialfall: NUR der spezifische System-Default-Reminder
        # triggert den Bypass. User-Reminders die zufällig "tagebuch" enthalten
        # ("Tagebuch schreiben für Klassenkamerad") sollen NICHT auto-einsortiert
        # werden.
        # Detection via stable Reminder-ID (robust gegen Text-Änderungen) +
        # Backward-Compat-Patterns für historische Reminder-Texte.
        msg = (message or "").strip()
        msg_lower = msg.lower()
        is_default_diary = (
            rid == DIARY_REMINDER_ID
            or msg_lower.startswith("wenn du heute zurückblickst")
            or msg_lower.startswith("tagebuch: highlight")
            or msg.startswith("📔 Tagebuch:")
            or msg_lower.startswith("📔 tagebuch")
        )

        # Standard-Pfad: normaler Reminder. Sunday-Anker-Sonderfall wurde
        # in Bot v2 entfernt (goal_anchor-Tool weg).
        push_kind = "reminder"
        await safe_send(
            ctx.bot, ALLOWED_USER_ID,
            f"<b>Erinnerung</b>\n\n{_esc_html(message)}",
            is_html=True,
        )
        log.info(f"Reminder fired: {rid}")
        if is_default_diary:
            _save_pending_diary()

        # Generischer Bot-Push-Log in History — LLM weiß bei späterer
        # User-Antwort dass gerade ein Reminder kam.
        await _log_bot_push_to_history(
            ALLOWED_USER_ID, push_kind,
            f"Reminder ausgelöst: {message[:150]}",
        )
    except Exception as e:
        log.exception(f"Reminder-Send fehlgeschlagen für {rid}")

    # Einmalige Reminder aus JSON + ACTIVE_JOBS entfernen.
    # Lock schützt Lookup+Remove-Sequenz vor Race mit cancel_reminder.
    with _REMINDERS_LOCK:
        reminders = _load_reminders()
        reminder = next((r for r in reminders if r["id"] == rid), None)
        if reminder and not reminder.get("recurrence"):
            reminders = [r for r in reminders if r["id"] != rid]
            _save_reminders(reminders)
            ACTIVE_REMINDER_JOBS.pop(rid, None)


def list_reminders() -> str:
    """Liste aller aktiven Reminders."""
    reminders = _load_reminders()
    if not reminders:
        return "Keine aktiven Erinnerungen."
    lines = [f"⏰ {len(reminders)} aktive Erinnerung(en):"]
    for r in sorted(reminders, key=lambda x: x.get("fire_at", "")):
        when = datetime.fromisoformat(r["fire_at"])
        when_str = when.strftime("%a %d.%m. %H:%M")
        rec = f" (wiederkehrend: {r['recurrence']})" if r.get("recurrence") else ""
        lines.append(f"• `{r['id']}` — {when_str}{rec}\n  {r['message'][:80]}")
    return "\n".join(lines)


def cancel_reminder(reminder_id: str) -> str:
    """Bricht einen Reminder ab (per ID, z.B. 'rem-20260426-153000-123')."""
    reminders = _load_reminders()
    found = next((r for r in reminders if r["id"] == reminder_id), None)
    if not found:
        return f"Erinnerung nicht gefunden: {reminder_id}"
    _remove_reminder_from_json(reminder_id)
    job = ACTIVE_REMINDER_JOBS.pop(reminder_id, None)
    if job:
        try:
            job.schedule_removal()
        except Exception:
            pass
    return f"✓ Erinnerung gecancelt: {found['message'][:60]}"


LIST_FILES_NOISE_DIRS = {
    ".obsidian", ".trash", "99_Archive",
    "08_Templates", "06_Meta", "07_Tools",
}
LIST_FILES_NOISE_FILES = {
    "README.md", "_index.md",
    "CLAUDE.md", "COMMANDS.md", "MOC.md",
    "PIPELINES.md", "SCHEMA.md",
}


# ─── list_files entfernt (Phase X3 Cleanup) ──────────────────────────
# LLM-Tool 'list_files' laeuft via mcp_thin_tools.list_files() → MCP.


# ─── Conversation Memory (3-Tier) ────────────────────────────────────────────
# Tier 1: RAM-Cache (letzte HISTORY_MAX_MESSAGES, schneller Zugriff)
# Tier 2: Persistent JSONL (überlebt Restart, lazy-loaded)
# Tier 3: Facts-File (long-term Fakten über User, always im System-Prompt)

CONVERSATION_HISTORY: dict[int, list] = {}
CONVERSATION_TIMESTAMPS: dict[int, float] = {}
HISTORY_MAX_MESSAGES = 60       # ca. 30 User+Assistant-Turns
HISTORY_TIMEOUT = 60 * 60       # 1h Inaktivität → RAM-Cache leeren, beim nächsten Zugriff von Disk lazy-laden
HISTORY_PERSIST_LIMIT = 1000    # max Lines im JSONL bevor compaction
HISTORY_COMPACT_KEEP = 200       # nach compact: behalte letzte N Lines

# Lock gegen Race-Conditions zwischen User-Messages und nightly_suggestion_job
# (asyncio-cooperative — kein echtes Threading, aber sauber)
_HISTORY_LOCK = asyncio.Lock()

BOT_MEMORY_DIR = VAULT / "06_Meta" / "bot-memory"
FACTS_FILE = BOT_MEMORY_DIR / "facts.md"
PREFERENCES_FILE = BOT_MEMORY_DIR / "preferences.md"
ACTIVE_PROJECT_FILE = BOT_MEMORY_DIR / "active-project.txt"
HISTORY_FILE = BOT_MEMORY_DIR / "conversation-history.jsonl"
CORRECTIONS_FILE = BOT_MEMORY_DIR / "corrections.jsonl"
PENDING_SUGGESTIONS_FILE = BOT_MEMORY_DIR / "pending-suggestions.json"
PENDING_GOAL_ANCHOR_FILE = BOT_MEMORY_DIR / "pending-goal-anchor.json"
# Pending-Anchor TTL: 4 Stunden (Sonntag 19:00 → bis 23:00 reagierbar)
PENDING_GOAL_ANCHOR_TTL_SEC = 4 * 3600

# Pending-Diary: Bot hat 20:00 Tagebuch-Reminder gepusht — User-Reply wird
# direkt in Daily-Note einsortiert (kein LLM-Roundtrip, keine "wie kann ich
# helfen"-Antworten auf einen Tagebuch-Eintrag).
PENDING_DIARY_FILE = BOT_MEMORY_DIR / "pending-diary.json"
PENDING_DIARY_TTL_SEC = 3 * 3600  # 3h: Reminder 20:00 → bis 23:00 reagierbar


def _ensure_memory_dir():
    BOT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def get_facts() -> str:
    """Long-term Fakten lesen (werden in System-Prompt eingespeist)."""
    if not FACTS_FILE.exists():
        return ""
    try:
        content = FACTS_FILE.read_text(encoding="utf-8").strip()
        # Frontmatter überspringen falls vorhanden
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].strip()
        return content
    except Exception as e:
        log.warning(f"facts file read failed: {e}")
        return ""


def remember(fact: str) -> str:
    """Fügt einen persistenten Fakt zur Memory-Datei hinzu."""
    if not fact or not fact.strip():
        return "Fakt-Text fehlt."
    fact = fact.strip()
    _ensure_memory_dir()

    # Initialize file with header if needed
    if not FACTS_FILE.exists():
        atomic_write(
            FACTS_FILE,
            "# Bot-Memory: persistente Fakten\n\n"
            "_Hier sammelt der Bot Fakten die er sich dauerhaft merken soll. "
            "Du kannst manuell editieren — Änderungen sind beim nächsten LLM-Call wirksam._\n\n",
        )

    today = today_iso()
    line = f"- ({today}) {fact}\n"
    with FACTS_FILE.open("a", encoding="utf-8") as f:
        f.write(line)
    log.info(f"Remembered fact: {fact[:80]}")
    return f"✓ Gemerkt: {fact[:120]}"


def forget_fact(pattern: str) -> str:
    """Entfernt Fakten die `pattern` enthalten (case-insensitive)."""
    if not FACTS_FILE.exists():
        return "Keine Fakten-Datei."
    if not pattern or not pattern.strip():
        return "Such-Text fehlt."
    needle = pattern.strip().lower()
    content = FACTS_FILE.read_text(encoding="utf-8")
    lines = content.splitlines()
    kept = []
    removed = []
    for line in lines:
        if line.startswith("- ") and needle in line.lower():
            removed.append(line)
        else:
            kept.append(line)
    if not removed:
        return f"Kein Fakt gefunden mit '{pattern}'."
    atomic_write(FACTS_FILE, "\n".join(kept) + "\n")
    return f"Entfernt ({len(removed)}):\n" + "\n".join(removed[:5])


# ─── Präferenzen (Stil/Tonalität — wie der Bot reden soll) ──────────────────

def _strip_md_intro(content: str) -> str:
    """Entfernt erste H1-Überschrift + Italic-Erklärung am Datei-Anfang."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].strip()
    lines = content.splitlines()
    # H1 raus
    while lines and (lines[0].startswith("# ") or lines[0].strip() == ""):
        lines.pop(0)
    # Italic-Doku (z.B. _Hier sammelt..._) raus
    while lines and lines[0].strip().startswith("_") and lines[0].strip().endswith("_"):
        lines.pop(0)
        # Leerzeile danach
        while lines and lines[0].strip() == "":
            lines.pop(0)
    return "\n".join(lines).strip()


def get_preferences() -> str:
    """Lese Präferenzen-Inhalt (für System-Prompt-Injection)."""
    if not PREFERENCES_FILE.exists():
        return ""
    try:
        return _strip_md_intro(PREFERENCES_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"preferences read failed: {e}")
        return ""


def set_preference(text: str) -> str:
    """Fügt Präferenz hinzu (z.B. 'Antworte direkt ohne Floskeln')."""
    if not text or not text.strip():
        return "Präferenz-Text fehlt."
    text = text.strip()
    _ensure_memory_dir()
    if not PREFERENCES_FILE.exists():
        atomic_write(
            PREFERENCES_FILE,
            "# Präferenzen — wie der Bot mir antworten soll\n\n"
            "_Stil, Tonalität, Format. Werden bei jedem LLM-Call automatisch in den System-Prompt eingespeist._\n\n",
        )
    today = today_iso()
    line = f"- ({today}) {text}\n"
    with PREFERENCES_FILE.open("a", encoding="utf-8") as f:
        f.write(line)
    return f"✓ Präferenz gemerkt: {text[:120]}"


def forget_preference(pattern: str) -> str:
    if not PREFERENCES_FILE.exists():
        return "Keine Präferenzen-Datei."
    if not pattern or not pattern.strip():
        return "Such-Text fehlt."
    needle = pattern.strip().lower()
    content = PREFERENCES_FILE.read_text(encoding="utf-8")
    lines = content.splitlines()
    kept, removed = [], []
    for line in lines:
        if line.startswith("- ") and needle in line.lower():
            removed.append(line)
        else:
            kept.append(line)
    if not removed:
        return f"Keine Präferenz mit '{pattern}'."
    atomic_write(PREFERENCES_FILE, "\n".join(kept) + "\n")
    return f"Entfernt ({len(removed)}):\n" + "\n".join(removed[:5])


def forget(kind: str, pattern: str) -> str:
    """Vereinheitlichtes Forget für Memory: kind='fact' oder 'preference'.

    Konsolidiert forget_fact + forget_preference zu einem Tool.
    Der LLM-Agent muss nur noch entscheiden welches Memory-Tier.
    """
    k = (kind or "").strip().lower()
    if k in ("fact", "facts", "f"):
        return forget_fact(pattern)
    if k in ("preference", "preferences", "pref", "p"):
        return forget_preference(pattern)
    return f"Unbekanntes kind '{kind}'. Erlaubt: 'fact' oder 'preference'."


# ─── Projekt-Kontext (per-Projekt CONTEXT.md, on-demand) ────────────────────

def get_active_project() -> Optional[str]:
    """Slug des aktuell-aktiven Projekts (oder None)."""
    if not ACTIVE_PROJECT_FILE.exists():
        return None
    try:
        slug = ACTIVE_PROJECT_FILE.read_text(encoding="utf-8").strip()
        return slug or None
    except Exception:
        return None


async def get_project_context(slug: str) -> str:
    """Liest CONTEXT.md eines Projekts via MCP read_project_context (async).

    Aufruf aus llm_loop (async). Caller muss `await` vor get_project_context.
    """
    try:
        from mcp_client import mcp as _mcp, MCPError as _MCPError
        res = await _mcp.read_project_context(project=slug)
    except Exception as e:
        log.warning(f"project context read failed for {slug}: {e}")
        return ""
    if not isinstance(res, dict) or not res.get("exists"):
        return ""
    content = res.get("content", "")
    return _strip_md_intro(content) if content else ""


async def activate_project(slug: str) -> str:
    """Setzt ein Projekt als aktiv. Nutzt MCP read_project_context fuer Existenz-Check."""
    if not slug or not slug.strip():
        return "Projekt-Slug fehlt."
    slug = slug.strip().lower()
    if slug.startswith("project-"):
        slug = slug[len("project-"):]
    # Existenz-Check via MCP
    try:
        from mcp_client import mcp as _mcp
        res = await _mcp.read_project_context(project=slug)
    except Exception as e:
        return f"Projekt-Check fehlgeschlagen: {e}"
    if not isinstance(res, dict) or res.get("path") is None:
        return f"Projekt nicht gefunden: {slug}"
    _ensure_memory_dir()
    atomic_write(ACTIVE_PROJECT_FILE, slug)
    has_ctx = bool(res.get("exists"))
    ctx_info = f" (CONTEXT.md geladen)" if has_ctx else " (noch keine CONTEXT.md)"
    return f"Projekt aktiviert: [[project-{slug}]]{ctx_info}"


def deactivate_project() -> str:
    """Bricht aktives Projekt ab — CONTEXT.md wird nicht mehr geladen."""
    was = get_active_project()
    if ACTIVE_PROJECT_FILE.exists():
        ACTIVE_PROJECT_FILE.unlink()
    return f"Aktives Projekt zurückgesetzt (war: {was or 'keins'})."


# ─── project_context: Dispatcher fuer activate/deactivate/update ────────────
# Phase X3d-thin: 'update'-Pfad via MCP, 'activate'/'deactivate' bleiben lokal
# weil sie Bot-Memory (06_Meta/bot-memory/active-project.txt) anfassen, NICHT
# Vault-Content. Diese Datei ist Bot-internal-state, nicht Single-Source-Material.

async def project_context(action: str, slug: Optional[str] = None,
                          text: Optional[str] = None, mode: str = "append") -> str:
    """Vereinheitlichtes Projekt-Kontext-Tool.

    action='activate' (slug noetig)   → setzt Projekt aktiv (Bot-Memory)
    action='deactivate' (slug egal)   → bricht aktives Projekt ab (Bot-Memory)
    action='update' (slug+text noetig) → schreibt CONTEXT.md (via MCP)
    """
    a = (action or "").strip().lower()

    if a in ("activate", "aktivier", "an"):
        if not slug:
            return "Slug fehlt fuer activate."
        return await activate_project(slug)
    if a in ("deactivate", "deaktivier", "aus", "stop"):
        return deactivate_project()
    if a in ("update", "edit", "set"):
        if not slug or not text:
            return "Slug und text noetig fuer update."
        try:
            from mcp_client import mcp as _mcp, MCPError as _MCPError
            res = await _mcp.project_context(project=slug, text=text, mode=mode)
        except _MCPError as e:
            return f"Fehler bei project_context.update: {type(e).__name__}: {e}"
        path = res.get("path", "?") if isinstance(res, dict) else "?"
        return f"OK CONTEXT.md fuer {slug} {mode}: {path}"
    return f"Unbekannte action '{action}'. Erlaubt: activate / deactivate / update."


# ─── Korrektur-Log (Trainings-Material für späteres Fine-Tuning) ─────────────

# ─── Nightly Memory-Vorschläge (Hybrid: LLM extrahiert, User approved) ─────

def log_correction(was_falsch: str, was_richtig: str, kontext: str = "") -> str:
    """Speichert eine Korrektur als Lern-Datenpunkt.

    Wird ausgelöst wenn Bot etwas getan hat und User korrigiert
    ('nein anders', 'ich meinte X', 'verschieb das doch nach Y').
    Diese Records sind später Fine-Tuning-Gold (instruction-pair-Format).
    """
    if not was_falsch or not was_richtig:
        return "log_correction: was_falsch + was_richtig sind beide Pflicht."
    _ensure_memory_dir()
    record = {
        "ts": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
        "was_falsch": was_falsch.strip(),
        "was_richtig": was_richtig.strip(),
        "kontext": (kontext or "").strip(),
    }
    try:
        with CORRECTIONS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info(f"Correction logged: {was_falsch[:60]}")
        return "✓ Korrektur gespeichert (für späteres Lernen)."
    except Exception as e:
        log.exception("log_correction failed")
        return f"Log-Fehler: {e}"


def _save_history_line(user_id: int, message: dict) -> None:
    """Append einzelne Message ans persistente JSONL."""
    _ensure_memory_dir()
    record = {"user_id": user_id, "ts": time.time(), "msg": message}
    try:
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"history persist failed: {e}")


HISTORY_TAIL_READ_BYTES = 2 * 1024 * 1024   # 2MB Tail-Read-Cap (verhindert OOM bei riesigem JSONL)


def _read_tail_lines(path: Path, max_bytes: int) -> list[str]:
    """Liest die letzten max_bytes der Datei + returnt Lines (ohne erste evtl. partial).

    Verhindert dass eine 500MB-history.jsonl die Memory-grenze sprengt.
    Bei normaler Größe (<2MB) liest das ganze File. Bei großen Files: nur Tail.
    """
    file_size = path.stat().st_size
    if file_size <= max_bytes:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    with path.open("rb") as f:
        f.seek(file_size - max_bytes)
        chunk = f.read()
    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if lines:
        lines = lines[1:]  # Erste Zeile ist evtl angeschnitten
    return lines


def _sanitize_loaded_history(msgs: list) -> list:
    """Repariert Messages für strenge Provider (Anthropic, Gemini).

    Kritisch: Anthropic verlangt **strikte Adjazenz** — eine tool-Message
    darf nur direkt nach einer assistant-Message mit tool_calls kommen,
    und die tool_call_id muss dort auftauchen. Sonst:
        400 'tool_call_id of X not found in tool_calls of previous message'.

    Fixes:
    1. tool_calls auf {id, type, function:{name, arguments}} whitelisten
    2. tool_call.function.arguments zu String normalisieren
    3. tool_call.id + function.name müssen vorhanden sein, sonst tc droppen
    4. assistant-msg mit tool_calls ohne content → content=""
    5. **Strikte Adjazenz**: tool-msgs müssen IMMEDIATELY auf assistant
       mit matching tool_call_id folgen. Orphans → DROP.
    6. assistant-msg mit tool_calls braucht passende tool-replies direkt
       danach. Wenn replies fehlen → tool_calls droppen (sonst Anthropic
       beschwert sich beim NÄCHSTEN Turn).
    """

    # === Pass 1: pro-Message normalisieren (in-place auf Kopien) ===
    normalized: list = []
    for m in msgs:
        role = m.get("role")
        if role == "tool":
            m = dict(m)
            if not isinstance(m.get("content"), str):
                m["content"] = str(m.get("content", ""))
        elif role == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                m = dict(m)
                if m.get("content") is None or "content" not in m:
                    m["content"] = ""
                clean_tcs = []
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id")
                    fn = tc.get("function") or {}
                    fn_name = fn.get("name") if isinstance(fn, dict) else None
                    fn_args = fn.get("arguments") if isinstance(fn, dict) else None
                    if not tc_id or not fn_name:
                        continue
                    if isinstance(fn_args, (dict, list)):
                        try:
                            fn_args = json.dumps(fn_args, ensure_ascii=False)
                        except Exception:
                            fn_args = "{}"
                    elif fn_args is None:
                        fn_args = "{}"
                    elif not isinstance(fn_args, str):
                        fn_args = str(fn_args)
                    clean_tcs.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": fn_name, "arguments": fn_args},
                    })
                if clean_tcs:
                    m["tool_calls"] = clean_tcs
                else:
                    m.pop("tool_calls", None)
                    if not m.get("content"):
                        continue
        normalized.append(m)

    # === Pass 2: Adjazenz erzwingen ===
    # Strategie: Walk forward. Wenn assistant mit tool_calls kommt,
    # sammle die erwarteten ids. Folgende tool-msgs MÜSSEN diese ids
    # treffen — fremde tool-msgs werden gedroppt. Wenn assistant
    # tool_calls hat, aber NICHT alle Replies kommen, → tool_calls
    # vom assistant droppen (oder die ganze Message wenn dann leer).
    # tool-msgs ohne vorhergehenden tool-call-Kontext → DROP.

    out: list = []
    expected_ids: set[str] = set()       # noch ausstehende tool_call_ids vom letzten assistant
    pending_assistant_idx: int = -1      # Index in `out` der wartet auf Replies
    pending_assistant_ids: list[str] = []  # alle ids vom pending assistant (für Cleanup)

    def _close_pending_assistant():
        """Falls pending assistant nicht alle Replies bekam: tool_calls droppen
        die keine Reply haben (sonst Anthropic-400 beim NÄCHSTEN Call).
        """
        nonlocal expected_ids, pending_assistant_idx, pending_assistant_ids
        if pending_assistant_idx >= 0 and expected_ids:
            am = out[pending_assistant_idx]
            am = dict(am)
            am["tool_calls"] = [
                tc for tc in am.get("tool_calls", [])
                if tc.get("id") not in expected_ids
            ]
            if not am["tool_calls"]:
                am.pop("tool_calls", None)
            if not am.get("tool_calls") and not am.get("content"):
                # Komplett leer → assistant-msg ganz raus
                out.pop(pending_assistant_idx)
            else:
                out[pending_assistant_idx] = am
        expected_ids = set()
        pending_assistant_idx = -1
        pending_assistant_ids = []

    for m in normalized:
        role = m.get("role")
        if role == "tool":
            tc_id = m.get("tool_call_id")
            if not tc_id or tc_id not in expected_ids:
                # Orphan tool-msg → DROP (Anthropic-400-Vermeidung)
                continue
            # Name-Feld nachrüsten falls fehlt (für Gemini)
            if not m.get("name"):
                # Lookup im pending assistant
                if pending_assistant_idx >= 0:
                    for tc in out[pending_assistant_idx].get("tool_calls", []):
                        if tc.get("id") == tc_id:
                            m = dict(m)
                            m["name"] = tc.get("function", {}).get("name", "unknown")
                            break
            expected_ids.discard(tc_id)
            out.append(m)
            # Wenn alle expected geliefert, pending closen (alle Replies da)
            if not expected_ids:
                pending_assistant_idx = -1
                pending_assistant_ids = []
        else:
            # Wenn neue Message (user oder assistant) während noch Replies fehlen,
            # → pending closen (= incomplete tool_calls droppen)
            if expected_ids:
                _close_pending_assistant()
            out.append(m)
            if role == "assistant" and m.get("tool_calls"):
                pending_assistant_idx = len(out) - 1
                pending_assistant_ids = [tc.get("id") for tc in m["tool_calls"] if tc.get("id")]
                expected_ids = set(pending_assistant_ids)

    # End-of-list cleanup: pending assistant ohne Replies
    if expected_ids:
        _close_pending_assistant()

    return out


def _load_persistent_history(user_id: int) -> list:
    """Letzte HISTORY_MAX_MESSAGES Messages für User aus JSONL laden.

    Memory-safe via _read_tail_lines: bei riesigen Files nur die letzten
    2MB lesen (immer noch 1000+ Records typisch).
    """
    if not HISTORY_FILE.exists():
        return []
    try:
        lines = _read_tail_lines(HISTORY_FILE, HISTORY_TAIL_READ_BYTES)
        records = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("user_id") == user_id:
                    records.append(rec)
            except json.JSONDecodeError:
                continue
        msgs = [r["msg"] for r in records[-HISTORY_MAX_MESSAGES:]]
        return _sanitize_loaded_history(msgs)
    except Exception as e:
        log.warning(f"history load failed: {e}")
        return []


def _maybe_compact_history() -> None:
    """JSONL trimmen wenn zu groß: nur letzte 200 Lines behalten.

    Memory-safe: nutzt _read_tail_lines statt full read_text wenn File riesig.
    """
    if not HISTORY_FILE.exists():
        return
    try:
        # Schnell-Check: File-Größe → wenn klein, direkt lesen, sonst tail-only
        file_size = HISTORY_FILE.stat().st_size
        if file_size <= HISTORY_TAIL_READ_BYTES:
            lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
        else:
            lines = _read_tail_lines(HISTORY_FILE, HISTORY_TAIL_READ_BYTES)
        if len(lines) <= HISTORY_PERSIST_LIMIT:
            return
        keep = lines[-HISTORY_COMPACT_KEEP:]
        atomic_write(HISTORY_FILE, "\n".join(keep) + "\n")
        log.info(f"History compacted: {len(lines)} → {len(keep)}")
    except Exception as e:
        log.warning(f"history compact failed: {e}")


def _check_cache_fresh(user_id: int) -> tuple[Optional[list], bool]:
    """Pure Read, no IO. Returns (cached_list_or_None, is_fresh).

    Wird ohne Lock aus async-Context gerufen (RAM-Read ist atomic in CPython).
    Bei is_fresh=True kann der Caller direkt den Cache verwenden.
    """
    last = CONVERSATION_TIMESTAMPS.get(user_id, 0.0)
    cached = CONVERSATION_HISTORY.get(user_id)
    is_fresh = cached is not None and (time.time() - last) <= HISTORY_TIMEOUT
    return cached, is_fresh


async def _ensure_history_loaded(user_id: int) -> list:
    """Stellt sicher dass die History für user_id im Cache ist.

    Disk-IO läuft via to_thread → blockiert NICHT den event-loop. Cache-
    Update unter Lock damit konkurrierende Calls keine inkonsistenten
    Reads sehen. Returns die geladene Liste (nicht aus Cache, da andere
    Tasks zwischenzeitlich verändert haben könnten).
    """
    cached, is_fresh = _check_cache_fresh(user_id)
    if is_fresh:
        return list(cached)
    # Disk-IO AUSSERHALB Lock — blockiert event loop nicht
    loaded = await asyncio.to_thread(_load_persistent_history, user_id)
    async with _HISTORY_LOCK:
        # Recheck — ein anderer Task könnte schon geladen+mutiert haben
        cached, is_fresh = _check_cache_fresh(user_id)
        if is_fresh:
            return list(cached)
        CONVERSATION_HISTORY[user_id] = loaded
        CONVERSATION_TIMESTAMPS[user_id] = time.time()
    return list(loaded)


async def get_history(user_id: int) -> list:
    """History für User. Bei Cache-Miss/Timeout: async lazy-load aus JSONL.

    Disk-IO läuft via to_thread → blockiert event-loop nicht mehr.
    """
    return await _ensure_history_loaded(user_id)


async def update_history(user_id: int, new_messages: list) -> None:
    """History anhängen + persistieren + auf Max-Länge trimmen.

    Hot-Path: RAM-Mutation unter Lock (schnell). Disk-Append + Compaction
    via to_thread AUSSERHALB Lock — JSONL-Append ist append-only, Race
    zwischen mehreren Schreibern wäre nur Reihenfolge-Issue (für Single-
    User-Bot irrelevant).
    """
    # Sicherstellen dass Cache da ist (lädt wenn nötig, async)
    await _ensure_history_loaded(user_id)
    # RAM-Update unter Lock (sehr schnell — keine Disk-IO)
    async with _HISTORY_LOCK:
        history = list(CONVERSATION_HISTORY.get(user_id, []))
        history.extend(new_messages)
        if len(history) > HISTORY_MAX_MESSAGES:
            history = history[-HISTORY_MAX_MESSAGES:]
        CONVERSATION_HISTORY[user_id] = history
        CONVERSATION_TIMESTAMPS[user_id] = time.time()
    # Disk-Append + ggf Compaction via to_thread, kein Lock
    await asyncio.to_thread(_persist_history_changes, user_id, new_messages)


def _persist_history_changes(user_id: int, new_messages: list) -> None:
    """Sync helper für update_history — schreibt JSONL-Append + ggf compact."""
    for msg in new_messages:
        _save_history_line(user_id, msg)
    _maybe_compact_history()


async def reset_history(user_id: int) -> None:
    """RAM-Cache leeren. JSONL bleibt erhalten — Memory wird beim nächsten Mal neu geladen.

    Wenn du wirklich permanent löschen willst: HISTORY_FILE manuell entfernen.
    """
    async with _HISTORY_LOCK:
        CONVERSATION_HISTORY.pop(user_id, None)
        CONVERSATION_TIMESTAMPS.pop(user_id, None)


# ============================================================================
# Token-Usage-Tracking (für /usage Reports + Cost-Awareness)
# ============================================================================
# Pro Tag eine Zeile in 06_Meta/usage/YYYY-MM-DD.json — append-only.
# Records: {"ts": iso, "model": str, "in": int, "out": int, "kind": str}
# kind: "chat" (LLM-Call), "vision" (Photo-Caption), "whisper" (lokal, $0)

USAGE_DIR = VAULT / "06_Meta" / "usage"

# Approximate Pricing (USD per 1M tokens) — nur für Schätzung, nicht exakt.
# Für Ollama-Cloud / lokale Modelle: 0/0. Bot zeigt im /usage diese Caveat.
# Werte sind Stand 2025/2026 grob — User updated bei Bedarf in der Datei.
PRICING_USD_PER_M = {
    # Anthropic direkt (ohne Provider-Prefix — Bot nutzt aktuell diesen Pfad)
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-5": (15.0, 75.0),
    # OpenRouter / Anthropic (mit Provider-Prefix)
    "anthropic/claude-sonnet-4-5": (3.0, 15.0),
    "anthropic/claude-opus-4": (15.0, 75.0),
    "anthropic/claude-haiku-4": (0.8, 4.0),
    "anthropic/claude-haiku-4-5": (1.0, 5.0),
    # OpenAI
    "openai/gpt-4o": (2.5, 10.0),
    "openai/gpt-4o-mini": (0.15, 0.60),
    # Google
    "google/gemini-2.5-flash": (0.075, 0.30),
    "google/gemini-2.5-pro": (1.25, 10.0),
    # Ollama Cloud / lokale → kostenlos
    "gpt-oss:120b-cloud": (0.0, 0.0),
    "qwen3:235b-cloud": (0.0, 0.0),
}


def _track_usage(model: str, prompt_tokens: int, completion_tokens: int, kind: str = "chat") -> None:
    """Speichert ein Token-Usage-Record in 06_Meta/usage/YYYY-MM-DD.jsonl.

    Wird sync aufgerufen aus dem llm_loop nach jedem LLM-Call. Failures
    werden geloggt aber nicht propagiert — Tracking darf den Bot nie crashen.
    """
    try:
        USAGE_DIR.mkdir(parents=True, exist_ok=True)
        today = today_iso()
        path = USAGE_DIR / f"{today}.jsonl"
        rec = {
            "ts": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
            "model": model,
            "in": int(prompt_tokens or 0),
            "out": int(completion_tokens or 0),
            "kind": kind,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.debug(f"usage-track failed (silently): {e}")


def _estimate_cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    """Schätzt Kosten in USD für Model+Tokens. 0 wenn Model unbekannt."""
    pricing = PRICING_USD_PER_M.get(model)
    if not pricing:
        return 0.0
    in_price, out_price = pricing
    return (in_tokens * in_price + out_tokens * out_price) / 1_000_000


def get_usage_summary(days: int = 7) -> str:
    """Liest letzte N Tage Usage und produziert Report.

    Tool-callable von User via /usage oder LLM-Tool.
    """
    if not USAGE_DIR.exists():
        return "Noch keine Usage-Daten."
    today = datetime.now(TIMEZONE).date()
    by_day: dict = {}  # day_iso → {"in": x, "out": y, "calls": z, "cost": $, "by_model": {model: ...}}
    for delta in range(days):
        d = today - timedelta(days=delta)
        d_iso = d.isoformat()
        path = USAGE_DIR / f"{d_iso}.jsonl"
        if not path.exists():
            continue
        day_data = {"in": 0, "out": 0, "calls": 0, "cost_usd": 0.0, "by_model": {}}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                model = rec.get("model", "unknown")
                ti = int(rec.get("in", 0) or 0)
                to = int(rec.get("out", 0) or 0)
                cost = _estimate_cost_usd(model, ti, to)
                day_data["in"] += ti
                day_data["out"] += to
                day_data["calls"] += 1
                day_data["cost_usd"] += cost
                m = day_data["by_model"].setdefault(model, {"in": 0, "out": 0, "calls": 0, "cost": 0.0})
                m["in"] += ti
                m["out"] += to
                m["calls"] += 1
                m["cost"] += cost
            by_day[d_iso] = day_data
        except Exception as e:
            log.warning(f"usage-summary day {d_iso} failed: {e}")

    if not by_day:
        return f"Keine Usage-Daten in den letzten {days} Tagen."

    # Aggregiert
    total_in = sum(d["in"] for d in by_day.values())
    total_out = sum(d["out"] for d in by_day.values())
    total_calls = sum(d["calls"] for d in by_day.values())
    total_cost = sum(d["cost_usd"] for d in by_day.values())

    parts = [f"**Token-Usage** (letzte {days} Tage)"]
    parts.append(f"")
    parts.append(f"**Total:** {total_calls} Calls · {total_in:,} in · {total_out:,} out")
    if total_cost > 0:
        parts.append(f"**Geschätzte Kosten:** ${total_cost:.3f} USD")
    else:
        parts.append("_Aktuelle Modelle haben keinen Preis hinterlegt (z.B. Ollama-Cloud = gratis)._")

    parts.append("")
    parts.append("**Pro Tag:**")
    for d_iso in sorted(by_day.keys(), reverse=True):
        d = by_day[d_iso]
        cost_str = f" · ${d['cost_usd']:.3f}" if d["cost_usd"] > 0 else ""
        parts.append(f"  • {d_iso}: {d['calls']} calls · {d['in']:,} in · {d['out']:,} out{cost_str}")

    # Top-Modelle
    model_totals: dict = {}
    for d in by_day.values():
        for m, mdata in d["by_model"].items():
            mt = model_totals.setdefault(m, {"in": 0, "out": 0, "calls": 0, "cost": 0.0})
            mt["in"] += mdata["in"]
            mt["out"] += mdata["out"]
            mt["calls"] += mdata["calls"]
            mt["cost"] += mdata["cost"]
    if len(model_totals) > 1:
        parts.append("")
        parts.append("**Pro Modell:**")
        for m, mt in sorted(model_totals.items(), key=lambda x: -x[1]["calls"]):
            cost_str = f" · ${mt['cost']:.3f}" if mt["cost"] > 0 else ""
            parts.append(f"  • `{m}`: {mt['calls']} calls · {mt['in']:,}/{mt['out']:,} in/out{cost_str}")

    return "\n".join(parts)


# ============================================================================
# Goal-System (5-Jahres-Plan unter 10_Life/goals/<goal-slug>/)
# ============================================================================
# Drei Tools — bewusst dem konsolidierten task-Pattern folgend:
#   goal_log      Daily-Einträge (sport, win, habit, book, lesson)
#   goal_anchor   Wochen/Monats/Quartals-Review (2-step: erst File anlegen +
#                 Fragen returnieren, dann mit answers aufrufen zum Schreiben)
#   goal_status   Read-only Aggregation (Säulen, Habits, Sport, Drift)
#
# Default goal-slug ist "5y-2031" — später kann via env oder Tool-Param
# auf andere Goals umgestellt werden.




# ─── goal_status entfernt (Phase X3 Cleanup) ──────────────────────────
# LLM-Tool 'goal_status' laeuft via mcp_thin_tools.goal_status() → MCP.


# ============================================================================
# Tool definitions (OpenAI function-calling format)
# ============================================================================

TOOLS = [
    {"type": "function", "function": {
        "name": "append_to_daily",
        "description": "Haengt Text an die heutige Daily-Note unter Sektion an. Sektionen: 'Heute' (Tasks/Termine), 'Notizen & Gedanken' (default), 'Offen / Einsortieren' (Backlog/Links), 'Abends' (Reflexion).",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string", "enum": ["Heute", "Notizen & Gedanken", "Offen / Einsortieren", "Abends"], "default": "Notizen & Gedanken"},
                "text": {"type": "string"}
            },
            "required": ["text"]
        }
    }},
    {"type": "function", "function": {
        "name": "task",
        "description": "Konsolidiertes Task-Tool. action='create' (title+optional priority/due/project/recurrence), 'done'/'reopen'/'update' (task_id noetig).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "done", "reopen", "update"]},
                "task_id": {"type": "string"},
                "title": {"type": "string"},
                "priority": {"type": "string", "enum": ["urgent", "high", "medium", "low"]},
                "due": {"type": "string"},
                "project": {"type": "string"},
                "context": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "recurrence": {"type": "string", "enum": ["daily", "weekdays", "weekly", "monthly"]}
            },
            "required": ["action"]
        }
    }},
    {"type": "function", "function": {
        "name": "create_note",
        "description": "Neue freie Note anlegen. project=<slug> fuer Projekt-Note, sonst landet sie in 10_Life/notes/.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "project": {"type": "string"}
            },
            "required": ["title", "body"]
        }
    }},
    {"type": "function", "function": {
        "name": "create_meeting",
        "description": "Meeting-Protokoll anlegen. attendees als Liste, optional project/meeting_date.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
                "meeting_date": {"type": "string", "description": "ISO YYYY-MM-DD, default heute"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "project": {"type": "string"}
            },
            "required": ["title"]
        }
    }},
    {"type": "function", "function": {
        "name": "create_project",
        "description": "Neuen Projekt-Container unter 05_Projects/<slug>/ anlegen. parent=<slug> macht Subprojekt. Erstellt README mit Dataview + leere CONTEXT.md.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Anzeige-Name; wird zu slug konvertiert"},
                "description": {"type": "string"},
                "parent": {"type": "string", "description": "Slug eines existierenden Projekts fuer Subprojekt"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name"]
        }
    }},
    {"type": "function", "function": {
        "name": "search_vault",
        "description": "Volltext-Regex-Suche durch alle .md-Files im Vault.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Inhalt eines Files lesen (max 8KB). Default ohne Frontmatter.",
        "parameters": {
            "type": "object",
            "properties": {
                "rel_path": {"type": "string"},
                "strip_frontmatter": {"type": "boolean", "default": True}
            },
            "required": ["rel_path"]
        }
    }},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "Alle .md-Files in einem Vault-Unterordner listen.",
        "parameters": {
            "type": "object",
            "properties": {
                "rel_dir": {"type": "string", "default": ""},
                "include_system": {"type": "boolean", "default": False}
            }
        }
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Find/Replace in einem File. regex=true fuer Regex-Pattern (mit ReDoS-Schutz, max 5 MB Files).",
        "parameters": {
            "type": "object",
            "properties": {
                "rel_path": {"type": "string"},
                "find": {"type": "string"},
                "replace": {"type": "string"},
                "regex": {"type": "boolean", "default": False}
            },
            "required": ["rel_path", "find", "replace"]
        }
    }},
    {"type": "function", "function": {
        "name": "move",
        "description": "Datei/Ordner verschieben. 3 Modi: einzeln (src+dst), bulk (srcs+dst), project (project_slug+optional parent).",
        "parameters": {
            "type": "object",
            "properties": {
                "src": {"type": "string"},
                "srcs": {"type": "array", "items": {"type": "string"}},
                "dst": {"type": "string"},
                "project_slug": {"type": "string"},
                "parent": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False}
            }
        }
    }},
    {"type": "function", "function": {
        "name": "request_delete",
        "description": "Loesch-Anfrage stellen. Akkumuliert Pfade lokal — confirm_delete fuehrt aus. permanent=true loescht hart, sonst Archiv.",
        "parameters": {
            "type": "object",
            "properties": {
                "rel_path": {"type": "string"},
                "rel_paths": {"type": "array", "items": {"type": "string"}},
                "permanent": {"type": "boolean", "default": False}
            }
        }
    }},
    {"type": "function", "function": {
        "name": "confirm_delete",
        "description": "Pending Loeschung bestaetigen oder abbrechen.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["confirm", "cancel"], "default": "confirm"}
            }
        }
    }},
    {"type": "function", "function": {
        "name": "list_open_tasks",
        "description": "Offene Tasks listen. Filter: 'overdue'/'today'/'tomorrow'/'week'/'nodate' oder leer (alle).",
        "parameters": {
            "type": "object",
            "properties": {
                "when": {"type": "string"},
                "project": {"type": "string"}
            }
        }
    }},
    {"type": "function", "function": {
        "name": "get_today_agenda",
        "description": "Heute-Agenda: ueberfaellige + heute-faellige Tasks + naechste 3 Tage + Inbox.",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "goal_status",
        "description": "5y-Goal-Status: Saeulen, Habits-Quote, Sport-Sessions, Drift-Anker.",
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": ["all", "saeule", "habits", "sport", "drift"], "default": "all"},
                "saeule": {"type": "string"}
            }
        }
    }},
    {"type": "function", "function": {
        "name": "project_context",
        "description": "Projekt-Kontext verwalten. action='activate' (slug) | 'deactivate' | 'update' (slug+text+optional mode='append'/'replace').",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["activate", "deactivate", "update"]},
                "slug": {"type": "string"},
                "text": {"type": "string"},
                "mode": {"type": "string", "enum": ["append", "replace"], "default": "append"}
            },
            "required": ["action"]
        }
    }},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Fakt ueber Julius im Bot-Memory speichern (06_Meta/bot-memory/facts.md).",
        "parameters": {
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"]
        }
    }},
    {"type": "function", "function": {
        "name": "forget",
        "description": "Memory-Eintrag loeschen. kind='fact' oder 'preference', pattern matched gegen Inhalt.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["fact", "preference"]},
                "pattern": {"type": "string"}
            },
            "required": ["kind", "pattern"]
        }
    }},
    {"type": "function", "function": {
        "name": "set_preference",
        "description": "Praeferenz/Stil-Vorgabe im Bot-Memory speichern.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        }
    }},
    {"type": "function", "function": {
        "name": "log_correction",
        "description": "Korrektur-Eintrag fuer spaetere Auswertung loggen.",
        "parameters": {
            "type": "object",
            "properties": {
                "was_falsch": {"type": "string"},
                "was_richtig": {"type": "string"},
                "kontext": {"type": "string"}
            },
            "required": ["was_falsch", "was_richtig"]
        }
    }},
    {"type": "function", "function": {
        "name": "create_reminder",
        "description": "Telegram-Reminder anlegen. when_iso ISO-Datetime, optional recurrence='daily'/'weekly'.",
        "parameters": {
            "type": "object",
            "properties": {
                "when_iso": {"type": "string"},
                "message": {"type": "string"},
                "recurrence": {"type": "string"}
            },
            "required": ["when_iso", "message"]
        }
    }},
    {"type": "function", "function": {
        "name": "list_reminders",
        "description": "Aktive Reminders listen.",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "cancel_reminder",
        "description": "Reminder via ID stornieren.",
        "parameters": {
            "type": "object",
            "properties": {"reminder_id": {"type": "string"}},
            "required": ["reminder_id"]
        }
    }},
]

TOOL_HANDLERS = {
    # MCP-routed Tools — alle Vault-Operationen via MCP-Server
    "search_vault":     mcp_thin_tools.search_vault,
    "read_file":        mcp_thin_tools.read_file,
    "list_files":       mcp_thin_tools.list_files,
    "append_to_daily":  mcp_thin_tools.append_to_daily,
    "create_note":      mcp_thin_tools.create_note,
    "create_meeting":   mcp_thin_tools.create_meeting,
    "create_project":   mcp_thin_tools.create_project,
    "task":             mcp_thin_tools.task,
    "goal_status":      mcp_thin_tools.goal_status,
    "edit_file":        mcp_thin_tools.edit_file,
    "move":             mcp_thin_tools.move,
    "list_open_tasks":  mcp_thin_tools.list_open_tasks,
    "get_today_agenda": mcp_thin_tools.get_today_agenda,
    "request_delete":   mcp_thin_tools.request_delete,
    "confirm_delete":   mcp_thin_tools.confirm_delete,
    "project_context":  project_context,  # async dispatcher: update→MCP, activate/deactivate→Bot-Memory
    # Bot-only (Telegram/Memory — keine Vault-Operationen)
    "remember":         remember,
    "forget":           forget,
    "set_preference":   set_preference,
    "log_correction":   log_correction,
    "create_reminder":  create_reminder,
    "list_reminders":   list_reminders,
    "cancel_reminder":  cancel_reminder,
}

# ============================================================================
# System prompt (cached)
# ============================================================================

SYSTEM_PROMPT = """Du bist Julius' Vault-Assistent ueber Telegram. Deutsch, direkt, kein Geschwurbel.

# VAULT-Layout (alle Schreib-Operationen via MCP)
- `10_Life/{daily,tasks,notes,meetings,goals}/` — Persoenliches
- `05_Projects/<slug>/` — Projekte (Subprojekte als Subordner)
- `02_Wiki/` — Kompiliertes Wissen
- `01_Raw/` — Externe Quellen

# Trigger → Tool

- "speicher/merk dir/notiere/tagebuch" → `append_to_daily` oder `create_note`
- "task/todo/Imperativ+Frist" → `task(action='create')` (recurrence bei "taeglich/woechentlich/monatlich")
- "was steht heute an/agenda" → `get_today_agenda`
- "alle offenen Tasks" → `list_open_tasks` (when=overdue/today/tomorrow/week/nodate, optional project)
- "X erledigt" → `task(action='done')` · Deadline aendern → `task(action='update', due=...)` · "wieder oeffnen" → `task(action='reopen')`
- "meeting:/war im Termin" → `create_meeting`
- "loesche X" → `request_delete` (Default Archiv); "endgueltig/hart" → permanent=true; mehrere Files → rel_paths=[...]
- "ja/bestaetigt" nach request_delete → `confirm_delete()`; "nein/abbrechen" → `confirm_delete(action='cancel')`
- "verschieb A nach B" → `move(src='A', dst='B')`; mehrere → `move(srcs=[...], dst='ordner/')`; Subprojekt → `move(project_slug='X', parent='Y')`
- "neues Projekt X" → `create_project(name='X')` (direkt anlegen, nicht nachfragen); Subprojekt → `create_project(name='X', parent='Y')`
- "erinner mich um Y/in N Min/taeglich um Z" → `create_reminder` (when_iso = absolute Lokalzeit, optional recurrence)
- "welche Reminder/cancel" → `list_reminders` / `cancel_reminder`
- "wo stehe ich/5y-Status" → `goal_status` (scope=all/saeule/habits/sport/drift)
- "ersetze X durch Y in Note/File" → `edit_file(rel_path, find, replace)` — vorher `search_vault` + `read_file` zur Verifikation des find-Strings

# Vault-Inhalts-Modell (read-only Lookup)

- "wer linkt auf X / Backlinks" → `get_backlinks(path)` (nicht search_vault mit `[[X]]`)
- "auf was linkt X / outgoing" → `get_outgoing_links(path)`
- "alle Tags / welche Tags habe ich" → `list_tags(scope?, min_count=1)`
- "alle Files mit Tag X" → `find_by_tag(tag, scope?)` (nicht search_vault mit `#X`)
- "alle Files mit status:open / due gestern / priority:urgent" → `find_by_property(field, value, op)` (op: eq|contains|gt|lt|exists|in)
- "find die Note mit Alias X / wer ist Spitzname X" → `resolve_alias(query)`
- "zeig mir nur Headings/Struktur von X" → `get_outline(path, include_tables?)` — vor edit_file bei grossen Files

# Refactoring + Recovery

- "haeng X unter ## Section Y in Datei Z an" → `append_under_heading(path, heading, content)` (ersetzt edit_file fuer Section-Append)
- "splitte Section X aus Datei Y" → `split_file(path, at_heading, new_path)` — Section wandert raus, Source behaelt Rest
- "merge Files A,B in C" → `merge_files(sources, target, mode='append'|'prepend')` (optional `delete_sources=True`)
- "neue Note nach Template X" → `apply_template(template_path, target_path, vars={...})` — Vars: `{{date}}`, `{{title}}`, `{{var:default}}`
- "rolle den Stand vor 30 Min zurueck" → `list_snapshots(rel_path?)` → `restore_snapshot(snapshot_id, target_path?)` (legt VOR der Restore noch einen pre_restore_snapshot an)
- "was ist heute geaendert worden an X" → `list_snapshots(rel_path=X, since=heute)`

# Dashboard / Aggregat / Explore

- "wie steht's um Projekt X / Projekt-Status" → `project_overview(slug)` (1 Call statt 4: Tasks + Notes + Stunden + Status)
- "Vault-Statistik / wieviele Tasks offen / wie viele Notes" → `vault_stats(scope?)`
- "zeig mir Notes im Umfeld von X / verlinkte Cluster" → `get_subgraph(start_path, depth=2)`
- "zeig mir was Altes zufaellig / random Note" → `random_note(scope?, tag_filter?)`
- "wer hat heute an Datei X was gemacht / File-History" → `file_audit(path, since?)`

# Memory (Bot-state, nicht Vault)

| User sagt | Tool |
|---|---|
| "antworte kuerzer / kein 'gerne!' / DD.MM. statt ISO" | `set_preference` |
| "merk dir / ich heisse Julius / KV-Lohn 32,80" | `remember` |
| "Projekt X: ..." (Projekt aktiv) | `project_context(action='update', slug=..., text=...)` |
| "lass uns ueber X reden / arbeite jetzt an X" | `project_context(action='activate', slug='X')` |
| "fertig mit X / weg vom Projekt" | `project_context(action='deactivate')` |

Multi-Fakt: ein `remember`-Call mit Newline-Trennung, nicht N× einzeln.
Beschwerden zuerst paraphrasieren + rueckbestaetigen, dann speichern.

# Daten

- Datum ISO `YYYY-MM-DD`. "morgen" = +1, "naechsten Montag" → berechnen.
- Tasks: `due` NUR wenn explizit genannt. Prioritaet aus Sprache: "dringend/asap" → urgent, "wichtig" → high, "irgendwann" → low.
- Auto-Tags bei task/create_note/create_meeting: 2-5 topische Tags (kebab-case Deutsch).

# Projekt-Routing

`10_Life/` = privates Leben ohne Projekt-Bezug. `05_Projects/<slug>/` = alles projekt-bezogen.

Bei jedem Note/Meeting: erkenne Projekt-Bezug aus Inhalt + aktivem Projekt. Wenn ja → `project=<slug>`. Sonst → ohne project-Parameter (landet in 10_Life/).

# Bestehende Datei aendern — PFLICHT-READ vor jedem Schreiben

Bevor du eine bestehende Datei aenderst (edit_file, edit_file_replace,
raw_write, append_table_row, append_under_heading): IMMER zuerst
`read_file(<pfad>)` aufrufen. Auch wenn du den Inhalt vor 5 Minuten
gelesen hast — erneut lesen. Du weisst nicht was zwischen Calls passiert.

Format-erhaltend schreiben:
- body hat Markdown-Tabelle → `append_table_row` (NIE Prosa-Block davor/danach)
- body hat ## Heading-Sections → `append_under_heading`
- body ist Prosa → `edit_file` (body) oder `edit_file_replace` (find/replace)

Standard-Flow fuer Edits:
1. `search_vault(<stichworte>)` falls Pfad unbekannt → Pfad in Backticks am Zeilenende
2. `read_file(<pfad>)` — IMMER, ohne Ausnahme
3. Format des body erkennen → richtiges Tool waehlen
4. Bestaetigen mit 1 Satz Klartext, kein Pfad-Dump.

NIEMALS `rel_path` aus ID raten (Files haben oft `2026-04-28_<id>.md`-Praefix).

# Aussagen ueber File-Inhalt = read_file ZUERST

Bevor du dem User sagst "Datei ist leer / hat Format X / enthaelt Y nicht /
ich finde keine Tabelle": IMMER zuerst `read_file` aufrufen. NIE aus dem
Conversation-State raten was im File steht. Wenn du irrtest, hast du
gelogen.

# Reminder-Context

Wenn deine letzte assistant-Message mit `[Bot-Push HH:MM kind=...]` beginnt, ist die naechste User-Message hoechstwahrscheinlich die Antwort auf den Push:
- `kind=reminder` → User reagiert. Direkt erledigen/einsortieren, kein "wie kann ich helfen".
- `kind=morning-briefing` → Tagesplan-Antwort. Items als Reminders/Tasks eintragen (Uhrzeit → reminder, ohne Uhrzeit → task).

# Korrekturen

User-Reaktion "nein/falsch/besser X statt Y" auf deine letzte Aktion → `edit_file` + `log_correction`. Bei Frust kurz entschuldigen, klaerend nachfragen, KEINE blinde Folge-Aktion.

# Ausgabe

- Deutsch, direkt. KEINE Emojis ausser Status (✓/✗). Priority-Symbole (🔴🟠🟡🟢) nur aus Tool-Output, nie selbst setzen.
- Aktion-Bestaetigung: 1 Satz. Wikilink `[[id]]` NUR bei NEU erstellten Items. Bei done/update/move/edit nur Klartext-Titel.
- Wikilinks: nur echte IDs aus search_vault/read_file. Keine Filepaths, keine Platzhalter.
- NIE HTML-Tags, NIE Frontmatter ausgeben.
"""

# ============================================================================
# LLM tool-use loop
# ============================================================================

# Sensitive-Pattern-Maskierung für Error-Messages die in History/User landen
_SENSITIVE_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "[REDACTED-API-KEY]"),       # OpenAI/Anthropic
    (re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"), "Bearer [REDACTED]"),
    # Telegram-Bot-Token: <8-12 digits>:<base64-ish 35 chars>
    (re.compile(r"\b\d{7,15}:[A-Za-z0-9_\-]{30,}"), "[REDACTED-TG-TOKEN]"),
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "[REDACTED-GH-PAT]"),           # GitHub PAT classic
    (re.compile(r"github_pat_[A-Za-z0-9_]{50,}"), "[REDACTED-GH-PAT]"),   # GitHub fine-grained
    (re.compile(r"https://[^@\s]+:[^@\s]+@"), "https://[REDACTED]@"),     # URL-mit-Credentials
]


def _sanitize_error(msg: str) -> str:
    """Maskiert Tokens/Keys/Credentials in Error-Strings.

    Wird auf Tool-Fehler-Messages angewandt bevor sie in History oder User-
    Reply landen. Verhindert dass Stack-Traces oder Lib-Errors versehentlich
    Bearer-Tokens, OpenAI-Keys, Telegram-Bot-Tokens leaken.
    """
    if not isinstance(msg, str):
        msg = str(msg)
    for pat, replacement in _SENSITIVE_PATTERNS:
        msg = pat.sub(replacement, msg)
    return msg


# ─── LLM-API-Retry mit exponential backoff ──────────────────────────────────
# Ollama/OpenRouter/OpenAI haben gelegentlich transiente Fehler (502, timeout,
# rate-limit). Vorher: Exception bubble → llm_loop crashed → User muss neu
# schreiben. Jetzt: 3 Versuche mit 1s/2s/4s Backoff, dann erst geben wir auf.

LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_BASE_DELAY = 1.0  # 1s, 2s, 4s

# Tool-Hard-Timeout: egal was ein Tool tut, nach 90s abbrechen.
# Real-World-Anker: backup_vault dauert ~30s, extract_pdf_text ~10-20s
# bei großem PDF, search_vault timeout intern auf 30s. 90s ist großzügig.
TOOL_TIMEOUT_SEC = 90

# Welche Exception-Typen retry-würdig sind (transient). Andere werfen direkt.
def _is_retriable_llm_error(e: Exception) -> bool:
    name = type(e).__name__
    # OpenAI-SDK + httpx Standard-Errors die transient sein können
    transient_names = (
        "APIConnectionError", "APITimeoutError", "InternalServerError",
        "RateLimitError", "ConnectionError", "TimeoutException",
        "ReadTimeout", "ConnectTimeout", "RemoteProtocolError",
    )
    if name in transient_names:
        return True
    # HTTPStatusError: 408/429/500/502/503/504 → retry, alles andere nicht
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status in (408, 429, 500, 502, 503, 504):
        return True
    # Generic check via String-Match (Fallback)
    msg = str(e).lower()
    if any(s in msg for s in ("timeout", "connection", "rate limit", "502", "503", "504")):
        return True
    return False


# History-Token-Budget — verhindert Context-Overflow bei langen Sessions.
# Konservativ: 80k von typisch 128k Context-Window, Rest für Tool-Schemas
# (~3k) + Response (~8k) + Safety-Buffer.
HISTORY_TOKEN_BUDGET = 80_000


def _estimate_tokens(msg_list: list) -> int:
    """Rough char-based estimate: ~4 chars/token. Schnell + ohne extra Dep."""
    total = 0
    for m in msg_list:
        content = m.get("content", "")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    total += len(str(c.get("text", "")))
                elif isinstance(c, str):
                    total += len(c)
        else:
            total += len(str(content))
        # Tool-calls dranzählen (Args sind JSON-strings)
        for tc in m.get("tool_calls", []) or []:
            try:
                total += len(json.dumps(tc, default=str))
            except Exception:
                pass
    return total // 4


def _truncate_history_for_budget(messages: list, budget: int = HISTORY_TOKEN_BUDGET) -> list:
    """Trim älteste history-Messages bis Token-Budget passt.

    Invarianten:
      - messages[0] (system) immer behalten
      - messages[-1] (neue user-message) immer behalten
      - PAIR-AWARE drop: assistant mit tool_calls + alle zugehörigen tool-Replies
        werden ATOMAR gedropt (nie nur halb), egal wo im middle die liegen.
      - Sonst: einzelne Message droppen.
      - Orphaned tool-Messages am middle-start werden mit-gedropt.
    """
    if not messages or len(messages) <= 2:
        return messages
    if _estimate_tokens(messages) <= budget:
        return messages

    system = messages[0]
    last = messages[-1]
    middle = list(messages[1:-1])
    dropped = 0

    def _drop_one_unit():
        """Droppt eine atomare Einheit vom Anfang von middle.

        Wenn middle[0] assistant mit tool_calls ist: drop assistant + alle
        nachfolgenden tool-msgs die zu seinen tool_call.ids gehören.
        Sonst: drop middle[0].
        Plus: orphaned tools am neuen Anfang mit-droppen.
        """
        nonlocal middle, dropped
        if not middle:
            return
        first = middle[0]
        if first.get("role") == "assistant" and first.get("tool_calls"):
            # Atomar: assistant + alle zugehörigen tool-Replies
            expected_ids = set()
            for tc in first.get("tool_calls") or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id:
                    expected_ids.add(tc_id)
            middle = middle[1:]
            dropped += 1
            # Alle direkt-folgenden tool-msgs die zu unseren ids gehören weg
            while middle and middle[0].get("role") == "tool":
                tc_id = middle[0].get("tool_call_id")
                if tc_id in expected_ids:
                    middle = middle[1:]
                    dropped += 1
                    expected_ids.discard(tc_id)
                else:
                    # Fremder tool-reply (orphan oder anderes Pair) → auch weg
                    middle = middle[1:]
                    dropped += 1
        else:
            middle = middle[1:]
            dropped += 1
            # Orphaned tools am neuen Anfang mit-droppen
            while middle and middle[0].get("role") == "tool":
                middle = middle[1:]
                dropped += 1

    while middle and _estimate_tokens([system] + middle + [last]) > budget:
        before_len = len(middle)
        _drop_one_unit()
        if len(middle) == before_len:
            # Defensive: kein Fortschritt → Notfall-break um Endlos-Loop zu verhindern
            break

    if dropped:
        log.warning(f"history-truncate: {dropped} alte Messages gecuttet (Budget {budget} tokens)")
    return [system] + middle + [last]


async def _llm_call_with_retry(client, **kwargs):
    """Wrapper um client.chat.completions.create mit Retry+Backoff.

    Bei retry-würdigem Error: bis zu 3 Versuche, exponential backoff.
    Bei nicht-retry-würdigem Error: sofort raise.
    """
    last_exc = None
    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            return await asyncio.to_thread(client.chat.completions.create, **kwargs)
        except Exception as e:
            last_exc = e
            if not _is_retriable_llm_error(e):
                raise
            if attempt < LLM_RETRY_ATTEMPTS - 1:
                delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(
                    f"LLM-Call fehlgeschlagen (attempt {attempt+1}/{LLM_RETRY_ATTEMPTS}, "
                    f"retry in {delay}s): {type(e).__name__}: {str(e)[:200]}"
                )
                await asyncio.sleep(delay)
    # Alle Versuche aufgebraucht
    raise last_exc


async def llm_loop(user_text: str, user_id: int) -> str:
    """Run tool-use loop until final answer or limit reached.

    Mit Conversation-Memory: letzte ~12 Turns werden als Context übergeben.
    """
    # SYSTEM_PROMPT bleibt UNVERÄNDERT als statischer Block — kein .replace() mehr.
    # Anthropic Prompt-Caching (cache_control) braucht einen identischen Prefix
    # damit der Cache greift. {today}/{now} im Prefix würde den Cache jede
    # Minute invalidieren → de facto 0% Hit-Rate. Lösung: dynamische Sachen
    # (Datum/Uhrzeit + Memory + aktives Projekt) ans ENDE als separater Block,
    # NACH dem cache_control-Marker. So bleibt der Prefix stabil, Cache greift.
    sys_text = SYSTEM_PROMPT  # statischer Cache-Prefix
    now_local = datetime.now(TIMEZONE)
    tz_str = TIMEZONE.key if hasattr(TIMEZONE, "key") else str(TIMEZONE)

    # Dynamischer Block — ändert sich pro Call, nicht gecacht.
    # Wochentag EXPLIZIT setzen — LLMs (auch Sonnet) berechnen Datum→Wochentag
    # oft falsch. User-Bug 2026-05-04: Bot dachte Mo sei So weil ISO-Datum
    # alleine nicht reicht. Plus DE-Format für menschliche Lesbarkeit.
    _WD_DE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
              "Freitag", "Samstag", "Sonntag"]
    _MO_DE = ["", "Januar", "Februar", "März", "April", "Mai", "Juni",
              "Juli", "August", "September", "Oktober", "November", "Dezember"]
    wd_name = _WD_DE[now_local.weekday()]
    de_date = f"{now_local.day:02d}. {_MO_DE[now_local.month]} {now_local.year}"
    dynamic_block = (
        f"\n\n# AKTUELLER ZUSTAND\n\n"
        f"Heute ist **{wd_name}, {de_date}** "
        f"(ISO: {today_iso()}), jetzt {now_local.strftime('%H:%M')} ({tz_str}).\n"
    )

    # ─── Memory-Tiers in dynamischen Block einspeisen ───
    prefs = get_preferences()
    if prefs:
        dynamic_block += f"\n# PRÄFERENZEN (Stil/Tonalität — befolge diese)\n\n{prefs}\n"
    facts = get_facts()
    if facts:
        dynamic_block += f"\n# PERSISTENTE FAKTEN (Hintergrund über Julius)\n\n{facts}\n"
    active_proj = get_active_project()
    if active_proj:
        proj_ctx = await get_project_context(active_proj)
        if proj_ctx:
            dynamic_block += f"\n# AKTIVES PROJEKT: {active_proj}\n\n{proj_ctx}\n"
        else:
            dynamic_block += f"\n# AKTIVES PROJEKT: {active_proj} (keine CONTEXT.md gesetzt)\n"

    # System-Prompt + History + neue User-Message
    history = await get_history(user_id)
    new_user_msg = {"role": "user", "content": user_text}
    # Provider-aware System-Message-Format:
    # Anthropic akzeptiert content-as-list mit cache_control (Prompt-Caching).
    # WICHTIG: Cache greift nur wenn der Prefix-Block IDENTISCH bleibt.
    # → Statisches sys_text als Block 1 mit cache_control (gecacht).
    # → Dynamischer Block (Datum/Uhrzeit/Memory) als Block 2 ohne cache_control
    #   (jeden Call frisch, kostet vollen Preis aber ist klein).
    # Gemini/OpenAI/Ollama erwarten content als plain String → konkateniert.
    if USE_ANTHROPIC_CACHE:
        system_msg = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": sys_text,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": dynamic_block,
                },
            ],
        }
    else:
        system_msg = {"role": "system", "content": sys_text + dynamic_block}
    messages = [system_msg] + history + [new_user_msg]

    # Token-Budget-Truncation: bei langen Sessions würde messages das
    # Context-Window von Ollama (128k) sprengen. Wir trimmen ältere
    # history-Messages BEFORE jedem LLM-Call (system + neueste user
    # bleiben immer drin).
    messages = _truncate_history_for_budget(messages)

    # Diese Messages werden am Ende zur History dazugefügt
    new_history_msgs = [new_user_msg]

    # Iterations-Limit: 25 Iterationen reichen auch für Multi-File-Workflows
    # (6 Uploads → Projekt anlegen → bulk-move → activate ≈ 4 Calls bei Bulk-Tools).
    # Wenn LLM single-shot statt bulk arbeitet, fängt's spätestens bei 15 mit
    # einem Hint zu Bulk-Tools auf, statt blind weiterzumachen.
    LOOP_LIMIT = 25
    BULK_HINT_AT = 15
    bulk_hint_sent = False
    completed_tool_calls = []  # für End-Of-Loop-Error-Message
    # Self-Healing: wenn dasselbe Tool 3× in Folge fehlschlägt, abbrechen
    # statt blind weiter zu loopen. Vermeidet Frust + Token-Verschwendung.
    SELF_HEAL_FAIL_THRESHOLD = 3
    consecutive_failures: dict = {}  # tool_name → fail-count in Folge

    for iteration in range(LOOP_LIMIT):
        # Innerhalb des Loops können messages durch Tool-Outputs wachsen
        # (z.B. read_file mit 8KB content). Vor jedem LLM-Call re-trim.
        messages = _truncate_history_for_budget(messages)
        # Sanitize-on-Send: Falls irgendwo eine kaputte tool-msg ohne
        # name oder eine assistant-msg mit tool_calls aber content=None
        # entstanden ist → reparieren bevor sie an den Provider geht.
        # Idempotent + günstig, schützt vor Provider-400ern (besonders Gemini).
        messages = _sanitize_loaded_history(messages)
        resp = await _llm_call_with_retry(
            llm,
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048,
        )
        # Token-Usage tracken (best-effort, nie crash-causing)
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                _track_usage(LLM_MODEL,
                             getattr(usage, "prompt_tokens", 0),
                             getattr(usage, "completion_tokens", 0),
                             kind="chat")
        except Exception:
            pass
        msg = resp.choices[0].message
        msg_dict = msg.model_dump(exclude_none=True)
        messages.append(msg_dict)
        new_history_msgs.append(msg_dict)

        if not msg.tool_calls:
            await update_history(user_id, new_history_msgs)
            return msg.content or "(keine Antwort)"

        for tc in msg.tool_calls:
            tool_failed = False  # für Self-Healing-Counter
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                handler = TOOL_HANDLERS.get(tc.function.name)
                if not handler:
                    result = f"Tool nicht bekannt: {tc.function.name}"
                    tool_failed = True
                else:
                    log.info(f"tool[{iteration+1}/{LOOP_LIMIT}]: {tc.function.name}({args})")
                    # Handler kann sync ODER async sein (MCP-Wrapper sind async).
                    # KRITISCH bei sync: Threadpool damit lokale Hot-Pfade
                    # (backup_vault/clip_url/_build_link_index/extract_pdf_text)
                    # den Telegram-Event-Loop nicht blockieren.
                    # PLUS: hard timeout pro Tool egal was es tut.
                    tool_start = time.time()
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            # Async handler (z.B. MCP-Thin-Client) — direkt awaiten,
                            # KEIN to_thread (sonst kommt unawaited coroutine zurueck)
                            result = await asyncio.wait_for(
                                handler(**args), timeout=TOOL_TIMEOUT_SEC
                            )
                        else:
                            result = await asyncio.wait_for(
                                asyncio.to_thread(handler, **args),
                                timeout=TOOL_TIMEOUT_SEC,
                            )
                    except asyncio.TimeoutError:
                        result = (f"Tool-Timeout: `{tc.function.name}` lief länger "
                                  f"als {TOOL_TIMEOUT_SEC}s und wurde abgebrochen.")
                        tool_failed = True
                        log.warning(f"tool[{tc.function.name}] timeout after {TOOL_TIMEOUT_SEC}s")
                    else:
                        elapsed = time.time() - tool_start
                        if elapsed > 5:
                            log.info(f"tool[{tc.function.name}] dauerte {elapsed:.1f}s")
                        completed_tool_calls.append(tc.function.name)
                        # Heuristic: Tool-Result der mit "Fehler"/"failed" anfängt → fail
                        if isinstance(result, str) and re.match(
                            r"^(Fehler|❌|Pfad-Fehler|Tool nicht|Ungültig|Tool-Timeout)", result
                        ):
                            tool_failed = True
            except Exception as e:
                log.exception(f"Tool {tc.function.name} failed")
                # Token/Credential-Maskierung — Error landet in LLM-History
                # und ggf in User-Reply, darf keine API-Keys leaken
                result = _sanitize_error(f"Tool-Fehler: {e}")
                tool_failed = True
            # Self-Healing: track consecutive failures pro Tool
            tn = tc.function.name
            if tool_failed:
                consecutive_failures[tn] = consecutive_failures.get(tn, 0) + 1
            else:
                consecutive_failures[tn] = 0  # Erfolg → Counter reset
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                # name-Feld nötig für Gemini's OpenAI-Kompat-Endpoint —
                # OpenRouter/Ollama akzeptieren auch ohne, Gemini wirft 400.
                "name": tc.function.name,
                "content": str(result),
            }
            messages.append(tool_msg)
            new_history_msgs.append(tool_msg)
            # Hard-Break wenn ein Tool 3× in Folge failed
            if consecutive_failures.get(tn, 0) >= SELF_HEAL_FAIL_THRESHOLD:
                await update_history(user_id, new_history_msgs)
                return (
                    f"⚠️ Tool `{tn}` ist {SELF_HEAL_FAIL_THRESHOLD}× in Folge fehlgeschlagen — "
                    f"breche ab statt weiter zu loopen.\n\n"
                    f"Letzter Fehler: {str(result)[:300]}\n\n"
                    f"Bitte prüfe ob das Tool ein Problem hat oder formuliere die Anfrage anders."
                )

        # Hint einschleusen wenn LLM bei langen Loops noch single-shot arbeitet.
        # Seit Tool-Konsolidierung gibt's nur noch `move` — mehrfach-Single-Calls
        # erkennt man am gleichen Tool-Namen mit string-src statt list.
        if iteration + 1 == BULK_HINT_AT and not bulk_hint_sent:
            move_count = sum(1 for c in completed_tool_calls if c == "move")
            if move_count >= 4:
                hint = {
                    "role": "user",
                    "content": (
                        f"[System-Hint]: Du hast bereits {move_count}× move einzeln "
                        f"aufgerufen. Nutze `move(src=[liste], dst='ordner/')` für Bulk in "
                        f"EINEM Call. Schließe die Operation jetzt zügig ab, sonst reicht "
                        f"das Iterations-Limit nicht."
                    ),
                }
                messages.append(hint)
                new_history_msgs.append(hint)
                bulk_hint_sent = True

    # Loop-Limit erreicht — User erfährt was tatsächlich passiert ist
    await update_history(user_id, new_history_msgs)
    summary = ", ".join(f"{n}×{t}" for t, n in
                        sorted({c: completed_tool_calls.count(c)
                                for c in set(completed_tool_calls)}.items(),
                               key=lambda x: -x[1]))
    return (
        f"⚠️ Operation zu komplex für eine Iteration ({LOOP_LIMIT} Tool-Calls verbraucht).\n\n"
        f"Bisher gemacht: {summary or '(nichts erfolgreich)'}.\n\n"
        f"Der Auftrag ist möglicherweise nur teilweise erledigt — "
        f"bitte prüfen oder nochmal mit präziserem/kleinerem Auftrag wiederholen. "
        f"Tipp: `move(src=[liste], dst='ordner/')` für Bulk statt 6× einzeln."
    )


# ============================================================================
# Telegram handlers
# ============================================================================

def require_auth(handler):
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id if update.effective_user else None
        if uid is None:
            return
        # Setup-Modus: noch kein User gebunden → Bot meldet User-ID, sonst nichts
        if ALLOWED_USER_ID == 0:
            log.info(f"Setup mode: first contact from user_id={uid}")
            await update.message.reply_text(
                f"🔓 <b>Setup-Modus</b>\n\n"
                f"Bot ist noch nicht an einen User gebunden.\n\n"
                f"Deine Telegram-User-ID: <code>{uid}</code>\n\n"
                f"Auf VPS:\n"
                f"<pre><code>nano /opt/bot/.env\n"
                f"# ALLOWED_USER_ID={uid} setzen\n"
                f"docker compose restart</code></pre>\n"
                f"Danach bin ich nur noch für dich da.",
                parse_mode=constants.ParseMode.HTML,
            )
            return
        if uid != ALLOWED_USER_ID:
            log.warning(f"Unauthorized access attempt: user_id={uid}")
            return
        return await handler(update, ctx)
    return wrapper


def _esc_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Telegram-Mobile rendert ~36-40 Zeichen pro Zeile in Pre-Blocks ohne Umbruch.
# Tabellen die breiter sind verlieren Spalten-Ausrichtung → unleserlich.
TELEGRAM_TABLE_MAX_WIDTH = 38


def _render_md_table_html(table_text: str) -> str:
    """Markdown-Tabelle → Telegram-passendes HTML.

    Strategie:
      - schmale 2-Spalten-Tabellen (≤TELEGRAM_TABLE_MAX_WIDTH gesamt) → "Key: Value"-Liste
      - schmale n-Spalten-Tabellen (≤TELEGRAM_TABLE_MAX_WIDTH) → monospaced <pre>
      - sonst → Sektions-Layout: pro Row ein Block mit Bold-Header + Bullets je Spalte
    """
    lines = [l for l in table_text.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return f"<pre><code>{_esc_html(table_text)}</code></pre>"

    def cells(line: str) -> list:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    rows = [cells(l) for l in lines[2:]]  # lines[1] ist Separator
    n = len(header)
    rows = [r[:n] + [""] * max(0, n - len(r)) for r in rows]

    widths = [len(c) for c in header]
    for r in rows:
        for i in range(n):
            widths[i] = max(widths[i], len(r[i]))

    total_width = sum(widths) + (n - 1) * 3  # " │ "-Separatoren

    # ─── A) Schmal genug für monospaced Pre? Behält Tabellen-Look. ───
    if total_width <= TELEGRAM_TABLE_MAX_WIDTH:
        def fmt_row(cells_):
            return " │ ".join(c.ljust(widths[i]) for i, c in enumerate(cells_))
        sep = "─┼─".join("─" * w for w in widths)
        out = [fmt_row(header), sep] + [fmt_row(r) for r in rows]
        return f"<pre><code>{_esc_html(chr(10).join(out))}</code></pre>"

    # ─── B) Genau 2 Spalten? Kompakte "Key: Value"-Liste. ───
    if n == 2:
        out_lines = []
        for r in rows:
            key, val = r[0], r[1]
            if not key and not val:
                continue
            out_lines.append(f"<b>{_esc_html(key)}:</b> {_esc_html(val)}")
        return "\n".join(out_lines)

    # ─── C) ≥3 Spalten & breit → Sektions-Layout. ───
    # Erste Spalte = Item-Header (bold), restliche = Bullet-Liste mit "Header: Wert".
    out_lines = []
    for r in rows:
        primary = r[0].strip()
        if not primary and not any(c.strip() for c in r[1:]):
            continue
        out_lines.append(f"<b>{_esc_html(primary or '—')}</b>")
        for i in range(1, n):
            val = r[i].strip()
            if not val:
                continue
            col_header = header[i].strip() if i < len(header) else ""
            if col_header:
                out_lines.append(f"  • <b>{_esc_html(col_header)}:</b> {_esc_html(val)}")
            else:
                out_lines.append(f"  • {_esc_html(val)}")
        out_lines.append("")  # Leerzeile zwischen Items
    # Trailing-Leerzeile abschneiden
    while out_lines and not out_lines[-1]:
        out_lines.pop()
    return "\n".join(out_lines)


# Markdown-Tabelle: Header-Zeile + Separator-Zeile (nur -:| und Spaces) + 1+ Datenzeilen
TABLE_RE = re.compile(
    r"(^\|[^\n]+\|[ \t]*\n"            # Header
    r"\|[ \t]*[-:][\-:| \t]*\|[ \t]*\n"  # Separator
    r"(?:\|[^\n]+\|[ \t]*\n?)+)",      # Daten (1 oder mehr)
    re.MULTILINE,
)


def md_to_telegram_html(text: str) -> str:
    """Konvertiere Markdown → Telegram-kompatibles HTML.

    Telegram-Subset: <b>, <i>, <u>, <s>, <a>, <code>, <pre>, <blockquote>.
    Block-Konstrukte (Tabellen, Listen, Headings) werden zu Inline-Formaten:
    - Tabellen → monospaced <pre> mit Box-Drawing-Chars
    - Headings → <b>
    - Bullets → Unicode •
    - Numbered Lists → bleiben "1. text"
    - Horizontale Linien → ━━━━━━━━━━━━
    """
    # Stash für bereits-fertiges HTML, das nicht weiter verarbeitet werden soll
    stash = []

    def add_stash(html_fragment: str) -> str:
        stash.append(html_fragment)
        return f"\x00S{len(stash)-1}\x00"

    # 0) LLM-HTML-Tags zu Newlines/Plain umwandeln, BEVOR escape rennt.
    #    gpt-oss & Co. bauen trotz "NIE HTML"-Prompt-Regel <br>, <p>, <li> etc.
    #    rein. Würde später als &lt;br&gt; im Telegram landen → unleserlich.
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?div\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li\s*/?>", "\n• ", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[ou]l\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<h[1-6]\s*/?>", "\n**", text, flags=re.IGNORECASE)
    text = re.sub(r"</h[1-6]>", "**\n", text, flags=re.IGNORECASE)
    # Kollabiere ≥3 Newlines zu max 2 (sonst tonnen Leerzeilen)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 1) TABELLEN — Renderer entscheidet je nach Breite/Spaltenzahl:
    #    schmal → monospaced <pre>, 2-Spalten breit → "Key: Value"-Liste,
    #    ≥3 Spalten breit → Sektions-Layout. Renderer liefert fertiges HTML.
    def _table_repl(m):
        return add_stash(_render_md_table_html(m.group(1)))
    text = TABLE_RE.sub(_table_repl, text)

    # 2) FENCED CODE BLOCKS (```...```)
    def _fenced_repl(m):
        return add_stash(f"<pre><code>{_esc_html(m.group(2))}</code></pre>")
    text = re.sub(r"```(\w+)?\n?(.*?)```", _fenced_repl, text, flags=re.DOTALL)

    # 3) INLINE CODE (`...`)
    def _inline_repl(m):
        return add_stash(f"<code>{_esc_html(m.group(1))}</code>")
    text = re.sub(r"`([^`\n]+)`", _inline_repl, text)

    # 4) Restlichen Text HTML-escapen
    text = _esc_html(text)

    # 5) Headings (# bis ######) → <b> + Newline davor für visuelle Trennung
    text = re.sub(r"^#{1,6}\s+(.+?)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # 6) Bold/Italic/Strike
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__([^_\n]+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"~~([^~\n]+?)~~", r"<s>\1</s>", text)

    # 7) Blockquotes (> text)  — beachte: > wurde zu &gt; escaped
    text = re.sub(
        r"(?:^&gt;\s?.+(?:\n|$))+",
        lambda m: "<blockquote>" + re.sub(r"^&gt;\s?", "", m.group(0), flags=re.MULTILINE).rstrip() + "</blockquote>\n",
        text,
        flags=re.MULTILINE,
    )

    # 8) Horizontale Linie (--- oder *** allein auf Zeile)
    text = re.sub(r"^[-*_]{3,}\s*$", "━" * 24, text, flags=re.MULTILINE)

    # 9) Links — Telegram akzeptiert nur absolute URLs in <a href>.
    # Echte http/https-Links → klickbar. Relative Pfade (../foo.md) → nur Text kursiv,
    # damit kein doppelter Markdown-Salat in Telegram entsteht.
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<a href="\2">\1</a>',
        text,
    )
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"<i>\1</i>", text)

    # 10) Wikilinks [[id]] → in <code> einpacken damit Telegram nicht auto-linkt
    # (z.B. wenn id "2026-04-25.md" enthält, würde Telegram .md → URL machen)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"<code>[[\1]]</code>", text)

    # 11) Bullet-Listen mit Indentation → Unicode • mit erhaltener Einrückung
    def _bullet_repl(m):
        indent = m.group(1)
        content = m.group(2)
        # Verschachtelte Bullets: 2 Leerzeichen pro Ebene → ◦ statt •
        depth = len(indent) // 2
        marker = "•" if depth == 0 else ("◦" if depth == 1 else "▪")
        return f"{indent}{marker} {content}"
    text = re.sub(r"^([ \t]*)[-*+]\s+(.+)$", _bullet_repl, text, flags=re.MULTILINE)

    # 12) Numbered Lists — bleiben als "1. text", aber Spaces normalisieren
    text = re.sub(r"^([ \t]*)(\d+)\.\s+(.+)$", r"\1\2. \3", text, flags=re.MULTILINE)

    # 13) Stashed HTML restoren
    def _restore(m):
        return stash[int(m.group(1))]
    text = re.sub(r"\x00S(\d+)\x00", _restore, text)

    return text


def _strip_html(text: str) -> str:
    """Fallback: entferne HTML-Tags für Plain-Text-Send."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text


def _strip_paren_wikilinks(text: str) -> str:
    """Entfernt redundante Slug-Wikilinks in Klammern: 'Foo ([[t-foo]])' → 'Foo'.

    Hintergrund: Bei Status-Bestätigungen (mark_task_done/delete/move) hängt
    der LLM trotz Prompt-Regel oft noch '([[t-slug]])' an den Titel — purer
    Lärm, weil der User den Task gerade selbst genannt hat. Wir strippen das
    deterministisch raus. Bare Wikilinks (nicht in Klammern) bleiben erhalten,
    weil das echte Klick-Anker zu Neu-Erstellungen sind.
    """
    if not text:
        return text
    # `([[...]])` mit optional Whitespace davor — Multi-line safe
    return re.sub(r"[ \t]*\(\[\[[^\[\]\n]+\]\]\)", "", text)


def _safe_split_html(html: str, max_len: int = None) -> list[str]:
    """Splittet HTML in Telegram-konforme Chunks ohne offene Tags zu hinterlassen.

    Strategie:
    - Cut-Position wird so gewählt dass keine `<pre>`/`<code>`-Range straddled wird
      (sonst Telegram-400 oder optisch kaputter Code-Block)
    - Bevorzugt Splits an Doppel-Newline > Newline > Space
    """
    if max_len is None:
        max_len = TG_MAX_MESSAGE

    if len(html) <= max_len:
        return [html] if html else []

    # Pre-compute alle <pre>...</pre>-Ranges (start, end)
    # Cut darf NICHT zwischen pre_start und pre_end liegen
    def _find_protected_ranges(text: str) -> list[tuple[int, int]]:
        ranges = []
        for m in re.finditer(r"<pre\b[^>]*>.*?</pre>", text, flags=re.DOTALL):
            ranges.append((m.start(), m.end()))
        # auch <code>...</code> (single-line) protect, weil Cut mitten
        # im code-block die Escaping-Symmetrie kaputt macht
        for m in re.finditer(r"<code\b[^>]*>.*?</code>", text, flags=re.DOTALL):
            ranges.append((m.start(), m.end()))
        return ranges

    def _is_in_range(pos: int, ranges: list[tuple[int, int]]) -> Optional[int]:
        """Return range_start wenn pos innerhalb einer protected range, sonst None."""
        for r_start, r_end in ranges:
            if r_start < pos < r_end:
                return r_start
        return None

    # Mindest-Cut-Position: 25% von max_len (verhindert winzige Chunks),
    # aber mindestens 50 Zeichen (für Tests mit kleinem max_len)
    min_cut = max(50, max_len // 4)

    chunks = []
    remaining = html
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        ranges = _find_protected_ranges(remaining)

        # Strategie:
        # 1) Bevorzuge Cut an \n\n / \n / Space (in dieser Prio-Reihenfolge)
        # 2) Cut-Position muss >= min_cut sein (kein winziger Chunk)
        # 3) Cut-Position darf NICHT innerhalb einer <pre>/<code>-Range liegen
        # 4) Wenn Range straddled wird: vor den Range-Start zurückziehen
        cut = -1
        for sep in ("\n\n", "\n", " "):
            candidate = remaining.rfind(sep, 0, max_len)
            if candidate < min_cut:
                continue
            range_start = _is_in_range(candidate, ranges)
            if range_start is not None:
                # In protected range → vor den Range-Start (wenn der weit genug ist)
                if range_start >= min_cut:
                    cut = range_start
                    break
                continue  # zu früh, nächster Separator
            cut = candidate
            break

        # Last resort: vor irgendeine Range, die im Cut-Bereich endet
        if cut < min_cut:
            for r_start, r_end in ranges:
                if r_start >= min_cut and r_start < max_len:
                    cut = r_start
                    break

        if cut < 1:
            # Härte-Notfall: harter Cut. Plain-Fallback im Send catched
            # mögliche Tag-Imbalance.
            cut = max_len

        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    return chunks


async def _send_split_html(send_fn, text: str, is_html: bool = False) -> None:
    """Schickt text in mehreren Chunks via send_fn.

    is_html=False (Default): text wird als Markdown interpretiert und konvertiert.
    is_html=True: text ist BEREITS Telegram-konformes HTML, nur splitten + senden.

    Hintergrund: compute_briefing baut sein HTML manuell (um spezielle Layouts
    zu kontrollieren), Tool-Outputs sind dagegen Markdown. Eine Funktion für
    beide Fälle, klares Flag.

    send_fn muss eine async-callable sein, die (text=str, parse_mode,
    disable_web_page_preview) akzeptiert. Bei Fehler ohne parse_mode
    (Plain) erneut versuchen.
    """
    if not text:
        await send_fn(text="(leer)")
        return

    if is_html:
        html = text
    else:
        # Noise-Wikilinks rausstrippen, BEVOR Markdown→HTML konvertiert
        text = _strip_paren_wikilinks(text)
        html = md_to_telegram_html(text)

    chunks = _safe_split_html(html)

    for i, chunk in enumerate(chunks, 1):
        prefix = f"({i}/{len(chunks)})\n" if len(chunks) > 1 else ""
        try:
            await send_fn(
                text=prefix + chunk,
                parse_mode=constants.ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except Exception as e:
            log.warning(f"HTML-Send fehlgeschlagen, Fallback Plain: {e}")
            await send_fn(text=prefix + _strip_html(chunk))


async def safe_reply(update: Update, text: str, is_html: bool = False) -> None:
    """Split + send via update.message.reply_text.

    is_html=True wenn text bereits HTML ist (z.B. compute_briefing-Output).
    """
    await _send_split_html(update.message.reply_text, text, is_html=is_html)


async def safe_send(bot, chat_id: int, text: str, is_html: bool = False) -> None:
    """Split + send via bot.send_message — für Job-Callbacks ohne Update.

    is_html=True wenn text bereits HTML ist (z.B. compute_briefing-Output).
    """
    async def _send(text: str, **kwargs):
        await bot.send_message(chat_id=chat_id, text=text, **kwargs)
    await _send_split_html(_send, text, is_html=is_html)


def _detect_pending_reply_intent(text: str) -> Optional[str]:
    """Erkennt ob User-Message eine Antwort auf pending Memory/Health-Liste ist.

    Returns: 'memory' oder 'health' oder None.
    Heuristik:
    - "memory <antwort>" / "memory: <antwort>" → memory
    - "health <antwort>" → health
    - Sonst nur wenn EXAKT ein Pending-File existiert (Disambig)
      UND Text matches typisches Reply-Pattern (Zahlen, "alle", "nein", "0", "ja")
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    # Explizite Präfixe
    if t.startswith(("memory ", "memory:")) or t == "memory":
        return "memory"
    if t.startswith(("health ", "health:")) or t == "health":
        return "health"
    # Typisches Reply-Pattern: nur Zahlen+Spaces+Komma, "alle", "ja", "nein", "0", "skip", "erkläre N"
    is_reply_shape = bool(
        re.fullmatch(r"\s*\d+(\s*[,\s]\s*\d+)*\s*", t)        # "1 2 3" / "1,3"
        or t in ("alle", "ja", "all", "yes", "y",
                 "nein", "no", "n", "0", "skip", "verwerfen")
        or re.match(r"^(erklär|erklar)", t)                    # "erkläre 2"
    )
    if not is_reply_shape:
        return None
    # Disambig: welches Pending-File existiert?
    has_mem = PENDING_SUGGESTIONS_FILE.exists()
    has_health = False  # health-Block entfernt in Bot v2
    if has_mem and not has_health:
        return "memory"
    if has_health and not has_mem:
        return "health"
    if has_mem and has_health:
        # Beide pending — ohne Präfix nicht eindeutig, lass LLM entscheiden
        return None
    return None


def _strip_intent_prefix(text: str) -> str:
    """Entfernt 'memory '/'health '-Präfix vom User-Text vor Action-Parser."""
    t = text.strip()
    for prefix in ("memory:", "memory ", "memory", "health:", "health ", "health"):
        if t.lower().startswith(prefix):
            return t[len(prefix):].strip()
    return t


@require_auth
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    log.info(f"text: {text[:120]}")
    await update.message.chat.send_action(constants.ChatAction.TYPING)

    # Intent-Detection: ist das eine Antwort auf pending Memory/Health-Liste?
    # Wenn ja: direkt den entsprechenden Action-Parser rufen (kein LLM-Roundtrip).
    intent = _detect_pending_reply_intent(text)
    if intent == "memory":
        action = _strip_intent_prefix(text) or text.strip()
        try:
            reply = await asyncio.to_thread(apply_memory_suggestion, action)
        except Exception as e:
            log.exception("apply_memory_suggestion failed")
            reply = _sanitize_error(f"Fehler: {e}")
        await safe_reply(update, reply)
        return
    if intent == "health":
        action = _strip_intent_prefix(text) or text.strip()
        try:
            reply = await asyncio.to_thread(apply_health_action, action)
        except Exception as e:
            log.exception("apply_health_action failed")
            reply = _sanitize_error(f"Fehler: {e}")
        await safe_reply(update, reply)
        return

    # Pending Tagebuch-Reply (20:00 Tagebuch-Reminder wurde gepingt) — kein LLM nötig
    # User-Reply geht direkt in heutige Daily-Note unter "Abends"
    pending_diary = _load_pending_diary()
    if pending_diary:
        t = text.strip()
        # Skip-Trigger
        if t.lower() in ("skip", "nichts", "morgen", "spaeter", "später", "nicht heute", "-"):
            _clear_pending_diary()
            await safe_reply(update, "OK, kein Tagebuch-Eintrag heute.")
            return
        # Sonst: in heutige Daily unter Abends einsortieren
        try:
            ts = datetime.now(TIMEZONE).strftime("%H:%M")
            entry = f"- ({ts}) {t}"
            await mcp_thin_tools.append_to_daily(section="Abends", text=entry)
            _clear_pending_diary()
            await safe_reply(
                update,
                "Tagebuch in heutige Daily eingetragen.",
            )
            return
        except Exception as e:
            log.exception("pending-diary append failed")
            _clear_pending_diary()
            # Fall through to LLM falls append fehlschlaegt

    # Normaler Pfad: ans LLM mit Tool-Loop
    try:
        reply = await llm_loop(text, update.effective_user.id)
    except Exception as e:
        log.exception("llm_loop failed")
        reply = f"Fehler: {e}"
    await safe_reply(update, reply)


@require_auth
async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice or update.message.audio
    if not voice:
        return
    log.info(f"voice: duration={voice.duration}s")
    await update.message.chat.send_action(constants.ChatAction.TYPING)
    try:
        file = await voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
        await file.download_to_drive(tmp_path)
        try:
            segments, _ = await asyncio.to_thread(
                whisper.transcribe, tmp_path, language=WHISPER_LANG
            )
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            try: os.unlink(tmp_path)
            except Exception: pass
        if not transcript:
            await update.message.reply_text("(Sprachnachricht leer/unverständlich)")
            return
        log.info(f"transcript: {transcript[:120]}")
        await update.message.reply_text(f"Transkript: {transcript}")
        reply = await llm_loop(transcript, update.effective_user.id)
    except Exception as e:
        log.exception("voice handler failed")
        reply = f"Voice-Fehler: {e}"
    await safe_reply(update, reply)


@require_auth
async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Datei-Upload: Bot extrahiert Text, MCP create_note speichert.

    Unterstuetzt: PDF (extract_pdf_text) und DOCX (extract_docx_text).
    Andere Formate werden abgewiesen (kein Vault-Save mehr fuer beliebige
    Binaeries — Single Source of Truth ist MCP).
    """
    doc = update.message.document
    if not doc:
        return
    user_caption = update.message.caption or ""
    filename = doc.file_name or f"upload-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log.info(f"document: {filename}, size={doc.file_size}, caption={user_caption[:60]}")
    await update.message.chat.send_action(constants.ChatAction.TYPING)

    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".docx"):
        await update.message.reply_text(
            f"Nur PDF und DOCX werden unterstuetzt — `{ext}` abgewiesen. "
            "Sende den Inhalt als Text/Voice oder konvertiere lokal."
        )
        return

    tmp_path = None
    try:
        # Download nach tmp
        file = await doc.get_file()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

        # Text-Extraktion (sync, in Threadpool)
        if ext == ".pdf":
            await update.message.chat.send_action(constants.ChatAction.TYPING)
            text, meta, total = await asyncio.to_thread(extract_pdf_text, Path(tmp_path))
            kind = "paper"
            count_label = f"{total} Seiten"
        else:  # .docx
            await update.message.chat.send_action(constants.ChatAction.TYPING)
            text, meta, total = await asyncio.to_thread(extract_docx_text, Path(tmp_path))
            kind = "document"
            count_label = f"{total} Absaetze"

        title = meta.get("title") or Path(filename).stem
        author = meta.get("author") or "unknown"

        # Body fuer die Note
        body = (
            f"**Datei**: `{filename}` | **Typ**: {kind} | **Autor**: {author} | {count_label}\n\n"
            "---\n\n"
            + (text.strip() if text else "(Text-Extraktion ergab keinen Inhalt)")
        )

        # An MCP geben — handle_text-Pfad analog zu create_note Tool-Call
        try:
            res = await mcp_thin_tools.create_note(
                title=title, body=body, tags=[kind], project=None,
            )
        except Exception as e:
            log.exception("MCP create_note fuer document upload failed")
            await update.message.reply_text(f"Upload-Fehler (MCP): {e}")
            return

        # Antwort an User
        await safe_reply(
            update,
            f"<b>{kind.capitalize()}</b> gespeichert: {res}\n"
            f"<b>Inhalt (Vorschau)</b>:\n"
            f"<pre><code>{_esc_html(body[:600])}</code></pre>",
            is_html=True,
        )

        # Upload-Event in LLM-History (fuer Folge-Anweisungen wie "lege als Projekt an")
        try:
            history_msg = (
                f"[Upload-Event] User hat {kind} {filename!r} hochgeladen - "
                f"als Note via MCP gespeichert ({res})."
            )
            await update_history(update.effective_user.id, [
                {"role": "user", "content": history_msg}
            ])
        except Exception as e:
            log.warning(f"Upload-Event nicht in History: {e}")

        # Caption als LLM-Anweisung weiterreichen
        if user_caption.strip():
            await update.message.chat.send_action(constants.ChatAction.TYPING)
            try:
                llm_reply = await llm_loop(user_caption.strip(), update.effective_user.id)
                await safe_reply(update, llm_reply)
            except Exception as e:
                log.exception("Caption-LLM-Routing failed")
                await update.message.reply_text(f"Caption-Verarbeitung fehlgeschlagen: {e}")

    except Exception as e:
        log.exception("document handler failed")
        await update.message.reply_text(f"Document-Fehler: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ─── Recurring Tasks: Daily-Reset ──────────────────────────────────────────
# Tasks mit frontmatter.recurrence werden vom Reset-Job morgens reaktiviert,
# wenn sie done sind und das Pattern fällig ist. So lebt EINE Task-Datei für
# alle Wiederholungen, History sammelt sich im Log.

async def _trigger_recurring_reset() -> int:
    """Triggert MCP task_reactivate_recurring + invalidiert Bot-Cache.

    Returns Anzahl reaktivierter Tasks. Bei MCP-Fail wird geloggt aber
    keine Exception propagiert (Bot soll trotzdem weiter funktionieren).
    """
    from mcp_client import mcp as _mcp, MCPError as _MCPError
    try:
        stats = await _mcp.task_reactivate_recurring()
    except _MCPError as e:
        log.warning("recurring-reset MCP fail: %s", e)
        return 0
    reactivated = stats.get("reactivated") or [] if isinstance(stats, dict) else []
    checked = stats.get("checked", 0) if isinstance(stats, dict) else 0
    if reactivated:
        log.info("recurring-reset: %d/%d reaktiviert: %s", len(reactivated), checked, reactivated)
    else:
        log.info("recurring-reset: %d Tasks geprueft, keine faellig", checked)
    return len(reactivated)


async def recurring_task_reset_job(ctx: ContextTypes.DEFAULT_TYPE):
    """JobQueue-Callback — läuft täglich um 05:00 (oder vor Briefing).

    MCP's Maintain-Pipeline reaktiviert recurring tasks alle 10 Min auto.
    Dieser explizite Call ist Sicherheits-Net (deterministischer Tagesstart).
    """
    await _trigger_recurring_reset()


async def daily_briefing_job(ctx: ContextTypes.DEFAULT_TYPE):
    """JobQueue-Callback — wird täglich um BRIEFING_HOUR ausgefuehrt."""
    try:
        # Erst recurring Tasks reaktivieren (via MCP), dann Briefing.
        await _trigger_recurring_reset()
        text = await mcp_thin_tools.compute_briefing()
        await safe_send(ctx.bot, ALLOWED_USER_ID, text, is_html=True)
        # Push-Log in History — kompakt, nicht voller Briefing-Text
        await _log_bot_push_to_history(
            ALLOWED_USER_ID, "morning-briefing",
            "Morgens-Briefing gesendet (Tagesplan + Tasks + 5y-Goal)",
        )
        log.info(f"Daily briefing sent to {ALLOWED_USER_ID}")
    except Exception as e:
        log.exception(f"daily_briefing_job failed: {e}")
        # Versuche zumindest eine Fehler-Notification zu schicken — plain text
        # weil safe_send selbst die Ursache sein könnte. Bewusst KEIN HTML hier.
        try:
            await ctx.bot.send_message(
                chat_id=ALLOWED_USER_ID,
                text=f"⚠️ Daily-Briefing-Fehler: {type(e).__name__}\n{str(e)[:300]}",
            )
        except Exception:
            pass


def _save_pending_diary() -> None:
    """Bot hat Tagebuch-Reminder gepusht. handle_text wird User-Reply
    direkt in Daily-Note einsortieren statt LLM zu rufen.
    """
    _ensure_memory_dir()
    atomic_write(PENDING_DIARY_FILE, json.dumps({"fired_at": time.time()}))


def _load_pending_diary() -> Optional[dict]:
    if not PENDING_DIARY_FILE.exists():
        return None
    try:
        data = json.loads(PENDING_DIARY_FILE.read_text(encoding="utf-8"))
        if time.time() - float(data.get("fired_at", 0)) > PENDING_DIARY_TTL_SEC:
            PENDING_DIARY_FILE.unlink(missing_ok=True)
            return None
        return data
    except Exception:
        return None


def _clear_pending_diary() -> None:
    if PENDING_DIARY_FILE.exists():
        PENDING_DIARY_FILE.unlink(missing_ok=True)


async def _log_bot_push_to_history(user_id: int, kind: str, summary: str) -> None:
    """Persistiert einen Bot-Push (Reminder, Briefing, Anker-Ping) als
    assistant-Message in der Conversation-History.

    WHY: Bot-Pushes sind proaktiv — gehen NICHT durch llm_loop. Wenn User
    minutenspäter antwortet, weiß der LLM ohne diesen Eintrag nichts vom Push
    und behandelt die User-Message als isolierte Anfrage ('Drücke mich vor
    dem Lernen' → analysiert Prokrastination, statt als Tagebuch-Eintrag
    zu erkennen). Mit Push-Log in History sieht der LLM den Kontext.

    Format kompakt damit Token-Footprint klein bleibt — Inhalt summary,
    nicht der ganze HTML-Block.
    """
    try:
        ts = datetime.now(TIMEZONE).strftime("%H:%M")
        summary_short = (summary or "").strip()[:200]
        msg = {
            "role": "assistant",
            "content": f"[Bot-Push {ts} kind={kind}] {summary_short}",
        }
        await update_history(user_id, [msg])
    except Exception as e:
        log.warning(f"_log_bot_push_to_history failed: {e}")


@require_auth
async def handle_briefing(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/briefing — manueller Trigger fuer das Morgens-Briefing (Daten via MCP)."""
    text = await mcp_thin_tools.compute_briefing()
    await safe_reply(update, text, is_html=True)


@require_auth
async def handle_reminders(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/reminders — alle aktiven Erinnerungen listen."""
    text = await asyncio.to_thread(list_reminders)
    await safe_reply(update, text)


@require_auth
async def handle_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Setzt Conversation-Memory + pending Deletes zurueck."""
    await reset_history(ALLOWED_USER_ID)
    mcp_thin_tools.PENDING_DELETIONS.pop(ALLOWED_USER_ID, None)
    await update.message.reply_text("Memory + pending Deletes geleert. Frischer Anfang.")


@require_auth
async def handle_today(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/today — heutige Agenda + Daily-Note Body via MCP."""
    agenda = await mcp_thin_tools.get_today_agenda()
    parts = [agenda]
    # Plus heutige Daily-Note Body via MCP
    try:
        rel = f"10_Life/daily/{today_iso()}.md"
        body = await mcp_thin_tools.read_file(rel_path=rel, strip_frontmatter=True)
        if body and not body.startswith("Datei nicht gefunden") and len(body.strip()) > 10:
            parts.append("\n" + "─" * 24)
            parts.append("📓 <b>Heutige Daily-Notes:</b>\n")
            parts.append(body)
    except Exception as e:
        log.warning(f"handle_today: daily-read fehlgeschlagen: {e}")
    await safe_reply(update, "\n".join(parts), is_html=True)


@require_auth
async def handle_tasks(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/tasks [filter] — offene Tasks. Optional Filter: today/tomorrow/week/overdue/nodate."""
    args = ctx.args if ctx.args else []
    when = args[0].strip().lower() if args else None
    if when and when not in ("today", "tomorrow", "week", "overdue", "nodate"):
        await update.message.reply_text(
            "Unbekannter Filter. Erlaubt: today, tomorrow, week, overdue, nodate (oder leer = alle gruppiert)."
        )
        return
    text = await mcp_thin_tools.list_open_tasks(when=when)
    await safe_reply(update, text, is_html=True)


@require_auth
async def handle_usage(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/usage [days] — Token-Usage + geschätzte Kosten der letzten N Tage (default 7)."""
    args = ctx.args if ctx.args else []
    days = 7
    if args:
        try:
            days = max(1, min(90, int(args[0])))
        except ValueError:
            pass
    text = await asyncio.to_thread(get_usage_summary, days)
    await safe_reply(update, text)


@require_auth
async def handle_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 <b>Vault-Assistent bereit</b>\n\n"
        "Standardmäßig <b>antworte ich nur</b> — speichere nichts ins Vault, "
        "außer du sagst's mir explizit.\n\n"
        "<b>Speicher-Verben</b> die ich erkenne:\n"
        "• \"speicher / merk dir / notiere / schreib auf …\"\n"
        "• \"task: …\" / \"todo: …\" / \"morgen X machen\"\n"
        "• \"meeting: …\" / \"war im termin mit …\"\n"
        "• \"X erledigt\" → markiert Task als done\n"
        "• \"lösche X\" → fragt nach Bestätigung\n\n"
        "<b>Multimedia:</b>\n"
        "Sprachnachricht → transkribiert + sortiert\n"
        "Foto → in 09_Attachments + Vision-Caption + OCR (Tesseract de+en)\n"
        ".md/.txt → in 01_Raw/uploads/\n"
        ".pdf → in 01_Raw/papers/ + Volltext-Extraktion + .md-Wrapper für Suche\n"
        ".docx → in 01_Raw/uploads/ + Text+Tabellen-Extraktion + .md-Wrapper\n"
        "sonstige Files → in 09_Attachments\n"
        "URL allein → fragt ob clippen\n\n"
        "<b>Commands:</b>\n"
        "/today — heutige Daily anzeigen\n"
        "/briefing — Tagesbriefing (überfällig + offen + heute)\n"
        "/reminders — alle aktiven Erinnerungen\n"
        "/backup — Vault in GitHub-Repo pushen\n"
        "/reset — Conversation-Memory leeren\n\n"
        "<i>3-Tier-Memory aktiv:\n"
        "• ~30 Turns aktiver Konversation (RAM)\n"
        "• gesamte History persistiert (überlebt Restart)\n"
        "• Long-term Fakten in 06_Meta/bot-memory/facts.md (immer im Kontext)\n"
        "Sag 'merk dir dass …' um persistente Fakten anzulegen.</i>",
        parse_mode=constants.ParseMode.HTML,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    log.info(f"Vault: {VAULT}")
    if ALLOWED_USER_ID == 0:
        log.warning("⚠️  ALLOWED_USER_ID=0 → Setup-Modus aktiv. Erste Nachricht im Telegram triggert Anleitung.")
    else:
        log.info(f"Allowed user: {ALLOWED_USER_ID}")
    log.info(f"LLM: {LLM_MODEL} @ {LLM_BASE_URL}")
    if not VAULT.exists():
        log.error(f"VAULT_PATH existiert nicht: {VAULT}")
        return
    if not TEMPLATES_DIR.exists():
        log.error(f"Templates-Ordner fehlt: {TEMPLATES_DIR}")
        return

    # ─── Tool-Konsistenz-Check: TOOLS-Schema ↔ TOOL_HANDLERS-Dispatch ───
    # Bei Drift wird der Bot beim Boot abgewiesen statt erst zur Laufzeit
    # mit "Tool nicht bekannt"-Errors zu enttäuschen.
    declared = {t["function"]["name"] for t in TOOLS if t.get("function", {}).get("name")}
    handled = set(TOOL_HANDLERS.keys())
    missing_handlers = declared - handled
    orphan_handlers = handled - declared
    if missing_handlers:
        log.error(f"Tools im Schema OHNE Handler: {sorted(missing_handlers)}")
        return
    if orphan_handlers:
        log.warning(f"⚠️  Handler ohne Tool-Schema (für LLM unsichtbar): {sorted(orphan_handlers)}")
    log.info(f"Tools: {len(declared)} declared, {len(handled)} handled — alle aligned")
    app = Application.builder().token(TG_TOKEN).build()

    # Globale Referenz fuer Tools die JobQueue brauchen (z.B. create_reminder)
    global BOT_APP
    BOT_APP = app

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("today", handle_today))
    app.add_handler(CommandHandler("tasks", handle_tasks))
    app.add_handler(CommandHandler("usage", handle_usage))
    app.add_handler(CommandHandler("briefing", handle_briefing))
    app.add_handler(CommandHandler("reminders", handle_reminders))
    app.add_handler(CommandHandler("reset", handle_reset))
    # Persistente Reminders aus JSON laden + neu schedulen
    persisted_reminders = _load_reminders()
    if persisted_reminders:
        cleaned = []
        for r in persisted_reminders:
            try:
                _schedule_reminder(app, r)
                # nur behalten wenn Reminder noch aktiv (einmalige in Vergangenheit
                # werden durch _schedule_reminder aus JSON entfernt — daher reload)
                cleaned.append(r)
            except Exception as e:
                log.warning(f"Reminder {r.get('id')} konnte nicht reactiviert werden: {e}")
        log.info(f"Reminders geladen: {len(cleaned)} aus {REMINDERS_FILE}")
    else:
        log.info("Keine persistierten Reminders.")

    # Daily-Briefing JobQueue (wenn BRIEFING_HOUR > 0 gesetzt)
    if BRIEFING_HOUR and ALLOWED_USER_ID > 0:
        try:
            briefing_time = dtime(hour=BRIEFING_HOUR, minute=0, tzinfo=TIMEZONE)
            app.job_queue.run_daily(
                daily_briefing_job,
                time=briefing_time,
                name="daily-briefing",
            )
            log.info(f"Daily-Briefing scheduled für {BRIEFING_HOUR}:00 {TIMEZONE.key}")
        except Exception as e:
            log.warning(f"JobQueue-Setup fehlgeschlagen (BRIEFING_HOUR={BRIEFING_HOUR}): {e}")
    else:
        # Briefing aus, aber recurring-Reset trotzdem schedulen — so funktionieren
        # wiederkehrende Tasks auch ohne Briefing. Default-Slot: 5:00 morgens.
        if ALLOWED_USER_ID > 0:
            try:
                reset_time = dtime(hour=5, minute=0, tzinfo=TIMEZONE)
                app.job_queue.run_daily(
                    recurring_task_reset_job,
                    time=reset_time,
                    name="recurring-task-reset",
                )
                log.info(f"Recurring-Task-Reset scheduled für 05:00 {TIMEZONE.key} (Briefing ist aus)")
            except Exception as e:
                log.warning(f"Recurring-Reset-Setup fehlgeschlagen: {e}")
        log.info("Daily-Briefing deaktiviert (BRIEFING_HOUR=0 oder Setup-Modus)")

    # Bot v2: kein Health/Suggestion/Anchor-JobQueue mehr — Vault-Maintenance
    # macht der MCP alle 10 Min (vault_maintain pipeline + recurring-reset).

    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    # Boot-Banner mit aktiver Konfiguration — hilft bei Diagnose ob ENV-
    # Änderungen (z.B. Modell-Switch) wirklich beim Container ankommen.
    log.info(f"=== KI-OS Bot startet ===")
    log.info(f"  Modell:    {LLM_MODEL}")
    log.info(f"  Endpoint:  {LLM_BASE_URL}")
    log.info(f"  Vault:     {VAULT}")
    log.info(f"  Timezone:  {TIMEZONE.key}")
    log.info(f"  Anthropic-Cache: {'aktiv' if USE_ANTHROPIC_CACHE else 'aus'}")
    log.info("Polling started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
