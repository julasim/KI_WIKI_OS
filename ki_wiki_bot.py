#!/usr/bin/env python3
"""
ki_wiki_bot.py — Telegram-Bot für KI_WIKI_Vault.

Single-file, single-user. Vault-Operations via LLM-Tool-Use.
Voice → faster-whisper (lokal). Photo → Vision-LLM. URL → trafilatura.

ENV:
  TG_TOKEN, ALLOWED_USER_ID, OPENROUTER_API_KEY,
  VAULT_PATH (default /vault),
  LLM_MODEL, VISION_MODEL,
  WHISPER_MODEL (small), WHISPER_DEVICE (cpu), WHISPER_LANG (de)
"""

import os
import re
import json
import time
import base64
import shutil
import asyncio
import logging
import subprocess
import tempfile
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import frontmatter
import trafilatura
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
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-5")
# Vision kann optional ein anderer Provider sein (z.B. wenn Haupt-LLM kein Vision kann)
VISION_API_KEY = os.environ.get("VISION_API_KEY", LLM_API_KEY)
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", LLM_BASE_URL)
VISION_MODEL = os.environ.get("VISION_MODEL", LLM_MODEL)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "de")

# Daily-Briefing
try:
    BRIEFING_HOUR = int(os.environ.get("BRIEFING_HOUR", "0") or "0")
except ValueError:
    BRIEFING_HOUR = 0
TIMEZONE = ZoneInfo(os.environ.get("TIMEZONE", "Europe/Vienna"))

TG_MAX_MESSAGE = 3800

LIFE = VAULT / "10_Life"
DAILY_DIR = LIFE / "daily"
TASKS_DIR = LIFE / "tasks"
NOTES_DIR = LIFE / "notes"
MEETINGS_DIR = LIFE / "meetings"
AREAS_DIR = LIFE / "areas"
TEMPLATES_DIR = VAULT / "08_Templates"
ATTACHMENTS_DIR = VAULT / "09_Attachments"
RAW_ARTICLES_DIR = VAULT / "01_Raw" / "articles"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ki_wiki_bot")

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
# Separater Vision-Client falls Vision via anderen Provider läuft
vision_llm = (
    llm if (VISION_BASE_URL == LLM_BASE_URL and VISION_API_KEY == LLM_API_KEY)
    else OpenAI(base_url=VISION_BASE_URL, api_key=VISION_API_KEY)
)

# ============================================================================
# Whisper (einmal beim Start laden, im Speicher halten)
# ============================================================================

log.info(f"Loading Whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}...")
whisper = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type="int8")
log.info("Whisper loaded.")

# ============================================================================
# Helpers
# ============================================================================

def slugify(s: str, max_len: int = 50) -> str:
    """kebab-case slug, deutsche Umlaute behandelt."""
    s = s.lower().strip()
    s = (s.replace("ä", "ae").replace("ö", "oe")
         .replace("ü", "ue").replace("ß", "ss"))
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len] or "untitled"


def safe_path(rel_path: str) -> Path:
    """Resolve rel_path inside VAULT, prevent traversal."""
    p = (VAULT / rel_path).resolve()
    if not str(p).startswith(str(VAULT.resolve())):
        raise ValueError(f"Path traversal: {rel_path}")
    return p


def atomic_write(path: Path, content: str) -> None:
    """Write atomic: tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def today_iso() -> str:
    return date.today().isoformat()


def load_template(name: str) -> str:
    """Read template file (e.g. 'daily' → daily_template.md)."""
    return (TEMPLATES_DIR / f"{name}_template.md").read_text(encoding="utf-8")


def ocr_image(image_path: Path) -> str:
    """OCR via Tesseract (deutsch+englisch). Leerer String wenn nichts erkennbar."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return ""
    try:
        img = Image.open(str(image_path))
        text = pytesseract.image_to_string(img, lang="deu+eng").strip()
        return text
    except Exception as e:
        log.warning(f"OCR failed for {image_path}: {e}")
        return ""


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

def ensure_daily() -> Path:
    """Create today's daily file if missing, return path."""
    path = DAILY_DIR / f"{today_iso()}.md"
    if path.exists():
        return path
    today = today_iso()
    template = load_template("daily")
    content = template.replace("{{date:YYYY-MM-DD}}", today)
    post = frontmatter.loads(content)
    post["id"] = f"daily-{today}"
    post["title"] = today
    atomic_write(path, frontmatter.dumps(post) + "\n")
    log.info(f"Created daily: {path.name}")
    return path


VALID_SECTIONS = {"Heute", "Notizen & Gedanken", "Offen / Einsortieren", "Abends"}

# Platzhalter aus dem daily_template, die beim ersten Append entfernt werden sollen
EMPTY_PLACEHOLDERS = {"- [ ]", "- [x]", "-", "•", "- Was lief gut?", "- Was nehme ich mit?"}


def _clean_section_body(body: str) -> str:
    """Entfernt nur-Platzhalter-Zeilen wenn die Sektion sonst leer wäre."""
    lines = body.split("\n")
    real_content = [l for l in lines if l.strip() and l.strip() not in EMPTY_PLACEHOLDERS]
    if not real_content:
        return ""  # Sektion war komplett leer (nur Template-Platzhalter)
    return "\n".join(l for l in lines if l.strip())  # Auch leere Zeilen weg


def append_to_daily(section: str, text: str) -> str:
    """Append text to today's daily under the given section.

    Spezielle Logik: wenn Sektion nur Template-Platzhalter enthält (z.B. '- [ ]'
    aus daily_template), wird der Platzhalter ersetzt statt zusätzlich gestapelt.
    """
    if section not in VALID_SECTIONS:
        section = "Notizen & Gedanken"
    path = ensure_daily()
    content = path.read_text(encoding="utf-8")

    pattern = rf"(?m)^## {re.escape(section)}\s*$"
    match = re.search(pattern, content)
    if not match:
        # Sektion existiert nicht → am Ende anhängen
        new_content = content.rstrip() + f"\n\n## {section}\n{text}\n"
    else:
        start = match.end()
        next_h2 = re.search(r"^## ", content[start:], re.MULTILINE)
        end = start + next_h2.start() if next_h2 else len(content)

        # Aktueller Body der Sektion (ohne Header)
        body_before_h = content[:start]
        body_section = content[start:end].strip("\n")
        body_after = content[end:]

        # Body von Platzhaltern bereinigen
        cleaned = _clean_section_body(body_section)

        # Neuen Text anfügen
        if cleaned:
            new_section_body = cleaned + "\n" + text
        else:
            new_section_body = text

        new_content = (
            body_before_h.rstrip("\n")
            + "\n" + new_section_body
            + "\n\n"
            + body_after.lstrip("\n")
        )

    post = frontmatter.loads(new_content)
    post["updated"] = today_iso()
    atomic_write(path, frontmatter.dumps(post) + "\n")
    return f"In Daily ({section}) eingetragen: {path.name}"


VALID_PRIORITIES = {"low", "medium", "high", "urgent"}
VALID_TASK_CONTEXTS = {"home", "work", "errand", "phone", "computer"}


def create_task(title: str, priority: str = "medium",
                due: Optional[str] = None, area: Optional[str] = None,
                project: Optional[str] = None, context: Optional[str] = None,
                tags: Optional[list] = None) -> str:
    """Create a new task file in 10_Life/tasks/.

    Body wird dynamisch gebaut (nicht Template-Substitution) damit
    Status/Priorität/Fälligkeit korrekt im Body erscheinen.
    """
    if not title or not title.strip():
        return "Fehler: Task-Titel darf nicht leer sein."

    # Priority normalisieren — LLM schickt manchmal Abkürzungen wie "M", "high prio" etc.
    p_lower = (priority or "").lower().strip()
    p_map = {
        "u": "urgent", "urgent": "urgent",
        "h": "high", "high": "high",
        "m": "medium", "medium": "medium", "med": "medium", "normal": "medium",
        "l": "low", "low": "low",
    }
    priority = p_map.get(p_lower, "medium")

    # Due nur akzeptieren wenn ISO-Format
    if due:
        due = due.strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", due):
            log.warning(f"create_task: ungültiges due-Format '{due}', ignoriert")
            due = None

    slug = slugify(title)
    path = TASKS_DIR / f"{slug}.md"
    n = 2
    while path.exists():
        path = TASKS_DIR / f"{slug}-{n}.md"
        n += 1
    today = today_iso()
    task_id = f"t-{path.stem}"

    # Body dynamisch bauen (statt Template mit Hardcoded-Defaults)
    body = (
        f"# {title}\n\n"
        f"**Status**: open · **Priorität**: {priority} · **Fällig**: {due or '—'}\n\n"
        f"## Was\n\n"
        f"## Warum\n\n"
        f"## Subschritte\n- [ ]\n\n"
        f"## Notizen\n\n"
        f"## Log\n- {today}: angelegt\n"
    )

    # Tags filtern (nur strings, nicht-leer, deduplizieren)
    clean_tags = []
    if tags and isinstance(tags, list):
        seen = set()
        for t in tags:
            if isinstance(t, str) and t.strip() and t.strip() not in seen:
                clean_tags.append(t.strip().lower())
                seen.add(t.strip())

    fm_data = {
        "id": task_id,
        "title": title,
        "type": "task",
        "created": today,
        "updated": today,
        "tags": clean_tags,
        "status": "open",
        "priority": priority,
    }
    if due:
        fm_data["due"] = due
    if area:
        fm_data["area"] = area
    if project:
        fm_data["project"] = project
    if context:
        ctx_lower = context.lower().strip()
        if ctx_lower in VALID_TASK_CONTEXTS:
            fm_data["context"] = ctx_lower

    post = frontmatter.Post(body, **fm_data)
    atomic_write(path, frontmatter.dumps(post) + "\n")

    # Link in heutige Daily
    try:
        append_to_daily("Heute", f"- [ ] [[{task_id}]] {title}")
    except Exception as e:
        log.warning(f"Daily-Link für Task fehlgeschlagen: {e}")

    extras = []
    if due:
        extras.append(f"fällig {due}")
    if project:
        extras.append(f"projekt {project}")
    if priority != "medium":
        extras.append(f"prio {priority}")
    extra_str = f" ({', '.join(extras)})" if extras else ""
    return f"Task angelegt: [[{task_id}]]{extra_str}"


def mark_task_done(slug: str) -> str:
    """Mark task as done."""
    filename = slug[2:] if slug.startswith("t-") else slug
    path = TASKS_DIR / f"{filename}.md"
    if not path.exists():
        return f"Task nicht gefunden: {slug}"
    post = frontmatter.load(path)
    post["status"] = "done"
    post["updated"] = today_iso()
    body = (post.content or "").rstrip() + f"\n- {today_iso()}: erledigt\n"
    post.content = body
    atomic_write(path, frontmatter.dumps(post) + "\n")
    return f"Task erledigt: [[t-{filename}]]"


def create_meeting(title: str, attendees: Optional[list] = None,
                   meeting_date: Optional[str] = None,
                   tags: Optional[list] = None) -> str:
    """Create meeting protocol."""
    today = meeting_date or today_iso()
    slug = slugify(title)
    path = MEETINGS_DIR / f"{today}_{slug}.md"
    template = load_template("meeting")
    content = template.replace("{{title}}", title).replace("{{date:YYYY-MM-DD}}", today)
    post = frontmatter.loads(content)
    post["id"] = f"meeting-{today}-{slug}"
    post["title"] = title
    post["date"] = today
    post["attendees"] = attendees or []
    post["status"] = "done" if today <= today_iso() else "planned"
    # Auto-Tags
    clean_tags = []
    if tags and isinstance(tags, list):
        seen = set()
        for t in tags:
            if isinstance(t, str) and t.strip() and t.strip() not in seen:
                clean_tags.append(t.strip().lower())
                seen.add(t.strip())
    post["tags"] = clean_tags
    atomic_write(path, frontmatter.dumps(post) + "\n")
    return f"Meeting angelegt: [[meeting-{today}-{slug}]]"


def create_note(title: str, body: str, tags: Optional[list] = None) -> str:
    """Create a free note in 10_Life/notes/."""
    today = today_iso()
    slug = slugify(title)
    path = NOTES_DIR / f"{today}_{slug}.md"
    template = load_template("note")
    content = template.replace("{{title}}", title).replace("{{date:YYYY-MM-DD}}", today)
    post = frontmatter.loads(content)
    post["id"] = slug
    post["title"] = title
    # Tags filtern (nur strings, nicht-leer, deduplizieren, lowercase)
    clean_tags = []
    if tags and isinstance(tags, list):
        seen = set()
        for t in tags:
            if isinstance(t, str) and t.strip() and t.strip() not in seen:
                clean_tags.append(t.strip().lower())
                seen.add(t.strip())
    post["tags"] = clean_tags
    post["quelle"] = "telegram"
    post.content = (post.content or "") + "\n" + body + "\n"
    atomic_write(path, frontmatter.dumps(post) + "\n")
    return f"Notiz angelegt: [[{slug}]]"


def search_vault(query: str, limit: int = 5) -> str:
    """Volltext-Suche via vault_search.py (subprocess)."""
    script = VAULT / "07_Tools" / "search" / "vault_search.py"
    if not script.exists():
        return "vault_search.py nicht gefunden."
    try:
        result = subprocess.run(
            ["python3", str(script), "--json", query],
            capture_output=True, text=True, timeout=30, cwd=str(VAULT),
        )
        if result.returncode != 0:
            return f"Suche fehlgeschlagen: {result.stderr.strip()[:300]}"
        data = json.loads(result.stdout) if result.stdout.strip() else []
        if not data:
            return f"Keine Treffer für: {query}"
        lines = [f"Suche '{query}' — {len(data)} Treffer (top {limit}):"]
        for hit in data[:limit]:
            hid = hit.get("id", "?")
            htype = hit.get("type", "?")
            score = hit.get("score", "?")
            lines.append(f"- [[{hid}]] · `{htype}` · score {score}")
        return "\n".join(lines)
    except Exception as e:
        return f"Suche-Fehler: {e}"


def read_file(rel_path: str, strip_frontmatter: bool = True) -> str:
    """Read a file (relative to vault root). Capped at 8KB.

    strip_frontmatter=True (default): YAML-Frontmatter wird entfernt für
    saubere Anzeige. Auf False setzen wenn du Metadaten brauchst.
    """
    try:
        path = safe_path(rel_path)
        if not path.exists():
            return f"Datei nicht gefunden: {rel_path}"
        content = path.read_text(encoding="utf-8")
        if strip_frontmatter:
            # Frontmatter zwischen --- ... --- am Anfang entfernen
            content = re.sub(r"^---\n.*?\n---\n+", "", content, count=1, flags=re.DOTALL)
        return content[:8000]
    except Exception as e:
        return f"Lese-Fehler: {e}"


def edit_file(rel_path: str, find: str, replace: str, regex: bool = False) -> str:
    """Find/replace in a file."""
    try:
        path = safe_path(rel_path)
        if not path.exists():
            return f"Datei nicht gefunden: {rel_path}"
        content = path.read_text(encoding="utf-8")
        if regex:
            try:
                new, n = re.subn(find, replace, content)
            except re.error as e:
                return f"Regex-Syntax-Fehler in '{find[:50]}': {e}"
        else:
            n = content.count(find)
            new = content.replace(find, replace)
        if n == 0:
            return f"Kein Treffer für '{find[:50]}' in {rel_path}"
        atomic_write(path, new)
        return f"{n}× ersetzt in {rel_path}"
    except Exception as e:
        return f"Edit-Fehler: {e}"


def clip_url(url: str) -> str:
    """Fetch URL via trafilatura, save as raw article."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Konnte URL nicht laden: {url}"
        text = trafilatura.extract(
            downloaded, include_comments=False, include_tables=True,
            include_links=True,
        )
        meta = trafilatura.extract_metadata(downloaded)
        if not text:
            return f"Kein Inhalt extrahierbar: {url}"
        title = (meta.title if meta else None) or "untitled"
        author = (meta.author if meta else None) or "unknown"
        domain = re.sub(r"^www\.", "", re.sub(r"^https?://", "", url).split("/")[0])
        slug = slugify(title)
        today = today_iso()
        filename = f"{today}_{slugify(domain)}_{slug}.md"
        path = RAW_ARTICLES_DIR / filename
        post = frontmatter.Post(
            text,
            id=slugify(f"{domain}-{slug}"),
            title=title,
            type="article",
            source=url,
            author=author,
            captured=today,
            tags=[],
        )
        atomic_write(path, frontmatter.dumps(post) + "\n")
        try:
            append_to_daily(
                "Offen / Einsortieren",
                f"- [[{post['id']}]] *(geclipt: {title})*",
            )
        except Exception:
            pass
        return f"Artikel geclipt: [[{post['id']}]] ({filename})"
    except Exception as e:
        log.exception("clip_url failed")
        return f"Clip-Fehler: {e}"


# ─── Pending Deletions (Multi-File-fähig) ───────────────────────────────────
# Two-Step-Delete: request_delete() merkt sich Pfad(e), confirm_delete() führt aus.
# Mehrere Files können angefragt werden, alle werden bei einer Bestätigung gelöscht.
PENDING_DELETIONS: dict[int, tuple[list[str], float]] = {}
DELETE_CONFIRM_TIMEOUT = 300  # Sekunden


def request_delete(rel_paths) -> str:
    """Merkt sich eine Lösch-Anfrage (1 oder mehrere Files). Akkumuliert.

    rel_paths: str (einzelne Datei) oder list[str] (mehrere)
    """
    # Input normalisieren
    if isinstance(rel_paths, str):
        paths_in = [rel_paths]
    elif isinstance(rel_paths, list):
        paths_in = rel_paths
    else:
        return f"request_delete: ungültiger Input-Typ {type(rel_paths)}"

    # Pfade validieren + sammeln
    valid = []
    errors = []
    for rp in paths_in:
        try:
            p = safe_path(rp)
            if not p.exists():
                errors.append(f"nicht gefunden: {rp}")
            else:
                valid.append(rp)
        except Exception as e:
            errors.append(f"{rp} ({e})")

    if not valid:
        return "Keine gültigen Pfade. " + "; ".join(errors)

    # Akkumulieren (existierende pending Liste erweitern)
    existing = PENDING_DELETIONS.get(ALLOWED_USER_ID, ([], 0.0))[0]
    combined = list(dict.fromkeys(existing + valid))  # dedupliziert, Reihenfolge erhält
    PENDING_DELETIONS[ALLOWED_USER_ID] = (combined, time.time())
    log.info(f"Pending delete: {combined}")

    msg = f"⚠️ Bestätigung: soll(en) {len(combined)} Datei(en) ins Archiv verschoben werden?\n\n"
    msg += "\n".join(f"• `{p}`" for p in combined)
    if errors:
        msg += "\n\n_(übersprungen: " + ", ".join(errors) + ")_"
    msg += f"\n\nAntworte mit 'ja' / 'bestätigt' / 'machs' (innerhalb {DELETE_CONFIRM_TIMEOUT//60} Min)."
    return msg


def confirm_delete() -> str:
    """Führt alle pending Löschungen aus."""
    pending = PENDING_DELETIONS.get(ALLOWED_USER_ID)
    if not pending or not pending[0]:
        return "Keine Löschung pending — gibts nichts zu bestätigen."
    paths, ts = pending
    age = time.time() - ts
    if age > DELETE_CONFIRM_TIMEOUT:
        PENDING_DELETIONS.pop(ALLOWED_USER_ID, None)
        return f"Bestätigung zu spät ({int(age)}s > {DELETE_CONFIRM_TIMEOUT}s). Bitte nochmal anfordern."

    archive_root = VAULT / "99_Archive"
    results = []
    for rel_path in paths:
        try:
            src = safe_path(rel_path)
            if not src.exists():
                results.append(f"✗ {rel_path} verschwunden")
                continue
            dst = archive_root / src.relative_to(VAULT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst = dst.with_name(f"{dst.stem}_{ts_str}{dst.suffix}")
            os.rename(src, dst)
            results.append(f"✓ {rel_path}")
            log.info(f"Archived: {rel_path} → {dst.relative_to(VAULT)}")
        except Exception as e:
            log.exception(f"delete {rel_path} failed")
            results.append(f"✗ {rel_path} ({e})")

    PENDING_DELETIONS.pop(ALLOWED_USER_ID, None)
    return f"Verschoben nach 99_Archive/ ({len(paths)} Datei(en)):\n" + "\n".join(results)


def cancel_delete() -> str:
    """Bricht alle pending Löschungen ab."""
    pending = PENDING_DELETIONS.pop(ALLOWED_USER_ID, None)
    if not pending or not pending[0]:
        return "Keine pending Löschung."
    return f"Löschanfrage für {len(pending[0])} Datei(en) abgebrochen."


# ─── Reminders (persistent + JobQueue) ──────────────────────────────────────
# Reminders überleben Bot-Restart: JSON in 06_Meta/reminders.json.
# Bei Startup werden alle aktiven Reminders neu in die JobQueue eingehängt.
REMINDERS_FILE = VAULT / "06_Meta" / "reminders.json"
ACTIVE_REMINDER_JOBS: dict = {}  # id → telegram.ext.Job
BOT_APP = None  # wird in main() gesetzt — brauchen Zugriff auf job_queue von Tools aus


def _load_reminders() -> list:
    if not REMINDERS_FILE.exists():
        return []
    try:
        return json.loads(REMINDERS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"reminders.json kaputt: {e}")
        return []


def _save_reminders(reminders: list) -> None:
    REMINDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    REMINDERS_FILE.write_text(
        json.dumps(reminders, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


VALID_RECURRENCE = {None, "", "daily", "weekly", "weekdays"}


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
    if rec not in VALID_RECURRENCE:
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
    reminders = _load_reminders()
    reminders = [r for r in reminders if r["id"] != rid]
    _save_reminders(reminders)


async def reminder_callback(ctx: ContextTypes.DEFAULT_TYPE):
    """Wird von JobQueue ausgelöst wenn ein Reminder fällig wird."""
    data = ctx.job.data
    rid = data["id"]
    message = data["message"]
    try:
        await ctx.bot.send_message(
            chat_id=ALLOWED_USER_ID,
            text=f"⏰ <b>Erinnerung</b>\n\n{_esc_html(message)}",
            parse_mode=constants.ParseMode.HTML,
        )
        log.info(f"Reminder fired: {rid}")
    except Exception as e:
        log.exception(f"Reminder-Send fehlgeschlagen für {rid}")

    # Einmalige Reminder aus JSON + ACTIVE_JOBS entfernen
    reminders = _load_reminders()
    reminder = next((r for r in reminders if r["id"] == rid), None)
    if reminder and not reminder.get("recurrence"):
        _remove_reminder_from_json(rid)
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
        rec = f" 🔁 {r['recurrence']}" if r.get("recurrence") else ""
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


def backup_vault() -> str:
    """Manuelles Backup: Vault in privates GitHub-Repo pushen.

    Voraussetzung: GITHUB_BACKUP_REPO (z.B. 'julasim/KI_WIKI_Vault_Backup') und
    GITHUB_BACKUP_TOKEN (PAT mit Repo-Schreibrechten) in .env.

    Workflow: clone-on-first-use → rsync vault-content → commit → push
    """
    repo = os.environ.get("GITHUB_BACKUP_REPO", "").strip()
    token = os.environ.get("GITHUB_BACKUP_TOKEN", "").strip()
    if not repo or not token:
        return ("Backup nicht konfiguriert. Setze GITHUB_BACKUP_REPO und "
                "GITHUB_BACKUP_TOKEN in /opt/bot/.env, dann docker compose restart.")

    backup_dir = Path("/vault-backup")
    # Token URL-encodieren falls Sonderzeichen drin (defensive — GitHub-PATs sind
    # zwar normalerweise ASCII-alphanumerisch, aber sicher ist sicher)
    from urllib.parse import quote
    safe_token = quote(token, safe="")
    repo_url = f"https://x-access-token:{safe_token}@github.com/{repo}.git"

    def _run(cmd, cwd=None, env_extra=None):
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)

    try:
        # Clone-on-first-use oder Pull
        if not (backup_dir / ".git").exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Cloning backup repo: {repo}")
            r = _run(["git", "clone", repo_url, str(backup_dir)])
            if r.returncode != 0:
                # Sanitize: Token aus Fehler entfernen falls drin
                err = r.stderr.replace(token, "***")
                return f"Clone-Fehler: {err}"
        else:
            r = _run(["git", "pull", "--rebase"], cwd=backup_dir)
            if r.returncode != 0 and "no upstream" not in r.stderr.lower():
                err = r.stderr.replace(token, "***")
                # Pull-Fehler nicht fatal — wir pushen trotzdem
                log.warning(f"Pull warning: {err}")

        # Git-Identity setzen (idempotent, lokal zum Repo)
        _run(["git", "config", "user.email", "bot@ki-wiki.local"], cwd=backup_dir)
        _run(["git", "config", "user.name", "KI Wiki Bot"], cwd=backup_dir)

        # Vault-Content rüber — Pre-Flight: Source muss existieren + Files haben,
        # sonst würde rsync --delete uns das Backup-Repo leeren
        if not VAULT.exists() or not VAULT.is_dir():
            return f"Backup-Abbruch: Vault-Source {VAULT} nicht erreichbar (Mount kaputt?)"
        if not any(VAULT.iterdir()):
            return f"Backup-Abbruch: Vault-Source {VAULT} ist leer — nichts zu sichern, schütze gegen Daten-Verlust im Backup."

        target = backup_dir / "vault"
        target.mkdir(exist_ok=True)
        r = _run([
            "rsync", "-a", "--delete",
            "--exclude=.obsidian/workspace*",
            "--exclude=.obsidian/cache",
            "--exclude=*.tmp",
            "--exclude=__pycache__",
            f"{VAULT}/", f"{target}/",
        ])
        if r.returncode != 0:
            return f"Rsync-Fehler: {r.stderr}"

        # Stage + Commit (nur wenn Änderungen)
        _run(["git", "add", "-A"], cwd=backup_dir)
        diff = _run(["git", "diff", "--cached", "--quiet"], cwd=backup_dir)
        if diff.returncode == 0:
            return "✓ Keine Änderungen seit letztem Backup."

        # Commit
        commit_msg = f"Backup {datetime.now().isoformat(timespec='seconds')}"
        r = _run(["git", "commit", "-m", commit_msg], cwd=backup_dir)
        if r.returncode != 0:
            return f"Commit-Fehler: {r.stderr}"

        # Push
        r = _run(["git", "push"], cwd=backup_dir)
        if r.returncode != 0:
            err = r.stderr.replace(token, "***")
            return f"Push-Fehler: {err}"

        # Stats
        hash_r = _run(["git", "rev-parse", "--short", "HEAD"], cwd=backup_dir)
        commit_hash = hash_r.stdout.strip()
        # Datei-Anzahl im Backup
        count_r = _run(["bash", "-c", f"find '{target}' -type f -name '*.md' | wc -l"])
        file_count = count_r.stdout.strip() or "?"

        return (f"✓ Backup gepusht\n"
                f"Repo: {repo}@{commit_hash}\n"
                f"Files: {file_count} .md\n"
                f"Zeit: {datetime.now().strftime('%H:%M:%S')}")

    except subprocess.TimeoutExpired:
        return "Backup-Fehler: Timeout (>60s). Repo zu groß oder Netz langsam?"
    except Exception as e:
        log.exception("backup_vault failed")
        return f"Backup-Fehler: {e}"


def list_existing_tags(top_n: int = 30) -> str:
    """Liste der existierenden Tags im Vault, sortiert nach Häufigkeit.

    Hilft dem LLM beim Auto-Tagging konsistent zu bleiben statt jeden
    Eintrag neu zu erfinden ('arbeit' vs 'work' vs 'job' Drift vermeiden).
    """
    from collections import Counter
    counter: Counter = Counter()
    for p in VAULT.rglob("*.md"):
        if any(part in (".obsidian", "99_Archive", ".trash") for part in p.parts):
            continue
        try:
            post = frontmatter.load(p)
            tags = post.get("tags") or []
            if isinstance(tags, list):
                for t in tags:
                    if isinstance(t, str) and t.strip():
                        counter[t.strip().lower()] += 1
        except Exception:
            continue
    if not counter:
        return "Keine Tags im Vault — du legst die Konvention fest."
    top = counter.most_common(top_n)
    lines = [f"Top {len(top)} bestehende Tags ({sum(counter.values())} Verwendungen gesamt):"]
    for tag, count in top:
        lines.append(f"• `{tag}` ({count}×)")
    return "\n".join(lines)


def create_project(name: str, description: str = "", area: Optional[str] = None) -> str:
    """Legt einen neuen Projekt-Ordner unter 10_Life/projects/<slug>/ an.

    Erzeugt Ordner + README.md mit Dataview-Queries die alle Tasks/Notes mit
    'project: <slug>' im Frontmatter sammeln. So können Files weiterhin in den
    Standard-Ordnern (10_Life/tasks/, /notes/) liegen, sind aber im Projekt-
    Container automatisch gelistet.
    """
    if not name or not name.strip():
        return "Fehler: Projekt-Name darf nicht leer sein."
    slug = slugify(name)
    if not slug or slug == "untitled":
        return f"Fehler: Slug aus '{name}' nicht ableitbar — bitte aussagekräftigeren Namen wählen."

    projects_root = LIFE / "projects"
    projects_root.mkdir(parents=True, exist_ok=True)
    proj_dir = projects_root / slug

    if proj_dir.exists():
        return f"Projekt existiert bereits: [[project-{slug}]] (10_Life/projects/{slug}/)"

    proj_dir.mkdir(parents=True, exist_ok=False)
    readme_path = proj_dir / "README.md"
    today = today_iso()

    desc_block = description.strip() if description else f"Projekt-Container für **{name}**. Tasks/Notes mit `project: {slug}` im Frontmatter werden hier automatisch gelistet."

    body = f"""# {name}

{desc_block}

## Status
- **Status**: active
- **Gestartet**: {today}{chr(10) + f"- **Area**: [[{area}]]" if area else ""}

## Offene Tasks
```dataview
TABLE WITHOUT ID file.link AS Task, priority AS Prio, due AS Fällig
FROM "10_Life/tasks"
WHERE project = "{slug}" AND status != "done" AND status != "cancelled"
SORT priority DESC, due ASC
```

## Notizen
```dataview
LIST
FROM "10_Life/notes"
WHERE project = "{slug}"
SORT file.ctime DESC
```

## Meetings
```dataview
TABLE WITHOUT ID file.link AS Meeting, date AS Datum
FROM "10_Life/meetings"
WHERE project = "{slug}"
SORT date DESC
```

## Log
- {today}: Projekt angelegt
"""

    post_data = {
        "id": f"project-{slug}",
        "title": name,
        "type": "project",
        "started": today,
        "status": "active",
        "tags": [],
    }
    if area:
        post_data["area"] = area
    post = frontmatter.Post(body, **post_data)
    atomic_write(readme_path, frontmatter.dumps(post) + "\n")

    return (f"✓ Projekt angelegt: [[project-{slug}]]\n"
            f"Ordner: `10_Life/projects/{slug}/`\n"
            f"Tipp: Tasks/Notes mit `project: {slug}` werden automatisch im Projekt-README gelistet.")


# Noise-Verzeichnisse die in list_files-Output rausgefiltert werden
# (System-Internals, Templates, Meta-Reports, Tooling — nicht User-Content)
LIST_FILES_NOISE_DIRS = {
    ".obsidian", ".trash", "99_Archive",
    "08_Templates", "06_Meta", "07_Tools",
}
LIST_FILES_NOISE_FILES = {
    "README.md", "_index.md",
    "CLAUDE.md", "COMMANDS.md", "MOC.md",
    "PIPELINES.md", "SCHEMA.md",
}


def list_files(rel_dir: str = "", include_system: bool = False) -> str:
    """Liste alle .md-Files in einem Vault-Unterordner.

    Standardmäßig werden System-Verzeichnisse (Templates, Meta, Tools, Trash, Archive)
    und System-Docs (CLAUDE.md etc.) ausgefiltert — User sieht nur eigenen Content.
    Mit include_system=True wird alles gezeigt (für Debug).
    """
    try:
        base = safe_path(rel_dir) if rel_dir else VAULT
        if not base.exists() or not base.is_dir():
            return f"Verzeichnis nicht gefunden: {rel_dir}"

        def is_visible(p: Path) -> bool:
            if include_system:
                return True
            if p.name in LIST_FILES_NOISE_FILES:
                return False
            return not any(part in LIST_FILES_NOISE_DIRS for part in p.parts)

        files = sorted(p for p in base.rglob("*.md") if is_visible(p))
        if not files:
            return f"Keine User-Files in {rel_dir or 'Vault-Root'} (System-Files via include_system=true sichtbar)."

        rels = [str(f.relative_to(VAULT)).replace("\\", "/") for f in files]

        # Bei vielen Files: nach Top-Level-Ordner gruppieren für lesbare Ausgabe
        if len(rels) > 12:
            from collections import defaultdict
            grouped = defaultdict(list)
            for r in rels:
                top = r.split("/", 1)[0]
                grouped[top].append(r)
            lines = [f"{len(rels)} Files in {rel_dir or 'Vault-Root'}, gruppiert:"]
            for top in sorted(grouped):
                lines.append(f"\n**{top}/** ({len(grouped[top])})")
                for r in grouped[top][:8]:
                    lines.append(f"• `{r}`")
                if len(grouped[top]) > 8:
                    lines.append(f"  _… {len(grouped[top])-8} weitere_")
            return "\n".join(lines)

        return f"{len(rels)} Files in {rel_dir or 'Vault-Root'}:\n" + "\n".join(f"• `{r}`" for r in rels)
    except Exception as e:
        return f"List-Fehler: {e}"


# ─── Conversation Memory ─────────────────────────────────────────────────────
# Letzte N Turns pro User. Lebt im RAM solange Container läuft.
# Reset nach 30 Min Inaktivität.
CONVERSATION_HISTORY: dict[int, list] = {}
CONVERSATION_TIMESTAMPS: dict[int, float] = {}
HISTORY_MAX_MESSAGES = 24       # ca. 12 User+Assistant-Turns
HISTORY_TIMEOUT = 30 * 60       # 30 Minuten


def get_history(user_id: int) -> list:
    """History für User holen, expirieren wenn zu alt."""
    last = CONVERSATION_TIMESTAMPS.get(user_id, 0.0)
    if time.time() - last > HISTORY_TIMEOUT:
        CONVERSATION_HISTORY[user_id] = []
    return CONVERSATION_HISTORY.get(user_id, [])


def update_history(user_id: int, new_messages: list) -> None:
    """History anhängen + auf Max-Länge trimmen."""
    history = get_history(user_id)
    history.extend(new_messages)
    if len(history) > HISTORY_MAX_MESSAGES:
        history = history[-HISTORY_MAX_MESSAGES:]
    CONVERSATION_HISTORY[user_id] = history
    CONVERSATION_TIMESTAMPS[user_id] = time.time()


def reset_history(user_id: int) -> None:
    """History komplett resetten (z.B. via /reset Command)."""
    CONVERSATION_HISTORY.pop(user_id, None)
    CONVERSATION_TIMESTAMPS.pop(user_id, None)


# ============================================================================
# Tool definitions (OpenAI function-calling format)
# ============================================================================

TOOLS = [
    {"type": "function", "function": {
        "name": "append_to_daily",
        "description": "Hängt Text an die heutige Daily-Note unter passender Sektion an. Wähle Sektion: 'Heute' für Tasks/Termine, 'Notizen & Gedanken' für Gedanken/Notizen (default), 'Offen / Einsortieren' für offene Sachen/Links, 'Abends' für Reflexion/Bewertung des Tages.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string", "enum": list(VALID_SECTIONS)},
                "text": {"type": "string", "description": "Markdown-Text zum Anhängen."},
            },
            "required": ["section", "text"],
        },
    }},
    {"type": "function", "function": {
        "name": "create_task",
        "description": (
            "Legt neuen Task in 10_Life/tasks/ an, verlinkt unter 'Heute' in heutiger Daily. "
            "WICHTIG: Optionale Parameter NUR setzen wenn explizit aus User-Input ableitbar — "
            "NIE raten/defaulten. Z.B. due nur wenn User wirklich ein Datum nennt; "
            "project nur wenn User Projekt nennt oder es klar aus Conversation-Memory hervorgeht. "
            "tags: 2-5 thematische Schlagworte aus dem Inhalt extrahieren (kebab-case, Deutsch)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Vollständiger Task-Titel (deutsch ok)"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"], "description": "Default medium falls User keine Prio nennt"},
                "due": {"type": "string", "description": "ISO YYYY-MM-DD. NUR setzen wenn User explizit ein Datum nennt!"},
                "area": {"type": "string", "description": "Area-ID, NUR wenn User Area nennt"},
                "project": {"type": "string", "description": "Projekt-Slug (z.B. 'sanierung-kiosk'), NUR wenn aus User-Input oder Memory klar"},
                "context": {"type": "string", "enum": ["home", "work", "errand", "phone", "computer"]},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-5 thematische Tags (kebab-case, Deutsch). Topical, nicht 'task'/'todo'. Bei Bedarf vorher list_existing_tags für konsistente Vokabular.",
                },
            },
            "required": ["title"],
        },
    }},
    {"type": "function", "function": {
        "name": "mark_task_done",
        "description": "Setzt einen Task auf done. Slug ohne 't-'-Präfix möglich.",
        "parameters": {
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"],
        },
    }},
    {"type": "function", "function": {
        "name": "create_meeting",
        "description": "Legt ein Meeting-Protokoll in 10_Life/meetings/ an. tags: thematische Schlagworte extrahieren (z.B. ['blitztext', 'kunde', 'kickoff']).",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
                "meeting_date": {"type": "string", "description": "ISO-Datum YYYY-MM-DD, default heute"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "2-5 thematische Tags, kebab-case Deutsch"},
            },
            "required": ["title"],
        },
    }},
    {"type": "function", "function": {
        "name": "create_note",
        "description": "Legt eine freie Notiz in 10_Life/notes/ an. Für längere strukturierte Inhalte (>3 Sätze, eigenes Thema), die weder Task noch Tagesreflexion sind. tags: 2-5 thematische Schlagworte aus dem Inhalt.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "2-5 thematische Tags, kebab-case Deutsch (z.B. ['gesundheit', 'ernaehrung'])"},
            },
            "required": ["title", "body"],
        },
    }},
    {"type": "function", "function": {
        "name": "search_vault",
        "description": "Volltextsuche im Vault. Nutze für 'Was weiß ich über X', Wiki-Lookups, Task-Suche.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": (
            "Liest eine Vault-Datei (Pfad relativ zu Vault-Root). "
            "strip_frontmatter=true (default): YAML-Header wird entfernt — nimm das wenn du "
            "dem User den Inhalt zeigst. strip_frontmatter=false: kompletter Inhalt inkl. "
            "Metadaten — nimm das wenn du Frontmatter-Felder brauchst (status, due, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rel_path": {"type": "string"},
                "strip_frontmatter": {"type": "boolean", "default": True},
            },
            "required": ["rel_path"],
        },
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Find/Replace in einer Vault-Datei. Bei regex=true wird find als Regex interpretiert.",
        "parameters": {
            "type": "object",
            "properties": {
                "rel_path": {"type": "string"},
                "find": {"type": "string"},
                "replace": {"type": "string"},
                "regex": {"type": "boolean", "default": False},
            },
            "required": ["rel_path", "find", "replace"],
        },
    }},
    {"type": "function", "function": {
        "name": "clip_url",
        "description": "Fetcht eine URL und speichert den Hauptinhalt als Markdown-Artikel in 01_Raw/articles/.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    }},
    {"type": "function", "function": {
        "name": "request_delete",
        "description": (
            "Schritt 1 von 2 für sicheres Löschen: meldet Löschanfrage an User. "
            "Akzeptiert eine ODER mehrere Pfade (Batch-Delete). Bei 'lösche alle X' "
            "erst list_files aufrufen, dann ALLE Pfade in einem request_delete-Call "
            "übergeben. NIEMALS direkt confirm_delete ohne vorherige User-Bestätigung. "
            "Pfade relativ zu Vault-Root."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rel_paths": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Einzelner Pfad oder Liste von Pfaden",
                },
            },
            "required": ["rel_paths"],
        },
    }},
    {"type": "function", "function": {
        "name": "confirm_delete",
        "description": (
            "Schritt 2 von 2: führt ALLE pending Löschungen aus (Files werden ins "
            "99_Archive/ verschoben, NICHT hart gelöscht — reversibel). "
            "Nur aufrufen NACHDEM User explizit bestätigt hat ('ja', 'bestätigt', 'machs')."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "cancel_delete",
        "description": "Bricht alle pending Löschungen ab. Bei 'nein', 'abbrechen', 'doch nicht'.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "list_files",
        "description": (
            "Listet User-Content-Files (.md) im Vault. System-Krempel (Templates, Meta, "
            "Tools, Trash, Archive, CLAUDE.md etc.) wird automatisch rausgefiltert. "
            "Bei >12 Files wird nach Top-Level-Ordner gruppiert."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rel_dir": {
                    "type": "string",
                    "description": "Verzeichnis relativ zu Vault-Root, z.B. '10_Life/tasks'. Leer = ganzer Vault.",
                },
                "include_system": {
                    "type": "boolean",
                    "description": "True = auch Templates/Meta/Tools/Archive zeigen (Debug). Default false.",
                    "default": False,
                },
            },
            "required": [],
        },
    }},
    {"type": "function", "function": {
        "name": "list_existing_tags",
        "description": (
            "Liste der bereits im Vault verwendeten Tags, sortiert nach Häufigkeit. "
            "Optional vor create_task/note/meeting aufrufen wenn unklar welche Tags konsistent sind. "
            "Hilft Drift zu vermeiden ('arbeit' vs 'work' vs 'job')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "default": 30, "description": "Anzahl Top-Tags zurückgeben"},
            },
            "required": [],
        },
    }},
    {"type": "function", "function": {
        "name": "create_project",
        "description": (
            "Legt einen neuen Projekt-Ordner unter 10_Life/projects/<slug>/ an. "
            "Erzeugt Folder + README.md mit Dataview-Queries für zugehörige "
            "Tasks/Notes/Meetings (gefiltert via project=<slug> Frontmatter). "
            "Nutze wenn User 'lege ein Projekt an', 'neues Projekt: ...' o.ä. sagt. "
            "WICHTIG: dieses Tool legt den Ordner UND die README-Datei direkt an — "
            "nicht erst fragen, dann ankündigen, dann nochmal fragen. Direkt machen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Projekt-Name, wird zu Slug umgewandelt"},
                "description": {"type": "string", "description": "Optional: kurze Beschreibung fürs README"},
                "area": {"type": "string", "description": "Optional: zugehörige Area-ID"},
            },
            "required": ["name"],
        },
    }},
    {"type": "function", "function": {
        "name": "create_reminder",
        "description": (
            "Setzt eine Erinnerung — Bot schickt zur angegebenen Zeit eine Telegram-Nachricht "
            "mit dem message-Text. Berechne when_iso ALS ABSOLUTES DATUM + ZEIT in Lokalzeit "
            "(Europe/Vienna). 'morgen 15:00' → 2026-04-27T15:00:00. 'in 30 Min' → "
            "aktuelle Zeit + 30 Min. Heutiges Datum kennst du aus dem System-Prompt. "
            "Wiederholungen via recurrence: daily, weekdays (Mo-Fr), weekly (jede Woche "
            "selber Wochentag). Leer/null = einmalig."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "when_iso": {"type": "string", "description": "ISO-Datetime YYYY-MM-DDTHH:MM:SS, lokal Europe/Vienna"},
                "message": {"type": "string", "description": "Text der gesendet wird"},
                "recurrence": {"type": "string", "enum": ["daily", "weekdays", "weekly"]},
            },
            "required": ["when_iso", "message"],
        },
    }},
    {"type": "function", "function": {
        "name": "list_reminders",
        "description": "Listet alle aktiven Erinnerungen (für 'was steht an?', 'welche Reminder hab ich').",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "cancel_reminder",
        "description": "Bricht eine Erinnerung ab. Erst list_reminders aufrufen wenn der User keine ID nennt.",
        "parameters": {
            "type": "object",
            "properties": {"reminder_id": {"type": "string", "description": "Reminder-ID, z.B. 'rem-20260426-153000-123'"}},
            "required": ["reminder_id"],
        },
    }},
    {"type": "function", "function": {
        "name": "backup_vault",
        "description": (
            "Manuelles Backup: pusht das gesamte Vault in das konfigurierte private "
            "GitHub-Repo. Nutze wenn der User 'mach backup', 'sicher das vault', "
            "'git push' o.ä. sagt. Idempotent — wenn nichts geändert, kommt entsprechende "
            "Meldung. Hat keine Parameter."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]

TOOL_HANDLERS = {
    "append_to_daily": append_to_daily,
    "create_task": create_task,
    "mark_task_done": mark_task_done,
    "create_meeting": create_meeting,
    "create_note": create_note,
    "search_vault": search_vault,
    "read_file": read_file,
    "edit_file": edit_file,
    "clip_url": clip_url,
    "request_delete": request_delete,
    "confirm_delete": confirm_delete,
    "cancel_delete": cancel_delete,
    "list_files": list_files,
    "list_existing_tags": list_existing_tags,
    "create_project": create_project,
    "create_reminder": create_reminder,
    "list_reminders": list_reminders,
    "cancel_reminder": cancel_reminder,
    "backup_vault": backup_vault,
}

# ============================================================================
# System prompt (cached)
# ============================================================================

SYSTEM_PROMPT = """Du bist Julius' Vault-Assistent über Telegram. Deutsch. Heute ist {today}, aktuelle Uhrzeit {now} ({tz}).

VAULT
- 10_Life/daily/YYYY-MM-DD.md   Tageseinträge (Sektionen: Heute · Notizen & Gedanken · Offen / Einsortieren · Abends)
- 10_Life/tasks/<slug>.md        Tasks (id: t-<slug>)
- 10_Life/notes/                 Freie Notizen
- 10_Life/meetings/              Meeting-Protokolle
- 10_Life/areas/                 Lebensbereiche
- 02_Wiki/                       Kompiliertes Wissen
- 01_Raw/                        Externe Quellen (Articles, Uploads)

# VERHALTEN

Du bist Gesprächspartner, kein Auto-Logger. **Default: nur antworten, nichts speichern.**
Tool nur aufrufen wenn Julius EXPLIZIT Speicher-Intent zeigt:

| Trigger | Tool |
|---|---|
| "speicher / merk dir / notiere / schreib auf / ins tagebuch" | `append_to_daily` oder `create_note` |
| "task: …" / "todo: …" / "morgen X machen" / Imperativ+Frist | `create_task` |
| "meeting: …" / "war im Termin mit …" | `create_meeting` |
| "X erledigt / fertig / done" | `mark_task_done` (ggf. erst `search_vault`) |
| URL allein, sonst nichts | erst fragen, dann `clip_url` |
| "lösche X / weg mit X" | `request_delete` (NIE direkt confirm!) |
| "lösche alle X" / "leere Y" | erst `list_files` für Verzeichnis, dann `request_delete` mit ALLEN Pfaden als Liste |
| "ja / bestätigt / machs" nach request_delete | `confirm_delete` |
| "nein / abbrechen" nach request_delete | `cancel_delete` |
| "lege ein Projekt an" / "neues Projekt: ..." | `create_project` (legt Ordner + README direkt an, NICHT erst nachfragen wo) |
| "erinner mich um X an Y" / "wecker für 14 Uhr" / "in 30 Min ping" | `create_reminder` (when_iso = absolute Lokalzeit berechnen!) |
| "jeden Montag 8 Uhr X" / "täglich um 7 X" | `create_reminder` mit `recurrence` (daily/weekdays/weekly) |
| "welche Reminder hab ich?" | `list_reminders` |
| "cancel den Reminder X" / "lösche die Erinnerung Y" | `cancel_reminder` (notfalls vorher list_reminders) |
| "mach backup" / "sicher das vault" / "git push" | `backup_vault` |

Begrüßung, Smalltalk, Fragen, Statements ohne Speicher-Verb → antworten, **nichts speichern**.

**Nachfragen vs. Direkt-Machen** (klare Regel):
- Mini-Input ohne Kontext ("Diese", "ja", "?", einzelne Wörter) → **nachfragen**, nicht raten.
- Vollständiger Auftrag mit klarem Speicher-Intent ("speicher: ...", "task: morgen X") → **direkt machen**, kein "wo soll ich das anlegen?"-Loop. Default-Pfade sind im Schema definiert.

# LESEN / FRAGEN BEANTWORTEN
- "Was weiß ich über X?" → `search_vault`, antworten mit echten `[[id]]` aus Treffern
- "Was steht heute an?" → `read_file` Daily + offene Tasks aus `list_files`/Frontmatter
- "Was ist im Vault?" → `list_files()` direkt aufrufen, NICHT 3× zurückfragen "welcher Bereich"
- File-Inhalt zeigen → `read_file` (Default strip_frontmatter=true), Original-Markdown direkt
  - KEINE Meta-Tabelle "Sektion | Inhalt | (leer)"
  - Leere Sektionen weglassen, kein "(leer)" reinschreiben
  - Kein Navigation-Footer (→ Life-Index) zeigen
- Lange Files (>2000 Zeichen) zusammenfassen statt roh dumpen

# AUSGABE

- Deutsch, direkt, kein Höflichkeits-Geschwurbel. Sparsame Emojis (✓ ✗ ⚠️).
- Bestätigung einer Aktion: 1 Satz mit `[[wikilink]]`.
- Frage: so lang wie nötig, gut strukturiert.
- Format frei wählen — Tabellen für Vergleiche, Bullets für Listen, Code-Blöcke für Code/Pfade,
  **bold** für Schlüsselbegriffe, *italic* für Betonung, ## Headings nur bei langen Antworten.
- **Wikilinks**: `[[id]]` mit `id` aus dem Frontmatter (z.B. `[[daily-2026-04-25]]`, `[[t-foo]]`).
  - NIE Filepath wie `[[10_Life/daily/2026-04-25.md]]`
  - NIE Beispiel-Platzhalter wie `[[t-example]]`, `[[wiki-example]]`, `[[id]]`, `[[file-name]]`
  - NUR IDs verwenden die du tatsächlich aus search_vault/read_file/list_files kennst.
    Wenn du keine IDs zur Hand hast → KEINE Wikilinks setzen, lieber Klartext.
- **NIE** HTML-Tags (`<br>`, `<p>`, `<span>`). NIE Frontmatter ausgeben.

# DATEN
- Datum: ISO `YYYY-MM-DD`. "morgen" = +1 Tag, "übermorgen" = +2, "nächsten Montag" → berechnen.
- **Tasks ohne explizit genanntes Datum → `due` LEER lassen, NICHT 'heute' raten!**
  "Schränke aufbauen" hat KEIN due. "Schränke aufbauen morgen" hat due=morgen.
- **Priorität aus Sprache erkennen** (statt default medium):
  - "dringend", "urgent", "asap", "sofort", "eilt" → `priority=urgent`
  - "wichtig", "hochpriorisiert", "high prio" → `priority=high`
  - "irgendwann", "nice to have", "low prio" → `priority=low`
  - Sonst → `priority=medium`
  Wenn ein Prio-Wort am Titel-Anfang steht, ENTFERNE es aus dem Titel
  (z.B. "dringend Server-Backup einrichten" → title="Server-Backup einrichten",
  priority="urgent")

# AUTO-TAGGING
Beim Anlegen von Tasks / Notizen / Meetings: extrahiere **2-5 thematische Tags**
aus dem Inhalt und übergib sie via `tags`-Parameter.
- Format: kebab-case, Deutsch (z.B. `gesundheit`, `auto-projekt`, `kunde-mueller`)
- TOPISCH, nicht strukturell — also NICHT `task`, `note`, `daily` (die sind im `type`-Feld).
- Bei wenig Kontext (z.B. "Mail an Anna schreiben"): wenige präzise Tags > viele vage. Lieber `[]` wenn Inhalt zu generisch.
- Bei häufiger Nutzung optional vorher `list_existing_tags` rufen — vermeidet Drift
  (`arbeit` vs `work` vs `job`). Default: ohne Lookup, LLM-Vokabular ist ok.

# KONTEXT-NUTZUNG (Conversation-Memory)
- Wenn du im Verlauf der letzten paar Turns ein Projekt erstellt hast (z.B. `sanierung-kiosk`)
  und der User danach Tasks anlegt die offensichtlich dazu gehören (z.B. "Kalkulation
  für Schiemer-Angebot" — Schiemer war im Projekt-Kontext erwähnt), dann setze
  AUTOMATISCH `project=sanierung-kiosk` im Task-Frontmatter. Frag nicht nach.
"""

# ============================================================================
# LLM tool-use loop
# ============================================================================

async def llm_loop(user_text: str, user_id: int) -> str:
    """Run tool-use loop until final answer or limit reached.

    Mit Conversation-Memory: letzte ~12 Turns werden als Context übergeben.
    """
    now_local = datetime.now(TIMEZONE)
    sys_text = (SYSTEM_PROMPT
                .replace("{today}", today_iso())
                .replace("{now}", now_local.strftime("%H:%M"))
                .replace("{tz}", TIMEZONE.key if hasattr(TIMEZONE, "key") else str(TIMEZONE)))

    # System-Prompt + History + neue User-Message
    history = get_history(user_id)
    new_user_msg = {"role": "user", "content": user_text}
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": sys_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ] + history + [new_user_msg]

    # Diese Messages werden am Ende zur History dazugefügt
    new_history_msgs = [new_user_msg]

    for _ in range(8):
        resp = await asyncio.to_thread(
            llm.chat.completions.create,
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048,
        )
        msg = resp.choices[0].message
        msg_dict = msg.model_dump(exclude_none=True)
        messages.append(msg_dict)
        new_history_msgs.append(msg_dict)

        if not msg.tool_calls:
            update_history(user_id, new_history_msgs)
            return msg.content or "(keine Antwort)"

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                handler = TOOL_HANDLERS.get(tc.function.name)
                if not handler:
                    result = f"Tool nicht bekannt: {tc.function.name}"
                else:
                    log.info(f"tool: {tc.function.name}({args})")
                    result = handler(**args)
            except Exception as e:
                log.exception(f"Tool {tc.function.name} failed")
                result = f"Tool-Fehler: {e}"
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            }
            messages.append(tool_msg)
            new_history_msgs.append(tool_msg)

    update_history(user_id, new_history_msgs)
    return "(Tool-Loop-Limit erreicht — bitte präziser fragen.)"


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


def _render_md_table(table_text: str) -> str:
    """Markdown-Tabelle → monospaced ASCII mit Unicode-Box-Zeichen."""
    lines = [l for l in table_text.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return table_text  # zu kurz für Tabelle, unverändert lassen

    def cells(line: str) -> list:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    rows = [cells(l) for l in lines[2:]]  # lines[1] ist die Separator-Linie
    n = len(header)
    rows = [r[:n] + [""] * max(0, n - len(r)) for r in rows]

    widths = [len(c) for c in header]
    for r in rows:
        for i in range(n):
            widths[i] = max(widths[i], len(r[i]))

    def fmt_row(cells_):
        return " │ ".join(c.ljust(widths[i]) for i, c in enumerate(cells_))

    sep = "─┼─".join("─" * w for w in widths)
    out = [fmt_row(header), sep]
    for r in rows:
        out.append(fmt_row(r))
    return "\n".join(out)


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

    # 1) TABELLEN — als monospaced Pre-Block stashen
    def _table_repl(m):
        rendered = _render_md_table(m.group(1))
        return add_stash(f"<pre><code>{_esc_html(rendered)}</code></pre>")
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


async def safe_reply(update: Update, text: str) -> None:
    """Split + send. HTML-Format mit Plain-Fallback bei Parse-Fehler."""
    if not text:
        await update.message.reply_text("(leer)")
        return

    # Erst Markdown→HTML, dann splitten
    html = md_to_telegram_html(text)

    # Splitten an Newlines wo möglich (Smart-Breaks)
    chunks = []
    remaining = html
    while remaining:
        if len(remaining) <= TG_MAX_MESSAGE:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n\n", 0, TG_MAX_MESSAGE)
        if cut < 1000:
            cut = remaining.rfind("\n", 0, TG_MAX_MESSAGE)
        if cut < 1000:
            cut = remaining.rfind(" ", 0, TG_MAX_MESSAGE)
        if cut < 1:
            cut = TG_MAX_MESSAGE
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()

    for i, chunk in enumerate(chunks, 1):
        prefix = f"({i}/{len(chunks)})\n" if len(chunks) > 1 else ""
        try:
            await update.message.reply_text(
                prefix + chunk,
                parse_mode=constants.ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except Exception as e:
            log.warning(f"HTML-Send fehlgeschlagen, Fallback Plain: {e}")
            await update.message.reply_text(prefix + _strip_html(chunk))


@require_auth
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    log.info(f"text: {text[:120]}")
    await update.message.chat.send_action(constants.ChatAction.TYPING)
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
        await update.message.reply_text(f"📝 {transcript}")
        reply = await llm_loop(transcript, update.effective_user.id)
    except Exception as e:
        log.exception("voice handler failed")
        reply = f"Voice-Fehler: {e}"
    await safe_reply(update, reply)


@require_auth
async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1] if update.message.photo else None
    if not photo:
        return
    user_caption = update.message.caption or ""
    log.info(f"photo: file_id={photo.file_id[:20]}..., caption={user_caption[:60]}")
    await update.message.chat.send_action(constants.ChatAction.TYPING)
    try:
        file = await photo.get_file()
        ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "jpg"
        if file.file_path and "." in file.file_path:
            ext = file.file_path.rsplit(".", 1)[-1].lower()
        filename = f"photo_{ts}.{ext}"
        save_path = ATTACHMENTS_DIR / filename
        await file.download_to_drive(str(save_path))
        # Vision-Caption
        with open(save_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        prompt_text = (
            "Beschreibe in 1-2 Sätzen knapp, was auf dem Bild ist."
            f" Kontext vom User: \"{user_caption}\"" if user_caption else
            "Beschreibe in 1-2 Sätzen knapp, was auf dem Bild ist."
        )
        vision_resp = await asyncio.to_thread(
            vision_llm.chat.completions.create,
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
                ],
            }],
            max_tokens=300,
        )
        vision_caption = (vision_resp.choices[0].message.content or "(keine Beschreibung)").strip()

        # OCR (Tesseract) — parallel zur Vision-Caption, macht Bild-Text suchbar
        ocr_text = await asyncio.to_thread(ocr_image, save_path)
        ocr_text = ocr_text.strip()
        # Sehr kurze OCR-Ergebnisse (<10 Zeichen) sind meist Rauschen
        ocr_block = ""
        if len(ocr_text) >= 10:
            # Auf 800 Zeichen begrenzen für Daily-Eintrag, voller Text bleibt im File-Namen referenzierbar
            ocr_truncated = ocr_text[:800] + ("..." if len(ocr_text) > 800 else "")
            ocr_block = f"\n  *OCR*:\n  > {ocr_truncated}"

        # Link in Daily — mit Vision-Caption + OCR (falls was erkannt)
        if user_caption:
            link_block = f"![[{filename}]] — {user_caption}\n  *Vision*: {vision_caption}{ocr_block}"
        else:
            link_block = f"![[{filename}]] — {vision_caption}{ocr_block}"
        try:
            append_to_daily("Notizen & Gedanken", link_block)
        except Exception as e:
            log.warning(f"Daily-Link für Photo fehlgeschlagen: {e}")

        # Reply
        reply = f"🖼 {filename}\n\n<b>Vision</b>: {_esc_html(vision_caption)}"
        if ocr_text:
            reply += f"\n\n<b>OCR</b> ({len(ocr_text)} Zeichen):\n<pre><code>{_esc_html(ocr_text[:1500])}</code></pre>"
        await update.message.reply_text(reply, parse_mode=constants.ParseMode.HTML)
    except Exception as e:
        log.exception("photo handler failed")
        await update.message.reply_text(f"Photo-Fehler: {e}")


@require_auth
async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Datei-Upload (.md, .txt, .pdf, oder beliebige Binärdatei) ins Vault speichern."""
    doc = update.message.document
    if not doc:
        return
    user_caption = update.message.caption or ""
    log.info(f"document: {doc.file_name}, size={doc.file_size}, caption={user_caption[:60]}")
    await update.message.chat.send_action(constants.ChatAction.TYPING)

    tmp_path = None
    try:
        # Download
        file = await doc.get_file()
        suffix = Path(doc.file_name or "upload").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

        filename = doc.file_name or f"upload-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ext = Path(filename).suffix.lower()

        # Ziel-Verzeichnis nach Typ
        if ext in (".md", ".markdown", ".txt"):
            dest_dir = VAULT / "01_Raw" / "uploads"
            kind = "Text/Markdown"
        elif ext in (".pdf",):
            dest_dir = VAULT / "01_Raw" / "papers"
            kind = "PDF"
        else:
            dest_dir = VAULT / "09_Attachments"
            kind = "Datei"

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Konflikt-handling: bei Duplikat Suffix
        dest = dest_dir / filename
        counter = 1
        while dest.exists():
            stem = Path(filename).stem
            dest = dest_dir / f"{stem}-{counter}{ext}"
            counter += 1

        shutil.move(tmp_path, dest)
        tmp_path = None  # nicht mehr cleanup-pflichtig
        rel = dest.relative_to(VAULT)

        # Bei Text-Files: Inhalt lesen für Echo
        body_preview = ""
        if ext in (".md", ".markdown", ".txt"):
            try:
                content = dest.read_text(encoding="utf-8", errors="replace")
                body_preview = content[:1500]
            except Exception:
                body_preview = "(Inhalt nicht lesbar)"

        # PDF-Sonderfall: Text extrahieren + .md-Wrapper anlegen
        wrapper_link = None
        pdf_extra = ""
        if ext == ".pdf":
            await update.message.chat.send_action(constants.ChatAction.TYPING)
            pdf_text, pdf_meta, total_pages = await asyncio.to_thread(extract_pdf_text, dest)
            pdf_title = pdf_meta.get("title") or Path(filename).stem
            pdf_author = pdf_meta.get("author") or "unknown"

            # .md-Sibling anlegen → wird volltext-suchbar via vault_search.py
            md_path = dest.with_suffix(".md")
            n = 1
            while md_path.exists():
                md_path = dest.with_name(f"{dest.stem}-{n}.md")
                n += 1

            md_id = slugify(f"paper-{pdf_title}")
            md_body = (
                f"# {pdf_title}\n\n"
                f"📎 [Original PDF: {dest.name}](./{dest.name})\n\n"
                f"**Autor**: {pdf_author} · **Seiten**: {total_pages}"
                + (f" · **Subject**: {pdf_meta['subject']}" if pdf_meta.get('subject') else "")
                + "\n\n---\n\n"
                + (pdf_text or "(Text-Extraktion ergab keinen Inhalt)")
            )
            md_post = frontmatter.Post(
                md_body,
                id=md_id,
                title=pdf_title,
                type="paper",
                source=str(dest.relative_to(VAULT)),
                author=pdf_author,
                captured=today_iso(),
                pages=total_pages,
                tags=[],
            )
            atomic_write(md_path, frontmatter.dumps(md_post) + "\n")

            wrapper_link = md_path.relative_to(VAULT)
            body_preview = pdf_text[:1500] if pdf_text else ""
            pdf_extra = f"\n📑 Wrapper: <code>{wrapper_link}</code> ({total_pages} S., id <code>{md_id}</code>)"

        # Eintrag in heutige Daily
        try:
            if wrapper_link:
                link_text = f"📄 PDF hochgeladen: <code>{rel}</code> → durchsuchbar als [[{md_id}]]"
            else:
                link_text = f"📄 Datei hochgeladen: <code>{rel}</code>"
            if user_caption:
                link_text += f" — {user_caption}"
            append_to_daily("Notizen & Gedanken", link_text)
        except Exception as e:
            log.warning(f"Daily-Link fuer Document fehlgeschlagen: {e}")

        # Antwort an User
        reply = f"📄 <b>{kind}</b> gespeichert: <code>{rel}</code>"
        if pdf_extra:
            reply += pdf_extra
        if user_caption:
            reply += f"\n<i>{user_caption}</i>"
        if body_preview:
            preview_html = _esc_html(body_preview)
            reply += f"\n\n<b>Inhalt (Vorschau):</b>\n<pre><code>{preview_html}</code></pre>"
        await update.message.reply_text(
            reply,
            parse_mode=constants.ParseMode.HTML,
        )

    except Exception as e:
        log.exception("document handler failed")
        await update.message.reply_text(f"Document-Fehler: {e}")
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except Exception: pass


def compute_briefing() -> str:
    """Generiere die morgendliche Zusammenfassung als HTML-String.

    Inhalt: Datum, überfällige Tasks, heute geplant (aus Daily 'Heute'),
    offene Tasks (sortiert nach Prio), gestern Abends-Reflexion.
    Bei fundamentalen Problemen (Vault nicht erreichbar) → klare Fehlermeldung.
    """
    # Vault-Reachability-Check
    if not VAULT.exists() or not VAULT.is_dir():
        return f"❌ <b>Briefing fehlgeschlagen</b>\nVault unter <code>{VAULT}</code> nicht erreichbar.\nMount oder Container-Volume prüfen."

    today = today_iso()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    parts = [f"☀️ <b>Guten Morgen — {today}</b>"]

    # ─── Today's Daily: was steht für heute geplant? ───
    today_path = DAILY_DIR / f"{today}.md"
    if today_path.exists():
        try:
            post = frontmatter.load(today_path)
            body = post.content or ""
            m = re.search(r"## Heute\s*\n(.*?)(?=\n## |\Z)", body, re.DOTALL)
            if m and m.group(1).strip() and m.group(1).strip() != "- [ ]":
                heute_text = m.group(1).strip()[:600]
                parts.append(f"\n📋 <b>Heute geplant</b>\n<pre>{_esc_html(heute_text)}</pre>")
        except Exception as e:
            log.warning(f"briefing: today daily parse failed: {e}")

    # ─── Tasks-Scan: überfällig + offen ───
    open_tasks = []
    overdue = []
    if TASKS_DIR.exists():
        for tp in TASKS_DIR.glob("*.md"):
            try:
                tpost = frontmatter.load(tp)
                status = str(tpost.get("status", "")).lower()
                if status in ("done", "cancelled"):
                    continue
                title = str(tpost.get("title", tp.stem))
                tid = str(tpost.get("id", f"t-{tp.stem}"))
                prio = str(tpost.get("priority", "medium"))
                due = str(tpost.get("due", "")).strip()
                entry = (tid, title, prio, due)
                if due and due < today:
                    overdue.append(entry)
                else:
                    open_tasks.append(entry)
            except Exception:
                continue

    if overdue:
        overdue.sort(key=lambda x: x[3])
        parts.append("\n⚠️ <b>Überfällig</b>")
        for tid, title, prio, due in overdue[:8]:
            parts.append(f"• {_esc_html(title)} <i>(seit {due})</i>")

    if open_tasks:
        prio_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        open_tasks.sort(key=lambda x: (prio_order.get(x[2], 9), x[3] or "9999"))
        parts.append("\n✏️ <b>Offene Tasks</b>")
        for tid, title, prio, due in open_tasks[:8]:
            due_str = f" <i>(bis {due})</i>" if due else ""
            prio_emoji = {"urgent": "🔴", "high": "🟠", "medium": "•", "low": "·"}.get(prio, "•")
            parts.append(f"{prio_emoji} {_esc_html(title)}{due_str}")
        if len(open_tasks) > 8:
            parts.append(f"<i>… und {len(open_tasks)-8} weitere</i>")

    # ─── Gestern: Abends-Reflexion ───
    yest_path = DAILY_DIR / f"{yesterday}.md"
    if yest_path.exists():
        try:
            ypost = frontmatter.load(yest_path)
            ybody = ypost.content or ""
            m = re.search(r"## Abends\s*\n(.*?)(?=\n## |\Z)", ybody, re.DOTALL)
            if m and m.group(1).strip():
                abends = m.group(1).strip()
                # Template-Standardtext rausfiltern
                template_lines = ("- Was lief gut?", "- Was nehme ich mit?")
                non_template = [l for l in abends.split("\n") if l.strip() and l.strip() not in template_lines]
                if non_template:
                    abends_clean = "\n".join(non_template)[:400]
                    parts.append(f"\n🌙 <b>Gestern Abends</b>\n<i>{_esc_html(abends_clean)}</i>")
        except Exception:
            pass

    # ─── Wenn nichts da: gentle morning ───
    if not overdue and not open_tasks and not today_path.exists():
        parts.append("\n<i>Heute steht noch nichts an. Schreib mir was du heute vorhast oder genieß den freien Kopf.</i>")

    return "\n".join(parts)


async def daily_briefing_job(ctx: ContextTypes.DEFAULT_TYPE):
    """JobQueue-Callback — wird täglich um BRIEFING_HOUR ausgeführt."""
    try:
        text = await asyncio.to_thread(compute_briefing)
        await ctx.bot.send_message(
            chat_id=ALLOWED_USER_ID,
            text=text,
            parse_mode=constants.ParseMode.HTML,
        )
        log.info(f"Daily briefing sent to {ALLOWED_USER_ID}")
    except Exception as e:
        log.exception(f"daily_briefing_job failed: {e}")
        # Versuche zumindest eine Fehler-Notification zu schicken
        try:
            await ctx.bot.send_message(
                chat_id=ALLOWED_USER_ID,
                text=f"⚠️ Daily-Briefing-Fehler: {type(e).__name__}\n<code>{_esc_html(str(e))[:300]}</code>",
                parse_mode=constants.ParseMode.HTML,
            )
        except Exception:
            pass


@require_auth
async def handle_briefing(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/briefing — manueller Trigger für das Morgens-Briefing."""
    text = await asyncio.to_thread(compute_briefing)
    await update.message.reply_text(text, parse_mode=constants.ParseMode.HTML)


@require_auth
async def handle_reminders(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/reminders — alle aktiven Erinnerungen listen."""
    text = await asyncio.to_thread(list_reminders)
    await safe_reply(update, text)


@require_auth
async def handle_backup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/backup — manueller Push des Vaults ins konfigurierte GitHub-Repo."""
    await update.message.chat.send_action(constants.ChatAction.TYPING)
    await update.message.reply_text("⏳ Backup läuft...")
    result = await asyncio.to_thread(backup_vault)
    await safe_reply(update, result)


@require_auth
async def handle_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Setzt Conversation-Memory + pending Deletes zurück."""
    # Konsistent ALLOWED_USER_ID nutzen — ist single-user-bot
    reset_history(ALLOWED_USER_ID)
    PENDING_DELETIONS.pop(ALLOWED_USER_ID, None)
    await update.message.reply_text("🔄 Memory + pending Deletes geleert. Frischer Anfang.")


@require_auth
async def handle_today(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show today's daily note (without frontmatter)."""
    path = DAILY_DIR / f"{today_iso()}.md"
    if not path.exists():
        await update.message.reply_text("Heutige Daily noch leer.")
        return
    content = path.read_text(encoding="utf-8")
    body = re.sub(r"^---\n.*?\n---\n", "", content, count=1, flags=re.DOTALL)
    await safe_reply(update, body.strip() or "(leer)")


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
        "🎤 Sprachnachricht → transkribiert + sortiert\n"
        "🖼 Foto → in 09_Attachments + Vision-Caption + OCR (Tesseract de+en)\n"
        "📄 .md/.txt → in 01_Raw/uploads/\n"
        "📑 .pdf → in 01_Raw/papers/ + Volltext-Extraktion + .md-Wrapper für Suche\n"
        "📎 sonstige Files → in 09_Attachments\n"
        "🔗 URL allein → fragt ob clippen\n\n"
        "<b>Commands:</b>\n"
        "/today — heutige Daily anzeigen\n"
        "/briefing — Tagesbriefing (überfällig + offen + heute)\n"
        "/reminders — alle aktiven Erinnerungen\n"
        "/backup — Vault in GitHub-Repo pushen\n"
        "/reset — Conversation-Memory leeren\n\n"
        "<i>Conversation-Memory aktiv: ich erinnere mich an die letzten ~12 Turns "
        "(30 Min Timeout). Follow-ups wie 'ja' oder 'und füg X dazu' funktionieren.</i>",
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
    log.info(f"Vision: {VISION_MODEL} @ {VISION_BASE_URL}")
    if not VAULT.exists():
        log.error(f"VAULT_PATH existiert nicht: {VAULT}")
        return
    if not TEMPLATES_DIR.exists():
        log.error(f"Templates-Ordner fehlt: {TEMPLATES_DIR}")
        return
    app = Application.builder().token(TG_TOKEN).build()

    # Globale Referenz fuer Tools die JobQueue brauchen (z.B. create_reminder)
    global BOT_APP
    BOT_APP = app

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("today", handle_today))
    app.add_handler(CommandHandler("briefing", handle_briefing))
    app.add_handler(CommandHandler("reminders", handle_reminders))
    app.add_handler(CommandHandler("reset", handle_reset))
    app.add_handler(CommandHandler("backup", handle_backup))

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
        log.info("Daily-Briefing deaktiviert (BRIEFING_HOUR=0 oder Setup-Modus)")
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    log.info("Polling started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
