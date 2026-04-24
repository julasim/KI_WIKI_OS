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
import base64
import asyncio
import logging
import subprocess
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Optional

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
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-5")
VISION_MODEL = os.environ.get("VISION_MODEL", "anthropic/claude-sonnet-4-5")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "de")

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
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/julius-sima/ki-wiki-bot",
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


def append_to_daily(section: str, text: str) -> str:
    """Append text to today's daily under the given section."""
    if section not in VALID_SECTIONS:
        section = "Notizen & Gedanken"
    path = ensure_daily()
    content = path.read_text(encoding="utf-8")
    pattern = rf"(^|\n)## {re.escape(section)}\s*\n"
    match = re.search(pattern, content)
    if not match:
        new_content = content.rstrip() + f"\n\n## {section}\n{text}\n"
    else:
        start = match.end()
        next_h2 = re.search(r"\n## ", content[start:])
        insert_at = start + next_h2.start() if next_h2 else len(content.rstrip())
        new_content = (
            content[:insert_at].rstrip() + f"\n{text}\n\n" + content[insert_at:].lstrip()
        )
    post = frontmatter.loads(new_content)
    post["updated"] = today_iso()
    atomic_write(path, frontmatter.dumps(post) + "\n")
    return f"In Daily ({section}) eingetragen: {path.name}"


def create_task(title: str, priority: str = "medium",
                due: Optional[str] = None, area: Optional[str] = None,
                project: Optional[str] = None, context: Optional[str] = None) -> str:
    """Create a new task file in 10_Life/tasks/."""
    slug = slugify(title)
    path = TASKS_DIR / f"{slug}.md"
    n = 2
    while path.exists():
        path = TASKS_DIR / f"{slug}-{n}.md"
        n += 1
    today = today_iso()
    template = load_template("task")
    content = template.replace("{{title}}", title).replace("{{date:YYYY-MM-DD}}", today)
    post = frontmatter.loads(content)
    post["id"] = f"t-{path.stem}"
    post["title"] = title
    post["priority"] = priority
    post["status"] = "open"
    if due:
        post["due"] = due
    if area:
        post["area"] = area
    if project:
        post["project"] = project
    if context:
        post["context"] = context
    atomic_write(path, frontmatter.dumps(post) + "\n")
    # Link in heutige Daily
    try:
        append_to_daily("Heute", f"- [ ] [[t-{path.stem}]] {title}")
    except Exception as e:
        log.warning(f"Daily-Link für Task fehlgeschlagen: {e}")
    return f"Task angelegt: [[t-{path.stem}]]"


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
                   meeting_date: Optional[str] = None) -> str:
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
    post["tags"] = tags or []
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


def read_file(rel_path: str) -> str:
    """Read a file (relative to vault root). Capped at 8KB."""
    try:
        path = safe_path(rel_path)
        if not path.exists():
            return f"Datei nicht gefunden: {rel_path}"
        return path.read_text(encoding="utf-8")[:8000]
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
            new, n = re.subn(find, replace, content)
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
        "description": "Legt einen neuen Task in 10_Life/tasks/ an und verlinkt ihn unter 'Heute' in der heutigen Daily.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "due": {"type": "string", "description": "ISO-Datum YYYY-MM-DD"},
                "area": {"type": "string", "description": "Wikilink-ID einer Area, z.B. 'dachboden-umbau'"},
                "project": {"type": "string"},
                "context": {"type": "string", "enum": ["home", "work", "errand", "phone", "computer"]},
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
        "description": "Legt ein Meeting-Protokoll in 10_Life/meetings/ an.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
                "meeting_date": {"type": "string", "description": "ISO-Datum YYYY-MM-DD, default heute"},
            },
            "required": ["title"],
        },
    }},
    {"type": "function", "function": {
        "name": "create_note",
        "description": "Legt eine freie Notiz in 10_Life/notes/ an. Für längere strukturierte Inhalte (>3 Sätze, eigenes Thema), die weder Task noch Tagesreflexion sind.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
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
        "description": "Liest eine Vault-Datei (Pfad relativ zu Vault-Root, z.B. '10_Life/daily/2026-04-22.md').",
        "parameters": {
            "type": "object",
            "properties": {"rel_path": {"type": "string"}},
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
}

# ============================================================================
# System prompt (cached)
# ============================================================================

SYSTEM_PROMPT = """Du bist Kurator von Julius' persönlichem Wissens-Vault (KI_WIKI_Vault), das er via Telegram bedient. Antworte auf Deutsch.

VAULT-STRUKTUR:
- 10_Life/daily/YYYY-MM-DD.md  → Tageseinträge (Sektionen: Heute, Notizen & Gedanken, Offen / Einsortieren, Abends)
- 10_Life/tasks/<slug>.md       → Einzelne Tasks (id: t-<slug>)
- 10_Life/notes/                → Freie Notizen (für längere strukturierte Inhalte)
- 10_Life/meetings/             → Meeting-Protokolle
- 10_Life/areas/                → Lebensbereiche (Container für Tasks/Notes)
- 02_Wiki/                      → Kompiliertes Wissen (concepts/people/tools/methods)
- 01_Raw/articles/              → Externe Quellen (URLs, Artikel)

KLASSIFIZIERUNG VON FREIEM TEXT (wähle GENAU EIN Tool):
- Klare Aufgabe / Imperativ ("X anrufen", "morgen Y machen") → create_task
- Reflexion über den Tag / Bewertung → append_to_daily section="Abends"
- Halbgare Idee, Link, "merken" → append_to_daily section="Offen / Einsortieren"
- Kurze Notiz/Gedanke (<3 Sätze) → append_to_daily section="Notizen & Gedanken"
- Längere strukturierte Notiz (>3 Sätze, eigenes Thema) → create_note
- "X erledigt" / "X ist fertig" → mark_task_done (notfalls erst search_vault)
- Meeting-Inhalt ("Termin mit X", "Besprechung über Y") → create_meeting
- URL allein → clip_url
- Frage nach gespeichertem Wissen → search_vault, dann antworten mit [[wikilinks]]

DATUMSANGABEN:
- "morgen" → +1 Tag ab heute
- "übermorgen" → +2 Tage
- "nächsten Montag" → ISO-Datum berechnen
- Alle `due`-Felder als YYYY-MM-DD

ANTWORT-STIL:
- Kurz und konkret (1-3 Sätze).
- Bestätige was getan wurde, mit `[[wikilink]]` zur erstellten Datei.
- Bei Mehrdeutigkeit: kurz nachfragen statt raten.
- Keine Emojis übermäßig.

Heute ist {today}.
"""

# ============================================================================
# LLM tool-use loop
# ============================================================================

async def llm_loop(user_text: str) -> str:
    """Run tool-use loop until final answer or limit reached."""
    sys_text = SYSTEM_PROMPT.replace("{today}", today_iso())
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
        {"role": "user", "content": user_text},
    ]
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
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
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
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })
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
                f"🔓 *Setup-Modus*\n\n"
                f"Bot ist noch nicht an einen User gebunden.\n\n"
                f"Deine Telegram-User-ID: `{uid}`\n\n"
                f"Auf VPS:\n"
                f"```\n"
                f"nano /opt/bot/.env\n"
                f"# ALLOWED_USER_ID={uid} setzen\n"
                f"docker compose restart\n"
                f"```\n"
                f"Danach bin ich nur noch für dich da.",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
            return
        if uid != ALLOWED_USER_ID:
            log.warning(f"Unauthorized access attempt: user_id={uid}")
            return
        return await handler(update, ctx)
    return wrapper


async def safe_reply(update: Update, text: str) -> None:
    """Split at TG_MAX_MESSAGE chars with smart breaks."""
    if not text:
        await update.message.reply_text("(leer)")
        return
    chunks = []
    while text:
        if len(text) <= TG_MAX_MESSAGE:
            chunks.append(text)
            break
        cut = text.rfind("\n\n", 0, TG_MAX_MESSAGE)
        if cut < 1000:
            cut = text.rfind("\n", 0, TG_MAX_MESSAGE)
        if cut < 1000:
            cut = text.rfind(" ", 0, TG_MAX_MESSAGE)
        if cut < 1:
            cut = TG_MAX_MESSAGE
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    for i, chunk in enumerate(chunks, 1):
        prefix = f"({i}/{len(chunks)})\n" if len(chunks) > 1 else ""
        await update.message.reply_text(prefix + chunk)


@require_auth
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    log.info(f"text: {text[:120]}")
    await update.message.chat.send_action(constants.ChatAction.TYPING)
    try:
        reply = await llm_loop(text)
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
        reply = await llm_loop(transcript)
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
            llm.chat.completions.create,
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
        # Link in Daily
        link_block = f"![[{filename}]] — {vision_caption}"
        if user_caption:
            link_block = f"![[{filename}]] — {user_caption}\n  *Vision*: {vision_caption}"
        try:
            append_to_daily("Notizen & Gedanken", link_block)
        except Exception as e:
            log.warning(f"Daily-Link für Photo fehlgeschlagen: {e}")
        await update.message.reply_text(f"🖼 {filename}\n{vision_caption}")
    except Exception as e:
        log.exception("photo handler failed")
        await update.message.reply_text(f"Photo-Fehler: {e}")


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
        "👋 KI Wiki Bot bereit.\n\n"
        "Schreib einfach Text — ich sortier's ins Vault.\n"
        "🎤 Sprachnachrichten werden transkribiert.\n"
        "🖼 Fotos werden gespeichert + beschrieben.\n"
        "🔗 Links werden geclipt.\n\n"
        "Commands:\n"
        "/today — heutige Daily anzeigen\n"
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
    log.info(f"LLM: {LLM_MODEL}")
    log.info(f"Vision: {VISION_MODEL}")
    if not VAULT.exists():
        log.error(f"VAULT_PATH existiert nicht: {VAULT}")
        return
    if not TEMPLATES_DIR.exists():
        log.error(f"Templates-Ordner fehlt: {TEMPLATES_DIR}")
        return
    app = Application.builder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("today", handle_today))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    log.info("Polling started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
