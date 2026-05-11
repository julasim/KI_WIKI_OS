# KI_WIKI_OS — Telegram-Bot

Thin-Client. Alle Vault-Ops laufen via MCP (`KI_WIKI_MCP`). Bot macht: Telegram-I/O, Whisper-Voice-Transkription, LLM-Tool-Use-Loop, lokale Hot-Paths (Briefing-Aggregation), Memory-Tiers, Reminder-Scheduler.

## Schlüssel-Dateien

| Datei | Was |
|---|---|
| `ki_wiki_bot.py` | Monolithische Bot-Logik (~2700 Zeilen). Tool-Loop, Telegram-Handler, Scheduler, Memory, History |
| `mcp_client.py` | Persistente async MCP-Session, lazy reconnect |
| `mcp_thin_tools.py` | Dynamic-Dispatch-Wrapper für alle MCP-Tools |
| `scripts/check-bot.sh` | Cron-Monitor (alle 5 min, Telegram-Ping bei Down/Up-Wechsel) |
| `scripts/backup-vault.sh` | Restic → Backblaze B2, täglich |

## Wichtige Patterns

### System-Prompt-Struktur (Anthropic Prompt Caching)
Statischer `SYSTEM_PROMPT` (Z. 1989) ist Block 1 mit `cache_control: ephemeral` — wird gecacht. Dynamischer Block 2 (Datum, Wochentag, Memory, aktives Projekt) ist OHNE `cache_control` — frisch pro Call. **Niemals** `{today}` o.ä. in den statischen Block — invalidiert den Cache jede Minute. Siehe Comment bei `llm_loop` (Z. 2302+).

### Wochentag explizit setzen
LLMs (auch Sonnet 4.5) berechnen Datum→Wochentag oft falsch. `dynamic_block` (Z. 2327) schreibt **"Heute ist Montag, 11. Mai 2026 (ISO: 2026-05-11)"** rein, nicht nur das ISO-Datum. Siehe `_WD_DE` Array (Z. 2321).

### History-Cross-Day-Cutoff
`_load_persistent_history` filtert Einträge älter als `HISTORY_MAX_AGE_DAYS` Vienna-Tage raus (Default 1 = nur heute). Verhindert dass gestriger "Sonntag"-Anker heutigen Kontext kontaminiert. Tunable via `.env`. Cutoff-Funktion: `_history_cutoff_ts()`.

### Memory-Tiers
- **Conversation-History** (`06_Meta/bot-memory/conversation-history.jsonl`): Rolling, today-only by default
- **Facts** (`06_Meta/bot-memory/facts.md`): Langzeit-Fakten, immer im System-Prompt
- **Preferences** (`06_Meta/bot-memory/preferences.md`): Stil/Ton, immer im System-Prompt
- **Active-Project** (`06_Meta/bot-memory/active-project.txt`): Projekt-Slug, lädt `CONTEXT.md` ins Prompt

### MCP-Client
`MCP_URL` aus env (Default `http://ki-os-mcp:5002/mcp/`, intern). Falls Production-Bot ohne Stack: setze auf TLS-URL `https://wiki-mcp.sima.business/mcp/`. Auth via Bearer-Token (`MCP_TOKEN`) — muss mit MCP-Server-`.env` matchen.

## Container

- Standalone möglich (eigene `docker-compose.yml`, `install.sh`, `update.sh`)
- Im Stack: läuft als `ki-os-bot` Container, auf `default`-Netz (kein TLS), erreicht MCP direkt
- Kein Port-Mapping — Telegram-Polling outbound only
- `TZ` nicht im Container — Python-Code nutzt `ZoneInfo("Europe/Vienna")` explizit
- Whisper-Modell wird beim ersten Start ~2-3 min geladen (faster-whisper-medium, ~500MB)

## Häufige Fallen

- **`docker compose restart bot` lädt `.env` NICHT neu** — für ENV-Änderungen `up -d --force-recreate bot`
- **Bot-Logs in UTC, Bot-Logik in Vienna** — kein Bug, nur visuell verwirrend
- **Bei "falscher Wochentag"-Symptom** → erst `grep -c <wochentag> /vault/06_Meta/bot-memory/conversation-history.jsonl` checken (History-Poisoning), dann Container-TZ
- **Zwei `today_iso()`-Definitionen** (Z. 162 und 180) — Python nimmt die zweite. Sollte gemerged werden, aktuell keine Funktions-Auswirkung

## Tools-Surface

Bot deklariert ~42 Tools (siehe `Tools: 42 declared, 42 handled — alle aligned` im Startup-Log). Mischung aus:
- MCP-Proxies (durchgereicht an MCP-Server): `task`, `append_to_daily`, `search_vault`, `read_file`, ...
- Lokale Bot-Tools (nur im Bot-Prozess): `clip_url`, `edit_file`, `goal_log`, `remember`, `forget`, `create_reminder`, ...

Tool-Liste im Code: `KNOWN_TOOLS` in `mcp_client.py` (Z. 47+) sowie Bot-eigene Definitionen weiter unten.
