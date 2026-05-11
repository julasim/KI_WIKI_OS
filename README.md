# KI Wiki Bot

Telegram-Bot der Julius' KI_WIKI_Vault als zweites Gehirn bedient.
**Thin Client** seit Phase X3 — Vault-Operationen laufen via MCP-Server.

## Features

- **Freitext** → wird via LLM klassifiziert und via MCP-Tool ausgeführt
- **Sprachnachrichten** → lokal via faster-whisper transkribiert → wie Text behandelt
- **Fotos** → in `09_Attachments/` gespeichert + automatische Vision-Beschreibung
- **URLs** → via trafilatura zu sauberem Markdown extrahiert → `01_Raw/articles/`
- **LLM-Tools (via MCP-Server):** `task`, `append_to_daily`, `create_note`, `create_meeting`, `search_vault`, `read_file`, `list_files`, `goal_status`
- **LLM-Tools (lokal — Bot-spezifisch):** `clip_url`, `edit_file`, `move`, `request_delete`/`confirm_delete`, `goal_log`, `project_context`, `remember`/`forget`, `create_reminder`, `backup_vault`

## Architektur

```
Telegram ─▶ Bot-Container ─┬─▶ MCP-Server (wiki-mcp.sima.business) ─▶ /vault
                           │     • Schema-Validation
                           │     • Self-Maintenance (alle 10 Min)
                           │     • Audit-Log + Snapshots
                           │
                           ├─▶ Anthropic Claude (LLM + Vision)
                           ├─▶ Lokales Whisper (Voice → Text)
                           └─▶ /vault (read-only Hot-Paths: Briefing, Auto-Link)
```

LLM-exposed Vault-Operationen laufen via `mcp_thin_tools.py` → MCP-HTTP. Bot
hat zusätzlich Direct-FS-Mount für interne Hot-Paths (Briefing-Aggregation).

## Setup

### Voraussetzungen
- VPS mit Docker + docker-compose
- Vault unter `/opt/vault/KI_WIKI_Vault/`
- Telegram-Bot-Token (von [@BotFather](https://t.me/BotFather))
- OpenRouter-API-Key (von [openrouter.ai](https://openrouter.ai))
- Deine Telegram-User-ID (von [@userinfobot](https://t.me/userinfobot))

### Erst-Installation (interaktiv)

```bash
cd /opt/KI_WIKI_OS
bash install.sh
```

Das Skript fragt nach den Credentials, schreibt `.env`, baut & startet den Container.

**Tipp**: Die Telegram-User-ID musst du nicht parat haben — leer lassen, dann startet der Bot im *Setup-Modus*. Beim ersten Senden an den Bot meldet er dir deine ID + Anleitung wie du sie in `.env` einträgst.

### Updates

```bash
cd /opt/KI_WIKI_OS
bash update.sh
```

Pullt aus Git, baut Container neu, startet. Idempotent — wenn nichts neues, passiert nichts.

### Manuell (falls install.sh nicht passt)

```bash
cp .env.example .env
nano .env  # Werte eintragen
docker compose up -d --build
docker compose logs -f
```

## Verwendung (in Telegram)

Schreib dem Bot einfach:

| Du sagst | Bot tut |
|---|---|
| "war heute am Dachboden, viel geschafft" | `append_to_daily` (section "Abends") |
| "morgen Schreibtisch fertigskizzieren" | `task(action=create)` |
| "t-dachboden-saugen erledigt" | `task(action=done)` |
| "Meeting morgen 15 Uhr mit Schneider" | `create_meeting` + `task(action=create)` |
| "Was steht heute an?" | `get_today_agenda` (lokal aggregiert) |
| "Was weiß ich über RAG?" | `search_vault` → Antwort |
| "Wo stehe ich beim 5y-Goal?" | `goal_status` → Drift + Habits + Sport |
| (forwarded URL) | `clip_url` |
| (Sprachnachricht) | Whisper → wie Text |
| (Foto) | speichern + Vision-Caption |

### Direkte Commands

- `/today` — heutige Daily anzeigen
- `/start` — Hilfe-Text

## Wartung

```bash
# Restart
docker compose restart bot

# Stop
docker compose down

# Update Code (nach Änderung an ki_wiki_bot.py)
docker compose up -d --build

# Whisper-Modell wechseln (z.B. medium für bessere Qualität)
# In .env: WHISPER_MODEL=medium
docker compose down && docker compose up -d
```

## Sicherheit

- **Auth**: nur `ALLOWED_USER_ID` darf den Bot bedienen, alle anderen Nachrichten werden silent ignoriert.
- **Path-Traversal-Schutz**: alle File-Operationen werden gegen `VAULT_PATH` validiert.
- **Atomic Writes**: Frontmatter-Updates via tmp+rename, kein File-Korruption-Risiko.
- **Outbound-only**: Bot öffnet keinen Port. Kommuniziert nur mit Telegram + OpenRouter.

## Kosten

- **VPS**: ~4 €/Mo
- **OpenRouter** (~30 Interaktionen/Tag, mit Caching): ~2-3 €/Mo
- **Whisper**: 0 € (lokal im Container)
- **Total**: ~6-7 €/Mo
