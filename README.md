# KI Wiki Bot

Telegram-Bot der Julius' KI_WIKI_Vault als zweites Gehirn bedient.

## Features

- **Freitext** → wird via LLM klassifiziert und ins richtige Vault-File einsortiert
- **Sprachnachrichten** → lokal via faster-whisper transkribiert → wie Text behandelt
- **Fotos** → in `09_Attachments/` gespeichert + automatische Vision-Beschreibung
- **URLs** → via trafilatura zu sauberem Markdown extrahiert → `01_Raw/articles/`
- **Tools die das LLM nutzen kann**: `append_to_daily`, `create_task`, `mark_task_done`, `create_meeting`, `create_note`, `search_vault`, `read_file`, `edit_file`, `clip_url`

## Architektur

```
Telegram ─▶ Bot-Container ─schreibt──▶ /vault (= /opt/vault/KI_WIKI_Vault auf VPS)
                │
                ├─▶ OpenRouter (LLM + Vision)
                └─▶ Lokales Whisper (Voice → Text)
```

## Setup

### Voraussetzungen
- VPS mit Docker + docker-compose
- Vault unter `/opt/vault/KI_WIKI_Vault/`
- Telegram-Bot-Token (von [@BotFather](https://t.me/BotFather))
- OpenRouter-API-Key (von [openrouter.ai](https://openrouter.ai))
- Deine Telegram-User-ID (von [@userinfobot](https://t.me/userinfobot))

### Erst-Installation (interaktiv)

```bash
cd /opt/bot
bash install.sh
```

Das Skript fragt nach den 3 Credentials, schreibt `.env`, baut & startet den Container.

### Updates

```bash
cd /opt/bot
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
| "war heute am Dachboden, viel geschafft" | append_to_daily section="Abends" |
| "morgen Schreibtisch fertigskizzieren" | create_task |
| "t-dachboden-saugen erledigt" | mark_task_done |
| "Meeting morgen 15 Uhr mit Schneider" | create_meeting + create_task |
| "Was steht heute an?" | liest Daily + offene Tasks |
| "Was weiß ich über RAG?" | search_vault → Antwort |
| (forwarded URL) | clip_url |
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
