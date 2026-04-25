# Crash-Notification & Auto-Restart

Cron-basierter Health-Monitor für den Bot-Container. Schickt dir Telegram-Ping wenn was kaputt geht, versucht Auto-Restart.

## Was passiert

Alle **5 Min** prüft ein Cron-Job auf dem VPS:
- Läuft der Container `ki-wiki-bot`?
- Vergleich mit letztem bekannten State (`/var/lib/ki-wiki-bot.state`)

**State-Change** triggert Action:

| Vorher | Jetzt | Aktion |
|---|---|---|
| up | down | ⚠️ Telegram-Ping + `docker compose up -d` Auto-Restart |
| down | up | ✅ "wieder online" Telegram-Ping |
| unverändert | unverändert | nichts (kein Spam) |

## Setup

```bash
ssh -i ~/.ssh/vps_ki_wiki root@76.13.10.79
cd /opt/bot
bash update.sh                       # ziehen den scripts-Ordner
bash scripts/install-monitor.sh
```

Der Installer:
- Kopiert `check-bot.sh` nach `/usr/local/bin/`
- Legt Cron-Eintrag in `/etc/cron.d/ki-wiki-bot-monitor` an (alle 5 Min)
- Macht direkt einen Test-Lauf
- Initialisiert State-File `/var/lib/ki-wiki-bot.state`

## Was der Telegram-Ping macht

**Bei Down-Erkennung:**
```
⚠️ KI Wiki Bot ist DOWN
2026-04-25 19:42:13 UTC

Versuche Auto-Restart...
```

Hintergrund: `docker compose up -d` läuft. Beim nächsten Cron-Tick (max 5 Min später) sendet der Monitor:

```
✅ KI Wiki Bot ist wieder ONLINE
2026-04-25 19:46:51 UTC
```

→ Du weißt: war kurz weg, ist zurück. Kein Action von dir nötig.

**Bei dauerhaftem Down** (Restart schlägt fehl):
- Erste Telegram-Down-Nachricht kommt
- Folgende Cron-Ticks erkennen "still down" → keine weiteren Spam-Nachrichten
- Du musst manuell schauen warum

## Wartung

```bash
# Live-Logs anschauen
tail -f /var/log/ki-wiki-bot-monitor.log

# Aktueller State?
cat /var/lib/ki-wiki-bot.state

# Manuell triggern (ohne 5-Min-Wait)
/usr/local/bin/check-bot.sh

# Test "down"-Szenario simulieren
docker stop ki-wiki-bot
# → innerhalb 5 Min kommt Telegram-Down-Meldung + Auto-Restart läuft

# Monitor deaktivieren
rm /etc/cron.d/ki-wiki-bot-monitor
# Optional komplett entfernen:
rm /usr/local/bin/check-bot.sh /var/lib/ki-wiki-bot.state
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Keine Telegram-Pings obwohl Bot down | `cat /var/lib/ki-wiki-bot.state` — wenn "down" steht aber kein Ping kam, evtl. Cron läuft nicht. `systemctl status cron` |
| Auto-Restart hilft nicht | Manueller Check: `cd /opt/bot && docker compose logs --tail 50` |
| Spam-Nachrichten | Sollte nicht passieren da nur State-Change triggered. Falls doch: State-File löschen, dann sollte Reset funktionieren |
| TG_TOKEN-Format-Fehler im Script | `.env` editieren, sicherstellen dass keine Quotes / Spaces drum sind |

## Was das NICHT abdeckt

- Bot läuft, aber LLM-API down → Bot ist "up" laut Container-Check, aber funktional kaputt. Nicht erkennbar.
- Whisper hängt → Container läuft, aber Voice-Messages timen aus. Nicht erkennbar.
- VPS selbst down → kein Cron läuft, keine Pings. Brauchst Uptime-Robot von extern (kostenlos: uptimerobot.com gegen die VPS-IP/Port).

Für v1 reicht der Container-Check. Spätere Erweiterung: HTTP-Healthcheck-Endpoint im Bot, der auch LLM-API testet.
