#!/usr/bin/env bash
# check-bot.sh — prüft alle 5 Min ob ki-wiki-bot läuft, schickt Telegram-Ping bei State-Change.
# Wird via Cron getriggert (siehe install-monitor.sh).
#
# Verhalten:
#   - Container läuft + war vorher down → "✅ wieder online" Telegram
#   - Container down + war vorher up → "⚠️ DOWN" Telegram + Auto-Restart-Versuch
#   - State unverändert → nichts senden (kein Spam)

set -euo pipefail

CONTAINER="ki-wiki-bot"
STATE_FILE="/var/lib/ki-wiki-bot.state"
ENV_FILE="/opt/bot/.env"
COMPOSE_DIR="/opt/bot"

# ─── ENV-Check ───
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ ENV-File fehlt: $ENV_FILE" >&2
    exit 1
fi
# .env hat keine 'export's → variables auslesen
TG_TOKEN=$(grep '^TG_TOKEN=' "$ENV_FILE" | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'")
ALLOWED_USER_ID=$(grep '^ALLOWED_USER_ID=' "$ENV_FILE" | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'")

if [ -z "$TG_TOKEN" ] || [ -z "$ALLOWED_USER_ID" ] || [ "$ALLOWED_USER_ID" = "0" ]; then
    echo "❌ TG_TOKEN oder ALLOWED_USER_ID nicht (richtig) gesetzt in $ENV_FILE" >&2
    exit 1
fi

# ─── Container-Check ───
if docker ps --filter "name=^${CONTAINER}$" --filter "status=running" --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    STATUS="up"
else
    STATUS="down"
fi

# ─── Vorheriger State ───
PREV="unknown"
[ -f "$STATE_FILE" ] && PREV=$(cat "$STATE_FILE" 2>/dev/null || echo "unknown")

# ─── Telegram senden bei State-Change ───
notify() {
    local message="$1"
    curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${ALLOWED_USER_ID}" \
        --data-urlencode "text=${message}" \
        --data-urlencode "parse_mode=HTML" \
        > /dev/null || true
}

if [ "$STATUS" != "$PREV" ]; then
    NOW="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    if [ "$STATUS" = "down" ]; then
        notify "⚠️ <b>KI Wiki Bot ist DOWN</b>
${NOW}

Versuche Auto-Restart..."
        # Auto-Restart-Versuch
        cd "$COMPOSE_DIR" && docker compose up -d 2>&1 | tail -5 >&2 || true
    else
        notify "✅ <b>KI Wiki Bot ist wieder ONLINE</b>
${NOW}"
    fi
fi

# State persistieren
mkdir -p "$(dirname "$STATE_FILE")"
echo "$STATUS" > "$STATE_FILE"
