#!/usr/bin/env bash
# install.sh — Interaktive Erst-Einrichtung des KI Wiki Bot
# Verwendung auf VPS: bash install.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════"
echo "   KI Wiki Bot — Setup"
echo "═══════════════════════════════════════════════"
echo

# ─── Pre-Checks ───
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker nicht installiert. Bitte erst Docker einrichten."
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "❌ Docker Compose Plugin nicht verfügbar."
    exit 1
fi

# ─── Existing .env? ───
if [ -f .env ]; then
    echo "⚠️  .env existiert bereits."
    read -rp "Überschreiben? [y/N] " ow
    if [[ ! "${ow,,}" =~ ^y ]]; then
        echo "Abbruch — bestehendes .env bleibt unverändert."
        exit 0
    fi
fi

# ─── Credentials abfragen ───
echo
echo "──── Credentials ────"
echo "Falls du noch keine hast:"
echo "  • Telegram-Token    → @BotFather in Telegram"
echo "  • Telegram User-ID  → @userinfobot in Telegram"
echo "  • OpenRouter-Key    → openrouter.ai → Keys"
echo

read -rp "Telegram Bot Token       : " TG_TOKEN
read -rp "Telegram User-ID         : " ALLOWED_USER_ID
read -rp "OpenRouter API Key       : " OPENROUTER_API_KEY

# Quick validation
if [[ -z "${TG_TOKEN}" || -z "${ALLOWED_USER_ID}" || -z "${OPENROUTER_API_KEY}" ]]; then
    echo "❌ Mindestens ein Feld leer. Abbruch."
    exit 1
fi
if ! [[ "${ALLOWED_USER_ID}" =~ ^[0-9]+$ ]]; then
    echo "❌ User-ID muss eine Zahl sein."
    exit 1
fi

# ─── Optionale Config ───
echo
echo "──── Optionale Konfiguration (Enter = Default) ────"
read -rp "Vault-Pfad [/opt/vault/KI_WIKI_Vault]      : " VAULT_PATH
VAULT_PATH=${VAULT_PATH:-/opt/vault/KI_WIKI_Vault}

read -rp "LLM-Modell [anthropic/claude-sonnet-4-5]   : " LLM_MODEL
LLM_MODEL=${LLM_MODEL:-anthropic/claude-sonnet-4-5}

read -rp "Vision-Modell [= LLM-Modell]               : " VISION_MODEL
VISION_MODEL=${VISION_MODEL:-$LLM_MODEL}

read -rp "Whisper-Modell [small/medium/large-v3]     : " WHISPER_MODEL
WHISPER_MODEL=${WHISPER_MODEL:-small}

# ─── Vault-Existenz prüfen ───
if [ ! -d "$VAULT_PATH" ]; then
    echo
    echo "⚠️  Vault-Pfad existiert nicht: $VAULT_PATH"
    read -rp "Trotzdem fortfahren? [y/N] " cont
    [[ "${cont,,}" =~ ^y ]] || exit 1
fi

# ─── .env schreiben ───
cat > .env <<EOF
TG_TOKEN=$TG_TOKEN
ALLOWED_USER_ID=$ALLOWED_USER_ID
OPENROUTER_API_KEY=$OPENROUTER_API_KEY
LLM_MODEL=$LLM_MODEL
VISION_MODEL=$VISION_MODEL
WHISPER_MODEL=$WHISPER_MODEL
WHISPER_DEVICE=cpu
WHISPER_LANG=de
EOF
chmod 600 .env

echo
echo "✓ .env geschrieben (chmod 600)"

# ─── docker-compose.yml ggf. patchen ───
if [ "$VAULT_PATH" != "/opt/vault/KI_WIKI_Vault" ]; then
    sed -i "s|/opt/vault/KI_WIKI_Vault|$VAULT_PATH|g" docker-compose.yml
    echo "✓ docker-compose.yml: Vault-Pfad auf $VAULT_PATH gesetzt"
fi

# ─── Container starten? ───
echo
read -rp "Container jetzt bauen und starten? [Y/n] " start
if [[ ! "${start,,}" =~ ^n ]]; then
    echo
    echo "──── Build + Start ────"
    docker compose up -d --build
    echo
    echo "✓ Container läuft."
    echo
    echo "Logs anschauen:"
    echo "  docker compose logs -f"
    echo
    echo "Test in Telegram:"
    echo "  Schick deinem Bot eine /start-Nachricht"
fi

echo
echo "═══════════════════════════════════════════════"
echo "   Setup fertig"
echo "═══════════════════════════════════════════════"
