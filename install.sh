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
echo "──── Telegram ────"
echo "Falls du noch keinen Bot hast: @BotFather in Telegram → /newbot"
echo

read -rp "Telegram Bot Token                       : " TG_TOKEN
read -rp "Telegram User-ID (Enter = Setup-Modus)   : " ALLOWED_USER_ID
ALLOWED_USER_ID=${ALLOWED_USER_ID:-0}

if [[ -z "${TG_TOKEN}" ]]; then
    echo "❌ TG_TOKEN fehlt. Abbruch."
    exit 1
fi
if ! [[ "${ALLOWED_USER_ID}" =~ ^[0-9]+$ ]]; then
    echo "❌ User-ID muss eine Zahl sein (oder leer für Setup-Modus)."
    exit 1
fi
if [ "${ALLOWED_USER_ID}" = "0" ]; then
    echo "ℹ️  Setup-Modus aktiv → Bot meldet User-ID beim ersten Senden."
fi

echo
echo "──── LLM-Provider ────"
echo "1) OpenRouter   (Pay-per-Token, große Modell-Auswahl)"
echo "2) Ollama Cloud (Flat-Subscription, gpt-oss/qwen-Modelle)"
echo "3) OpenAI       (Pay-per-Token, gpt-4o/gpt-4o-mini)"
echo "4) Custom       (eigener OpenAI-API-kompatibler Endpoint)"
echo
read -rp "Wahl [1-4, default 1]                    : " PROVIDER_CHOICE
PROVIDER_CHOICE=${PROVIDER_CHOICE:-1}

case "$PROVIDER_CHOICE" in
    1)
        LLM_BASE_URL="https://openrouter.ai/api/v1"
        DEFAULT_MODEL="anthropic/claude-sonnet-4-5"
        echo "→ OpenRouter ausgewählt. Key holen: https://openrouter.ai → Keys"
        ;;
    2)
        LLM_BASE_URL="https://ollama.com/v1"
        DEFAULT_MODEL="gpt-oss:120b-cloud"
        echo "→ Ollama Cloud ausgewählt. Key holen: https://ollama.com/settings/keys"
        echo "  Hinweis: Vision ggf. separat über OpenRouter konfigurieren (per nano .env)"
        ;;
    3)
        LLM_BASE_URL="https://api.openai.com/v1"
        DEFAULT_MODEL="gpt-4o-mini"
        echo "→ OpenAI ausgewählt. Key holen: https://platform.openai.com/api-keys"
        ;;
    4)
        read -rp "Base URL                                 : " LLM_BASE_URL
        read -rp "Default Modell-Name                      : " DEFAULT_MODEL
        ;;
    *)
        echo "❌ Ungültige Wahl. Abbruch."
        exit 1
        ;;
esac

read -rp "API Key                                  : " LLM_API_KEY
if [[ -z "${LLM_API_KEY}" ]]; then
    echo "❌ API Key fehlt. Abbruch."
    exit 1
fi

# ─── Optionale Config ───
echo
echo "──── Optionale Konfiguration (Enter = Default) ────"
read -rp "Vault-Pfad [/opt/vault/KI_WIKI_Vault]      : " VAULT_PATH
VAULT_PATH=${VAULT_PATH:-/opt/vault/KI_WIKI_Vault}

read -rp "LLM-Modell [${DEFAULT_MODEL}]              : " LLM_MODEL
LLM_MODEL=${LLM_MODEL:-$DEFAULT_MODEL}

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
# Telegram
TG_TOKEN=$TG_TOKEN
ALLOWED_USER_ID=$ALLOWED_USER_ID

# LLM-Provider
LLM_API_KEY=$LLM_API_KEY
LLM_BASE_URL=$LLM_BASE_URL
LLM_MODEL=$LLM_MODEL

# Vision (Default: gleicher Provider wie LLM)
VISION_MODEL=$VISION_MODEL

# Whisper
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
