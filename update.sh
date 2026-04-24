#!/usr/bin/env bash
# update.sh — Holt neueste Version aus Git, baut Container neu, startet
# Verwendung auf VPS: bash update.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════"
echo "   KI Wiki Bot — Update"
echo "═══════════════════════════════════════════════"
echo

# ─── Git Pull ───
if [ ! -d .git ]; then
    echo "❌ Kein Git-Repo. Verwende install.sh oder klone das Repo neu."
    exit 1
fi

echo "──── Pull aus Git ────"
BEFORE=$(git rev-parse HEAD)
git pull
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ]; then
    echo "✓ Bereits auf neuestem Stand. Nichts zu tun."
    exit 0
fi

echo
echo "──── Änderungen ────"
git log --oneline "$BEFORE..$AFTER"

# ─── .env-Schema-Drift check (nur fehlende Pflicht-Felder warnen) ───
if [ -f .env ] && [ -f .env.example ]; then
    missing=$(comm -23 \
        <(grep -oE '^[A-Z_]+=' .env.example | sed 's/=$//' | sort -u) \
        <(grep -oE '^[A-Z_]+=' .env | sed 's/=$//' | sort -u))
    if [ -n "$missing" ]; then
        echo
        echo "⚠️  In deiner .env fehlen folgende Felder (laut .env.example):"
        echo "$missing" | sed 's/^/    /'
        echo
        read -rp "Trotzdem starten? [Y/n] " cont
        [[ ! "${cont,,}" =~ ^n ]] || exit 1
    fi
fi

# ─── Rebuild + Restart ───
echo
echo "──── Container neu bauen + starten ────"
docker compose up -d --build

echo
echo "✓ Update fertig."
echo "Logs: docker compose logs -f"
