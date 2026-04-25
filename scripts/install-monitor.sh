#!/usr/bin/env bash
# install-monitor.sh — installiert check-bot.sh + Cron-Eintrag (alle 5 Min)
# Verwendung auf VPS als root: bash install-monitor.sh

set -euo pipefail

CHECK_SCRIPT_SRC="$(dirname "$(readlink -f "$0")")/check-bot.sh"
CHECK_SCRIPT_DST="/usr/local/bin/check-bot.sh"
CRON_FILE="/etc/cron.d/ki-wiki-bot-monitor"
LOG_FILE="/var/log/ki-wiki-bot-monitor.log"

if [ "$EUID" -ne 0 ]; then
    echo "❌ Bitte als root: sudo bash $0"
    exit 1
fi

if [ ! -f "$CHECK_SCRIPT_SRC" ]; then
    echo "❌ check-bot.sh nicht im selben Ordner gefunden"
    exit 1
fi

cp "$CHECK_SCRIPT_SRC" "$CHECK_SCRIPT_DST"
chmod +x "$CHECK_SCRIPT_DST"
echo "✓ Script installiert: $CHECK_SCRIPT_DST"

# Cron-Eintrag: alle 5 Minuten
cat > "$CRON_FILE" <<EOF
# KI Wiki Bot — Health-Monitor (alle 5 Min)
*/5 * * * * root $CHECK_SCRIPT_DST >> $LOG_FILE 2>&1
EOF
chmod 644 "$CRON_FILE"
echo "✓ Cron-Eintrag: $CRON_FILE (alle 5 Min)"

# Test-Lauf
echo
echo "──── Test-Lauf ────"
"$CHECK_SCRIPT_DST"
echo "✓ State-File:"
cat /var/lib/ki-wiki-bot.state 2>/dev/null || echo "(noch nicht da)"

echo
echo "═══════════════════════════════════════════════"
echo "   Monitor aktiv ✓"
echo "═══════════════════════════════════════════════"
echo "Logs:           tail -f $LOG_FILE"
echo "State:          cat /var/lib/ki-wiki-bot.state"
echo "Manueller Test: $CHECK_SCRIPT_DST"
echo "Deinstallieren: rm $CRON_FILE $CHECK_SCRIPT_DST /var/lib/ki-wiki-bot.state"
echo
echo "Wenn der Bot down geht, kommt eine Telegram-Nachricht innerhalb von 5 Min."
