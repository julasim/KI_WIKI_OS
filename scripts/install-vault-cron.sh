#!/usr/bin/env bash
# install-vault-cron.sh — installiert nightly Cron für vault_toolkit.py health
# Hält Wiki-/Life-Indexe, Stats, Orphan-Reports und Lint aktuell.
# Verwendung auf VPS als root: bash install-vault-cron.sh

set -euo pipefail

VAULT_PATH="${VAULT_PATH:-/opt/vault/KI_WIKI_Vault}"
TOOLKIT="$VAULT_PATH/07_Tools/vault_toolkit.py"
CRON_FILE="/etc/cron.d/vault-toolkit"
LOG_FILE="/var/log/vault-toolkit-cron.log"

if [ "$EUID" -ne 0 ]; then
    echo "❌ Bitte als root: sudo bash $0"
    exit 1
fi

if [ ! -f "$TOOLKIT" ]; then
    echo "❌ vault_toolkit.py nicht gefunden: $TOOLKIT"
    echo "   Setze VAULT_PATH-ENV oder pass den Pfad an."
    exit 1
fi

if ! command -v python3 >/dev/null; then
    echo "Installiere python3..."
    apt-get update -qq && apt-get install -y -qq python3
fi

# Cron-Eintrag: nightly 02:30 UTC
cat > "$CRON_FILE" <<EOF
# Vault-Toolkit Health-Check — nightly 02:30
30 2 * * * root cd "$VAULT_PATH" && python3 07_Tools/vault_toolkit.py health >> "$LOG_FILE" 2>&1
EOF
chmod 644 "$CRON_FILE"
echo "✓ Cron-Eintrag: $CRON_FILE"

# Test-Lauf jetzt
echo
echo "──── Test-Lauf ────"
cd "$VAULT_PATH" && python3 07_Tools/vault_toolkit.py health 2>&1 | tail -10

echo
echo "═══════════════════════════════════════════════"
echo "   Vault-Cron aktiv ✓"
echo "═══════════════════════════════════════════════"
echo "Logs:           tail -f $LOG_FILE"
echo "Manuell:        cd $VAULT_PATH && python3 07_Tools/vault_toolkit.py health"
echo "Deinstall:      rm $CRON_FILE"
echo
echo "Was passiert nightly:"
echo "  - lint:         Frontmatter-Validation"
echo "  - orphans:      Artikel ohne Backlinks"
echo "  - broken-links: kaputte [[wikilinks]]"
echo "  - graph:        Wiki-Graph-Export (für Visualisierung)"
echo "  - stats:        Notizen/Wörter/Tags-Stats"
echo "  - index:        02_Wiki/_index.md, 03_Summaries/_index.md, 10_Life/_index.md"
echo
echo "Reports landen in $VAULT_PATH/06_Meta/health_checks/"
