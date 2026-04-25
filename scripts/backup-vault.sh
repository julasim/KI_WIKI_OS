#!/usr/bin/env bash
# backup-vault.sh — täglicher Backup des KI_WIKI_Vault nach Backblaze B2 via restic.
# Wird von install-backup.sh nach /usr/local/bin/ installiert + per cron getriggert.
# Manuell ausführen: bash /usr/local/bin/backup-vault.sh

set -euo pipefail

# Lädt B2_*, RESTIC_REPOSITORY, RESTIC_PASSWORD
if [ ! -f /etc/restic.env ]; then
    echo "❌ /etc/restic.env nicht gefunden. install-backup.sh laufen lassen."
    exit 1
fi
source /etc/restic.env

# Was wird gesichert
PATHS_TO_BACKUP=(
    /opt/vault
    /opt/bot/.env
)

echo "═══ Backup-Start: $(date -Iseconds) ═══"

# Snapshot anlegen
restic backup "${PATHS_TO_BACKUP[@]}" \
    --exclude '*.tmp' \
    --exclude '.obsidian/workspace*' \
    --exclude 'node_modules' \
    --exclude '__pycache__' \
    --tag scheduled \
    --tag "host:$(hostname)"

# Retention: 7 daily, 4 weekly, 6 monthly snapshots behalten
restic forget \
    --keep-daily 7 \
    --keep-weekly 4 \
    --keep-monthly 6 \
    --prune

# Status
echo
echo "─── Letzte 5 Snapshots ───"
restic snapshots --compact | tail -n 8

# Repository-Größe
echo
echo "─── Repo-Stats ───"
restic stats --mode raw-data 2>/dev/null | grep -E "(Total Size|Total File Count)"

echo "═══ Backup-Ende: $(date -Iseconds) ═══"
