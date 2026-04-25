#!/usr/bin/env bash
# install-backup.sh — Einmalige Einrichtung des Vault-Backups (restic + B2 + cron).
# Verwendung auf VPS: bash install-backup.sh

set -euo pipefail

ENV_FILE="/etc/restic.env"
BACKUP_SCRIPT_SRC="$(dirname "$(readlink -f "$0")")/backup-vault.sh"
BACKUP_SCRIPT_DST="/usr/local/bin/backup-vault.sh"
CRON_FILE="/etc/cron.d/vault-backup"
LOG_FILE="/var/log/vault-backup.log"

echo "═══════════════════════════════════════════════"
echo "   KI Wiki Vault — Backup-Setup"
echo "═══════════════════════════════════════════════"
echo

# ─── Check root ───
if [ "$EUID" -ne 0 ]; then
    echo "❌ Bitte als root ausführen (sudo bash $0)"
    exit 1
fi

# ─── Check restic ───
if ! command -v restic >/dev/null 2>&1; then
    echo "Installiere restic..."
    apt-get update -qq && apt-get install -y -qq restic
fi
echo "✓ restic: $(restic version | head -1)"

# ─── Existierende Config? ───
if [ -f "$ENV_FILE" ]; then
    echo "⚠️  $ENV_FILE existiert bereits."
    read -rp "Überschreiben? Bestehender Restic-Repo bleibt erhalten. [y/N] " ow
    if [[ ! "${ow,,}" =~ ^y ]]; then
        echo "Abbruch."; exit 0
    fi
fi

# ─── B2-Credentials abfragen ───
echo
echo "──── Backblaze B2 Credentials ────"
echo "Falls noch nichts da: https://www.backblaze.com/cloud-storage"
echo "  1. Sign up (kostenlos, 10 GB free tier)"
echo "  2. Buckets → Create a Bucket → Privat → Name z.B. 'ki-wiki-vault-backup'"
echo "  3. App Keys → Add a New Application Key"
echo "     - Name: 'ki-wiki-bot-backup'"
echo "     - Allow access to: nur dein neues Bucket"
echo "     - Type of Access: Read and Write"
echo "  4. Du bekommst keyID + applicationKey angezeigt — JETZT kopieren (zeigt's nur 1×)"
echo

read -rp "B2 Application Key ID         : " B2_ID
read -rp "B2 Application Key            : " B2_KEY
read -rp "B2 Bucket Name                : " B2_BUCKET
read -rp "Pfad-Präfix im Bucket [vault]  : " B2_PREFIX
B2_PREFIX=${B2_PREFIX:-vault}

if [[ -z "$B2_ID" || -z "$B2_KEY" || -z "$B2_BUCKET" ]]; then
    echo "❌ B2-Credentials unvollständig. Abbruch."
    exit 1
fi

# ─── Restic-Password ───
echo
echo "──── Restic Repository-Password ────"
echo "Verschlüsselt deine Backups. OHNE DIESES PASSWORD ist KEINE Wiederherstellung möglich!"
read -rp "Eigenes Password setzen oder generieren lassen? [enter = generieren] " PWCHOICE

if [ -z "$PWCHOICE" ]; then
    RESTIC_PASSWORD=$(openssl rand -base64 32 | tr -d '=' | head -c 32)
    echo
    echo "🔑 Generiertes Password:"
    echo "   $RESTIC_PASSWORD"
    echo
    echo "⚠️  KOPIERE DIESES PASSWORD JETZT IN DEINEN PASSWORT-MANAGER!"
    echo "   (1Password, Bitwarden, KeePass, ...)"
    echo
    read -rp "Drücke Enter wenn kopiert..."
else
    read -rsp "Password: " RESTIC_PASSWORD
    echo
    if [ ${#RESTIC_PASSWORD} -lt 16 ]; then
        echo "❌ Password zu kurz (min 16 Zeichen)."
        exit 1
    fi
fi

# ─── /etc/restic.env schreiben (root-only) ───
cat > "$ENV_FILE" <<EOF
# Restic + Backblaze B2 — KI Wiki Vault Backup
export B2_ACCOUNT_ID="$B2_ID"
export B2_ACCOUNT_KEY="$B2_KEY"
export RESTIC_REPOSITORY="b2:$B2_BUCKET:$B2_PREFIX"
export RESTIC_PASSWORD="$RESTIC_PASSWORD"
EOF
chmod 600 "$ENV_FILE"
chown root:root "$ENV_FILE"
echo "✓ $ENV_FILE geschrieben (chmod 600, root only)"

# ─── Repo initialisieren (falls noch nicht da) ───
echo
echo "──── Repository initialisieren ────"
source "$ENV_FILE"
if restic snapshots >/dev/null 2>&1; then
    echo "✓ Repository existiert bereits, behalte aktuelle Snapshots."
else
    restic init
    echo "✓ Repository initialisiert."
fi

# ─── Backup-Script installieren ───
if [ ! -f "$BACKUP_SCRIPT_SRC" ]; then
    echo "❌ backup-vault.sh nicht gefunden bei $BACKUP_SCRIPT_SRC"
    echo "   Skript erwartet, dass es im selben Ordner liegt."
    exit 1
fi
cp "$BACKUP_SCRIPT_SRC" "$BACKUP_SCRIPT_DST"
chmod +x "$BACKUP_SCRIPT_DST"
echo "✓ Backup-Script installiert: $BACKUP_SCRIPT_DST"

# ─── Cron-Eintrag (täglich 03:00) ───
cat > "$CRON_FILE" <<EOF
# KI Wiki Vault — täglicher Backup um 03:00 UTC
0 3 * * * root $BACKUP_SCRIPT_DST >> $LOG_FILE 2>&1
EOF
chmod 644 "$CRON_FILE"
echo "✓ Cron-Eintrag angelegt: $CRON_FILE"

# ─── Test-Backup ───
echo
echo "──── Test-Backup wird jetzt ausgeführt ────"
"$BACKUP_SCRIPT_DST"

echo
echo "═══════════════════════════════════════════════"
echo "   Setup fertig ✓"
echo "═══════════════════════════════════════════════"
echo
echo "Wichtige Befehle:"
echo "  Snapshot manuell:   $BACKUP_SCRIPT_DST"
echo "  Snapshots listen:   source $ENV_FILE && restic snapshots"
echo "  Letzte Logs:        tail $LOG_FILE"
echo "  Restore (latest):   source $ENV_FILE && restic restore latest --target /tmp/restored"
echo
echo "Restore-Anleitung im Detail: docs/BACKUP_SETUP.md"
echo
echo "⚠️  ERINNERUNG: Restic-Password ist im Passwort-Manager?"
