# Backup-Setup: Vault-Snapshots auf Backblaze B2

Verschlüsselte, deduplizierende, versionierte Backups via **restic** nach **Backblaze B2**.
~5 Cent/Monat bei <10 GB Vault. Tägliche Snapshots, 6 Monate Retention.

## Was wird gesichert

- `/opt/vault/` — komplettes Vault inkl. `.obsidian/` (außer `workspace*` und temp-Files)
- `/opt/bot/.env` — Bot-Credentials (verschlüsselt im Backup, fine)

**Nicht** gesichert (zu groß / re-erzeugbar):
- Whisper-Modell-Cache (re-downloadbar, 250MB-3GB je nach Modell)
- Docker-Container-State (Bot-Code ist auf GitHub)

## Retention-Policy

| Aufbewahrung | Anzahl |
|---|---|
| Tägliche Snapshots | 7 |
| Wöchentliche | 4 |
| Monatliche | 6 |

Ältere werden automatisch via `restic forget --prune` entfernt.

---

## Erst-Setup (einmalig)

### 1. Backblaze-B2-Account anlegen

- [backblaze.com/cloud-storage](https://www.backblaze.com/cloud-storage)
- Sign up (Email reicht, Free-Tier: **10 GB gratis** ohne Credit Card)
- Email-Verifizierung

### 2. Bucket erstellen

- Buckets → "Create a Bucket"
- **Name**: z.B. `ki-wiki-vault-backup` (muss global eindeutig sein, evtl. `julias-vault-backup-2026`)
- **Privacy**: Private
- **Encryption**: Disable (restic verschlüsselt selbst)
- **Object Lock**: Disable
- → Create

### 3. Application Key generieren

- App Keys → "Add a New Application Key"
- **Name**: `ki-wiki-bot-backup`
- **Allow access to**: nur dein neues Bucket
- **Type of Access**: Read and Write
- **File name prefix**: leer
- → Create

⚠️ **WICHTIG**: Die `applicationKey` wird **nur ein einziges Mal** angezeigt. Sofort kopieren.

Du brauchst dann drei Werte:
- `keyID` (z.B. `005abc...`)
- `applicationKey` (z.B. `K005...`)
- Bucket-Name

### 4. Backup auf VPS einrichten

```bash
ssh -i ~/.ssh/vps_ki_wiki root@76.13.10.79
cd /opt/bot
bash update.sh   # zieht den neuen scripts/-Ordner
bash scripts/install-backup.sh
```

Das Skript fragt:
- B2 keyID, applicationKey, Bucket-Name
- Restic-Password (Enter = generiert eins, das du in dein Passwort-Manager kopierst)

Dann installiert es:
- `restic` via apt
- `/etc/restic.env` (Credentials, root-only)
- `/usr/local/bin/backup-vault.sh` (das Backup-Script)
- `/etc/cron.d/vault-backup` (täglich 03:00 UTC)
- Initialisiert das Restic-Repo
- Macht direkt einen ersten Test-Backup

### 5. Verifizieren

```bash
source /etc/restic.env
restic snapshots --compact
```

Sollte den ersten Snapshot zeigen.

---

## Tägliche Wartung

**Nichts.** Backup läuft automatisch jede Nacht um 03:00 UTC via Cron.

## Status checken

```bash
# Letzten Backup-Run anschauen
tail /var/log/vault-backup.log

# Alle Snapshots auflisten
source /etc/restic.env && restic snapshots --compact

# Wieviel Speicher belegt?
source /etc/restic.env && restic stats
```

## Manuell triggern (ad-hoc)

```bash
bash /usr/local/bin/backup-vault.sh
```

---

## Restore-Anleitung

### Komplettes Vault wiederherstellen (latest)

```bash
source /etc/restic.env
restic restore latest --target /tmp/restored
```

Restore landet unter `/tmp/restored/opt/vault/` (Pfad-Struktur erhalten).
Dann manuell zurückkopieren wo nötig.

### Spezifischen Snapshot wiederherstellen

```bash
# Snapshots auflisten:
restic snapshots
# Output zeigt IDs wie 'a1b2c3d4'

# Den restoren:
restic restore a1b2c3d4 --target /tmp/restored-2026-04-22
```

### Nur einzelne Datei

```bash
restic restore latest --target /tmp/r --include /opt/vault/KI_WIKI_Vault/10_Life/daily/2026-04-22.md
```

### File-Browser-Modus (Files anschauen ohne Restore)

```bash
restic mount /tmp/restic-browse
# In neuem Terminal:
ls /tmp/restic-browse/snapshots/
# Cool: jeder Snapshot ist als Verzeichnis gemountet, du kannst stöbern.
# Ctrl+C im ersten Terminal beendet den Mount.
```

### Recovery auf einem komplett neuen VPS

Wenn der ursprüngliche VPS down ist und du das Vault auf einem neuen Server brauchst:

```bash
# 1. Restic installieren
apt-get update && apt-get install -y restic

# 2. /etc/restic.env neu anlegen mit den ALTEN Credentials
cat > /etc/restic.env <<EOF
export B2_ACCOUNT_ID="<deine keyID>"
export B2_ACCOUNT_KEY="<dein applicationKey>"
export RESTIC_REPOSITORY="b2:<dein-bucket>:vault"
export RESTIC_PASSWORD="<dein password aus dem Manager>"
EOF
chmod 600 /etc/restic.env
source /etc/restic.env

# 3. Restore
restic restore latest --target /
# → restored direkt nach /opt/vault/ und /opt/bot/.env

# 4. Bot-Code per git neu klonen + docker compose up
git clone https://github.com/julasim/KI_WIKI_OS.git /opt/bot
cd /opt/bot && docker compose up -d
```

---

## Sicherheits-Hinweise

- **Restic-Password** ist der einzige Schlüssel zur Entschlüsselung. **Verlierst du es, sind alle Backups Müll.** → Passwort-Manager + ggf. zweitkopie an sicherem Ort.
- **B2 Application Key** sollte auf den Backup-Bucket beschränkt sein (im Setup-Schritt), nicht "Master Key".
- `/etc/restic.env` hat `chmod 600`, root-only. Andere User auf VPS sehen es nicht.
- Backups sind **client-seitig verschlüsselt** (AES-256). Backblaze sieht nur unleserliche Blobs.

---

## Kosten-Schätzung

| Vault-Größe | B2-Kosten/Monat |
|---|---|
| 1 GB | $0.005 (~0.5 Cent) |
| 5 GB | $0.025 |
| 10 GB | $0.05 (gratis im Free Tier) |
| 50 GB | $0.25 |
| 100 GB | $0.50 |

Plus minimaler Download-Traffic bei Restores (erste 1 GB pro Tag gratis).

---

## Troubleshooting

### "Failed to connect" → B2-Credentials falsch
```bash
source /etc/restic.env && restic snapshots
# Wenn Auth-Fehler: Werte in /etc/restic.env prüfen
```

### Backup wird nicht ausgeführt nachts
```bash
# Cron läuft?
systemctl status cron
# Cron-Eintrag noch da?
cat /etc/cron.d/vault-backup
# Letzte Logs?
tail /var/log/vault-backup.log
```

### Repo "locked" Error
```bash
# Locks räumen (passiert wenn Backup-Lauf abgebrochen wurde)
source /etc/restic.env && restic unlock
```

### Repo zu groß / Pruning manuell
```bash
source /etc/restic.env
restic forget --keep-last 5 --prune  # nur letzte 5 behalten, Rest weg
```

### Restic-Password vergessen — KEIN Restore möglich
Das ist by design — Backups sind verschlüsselt, ohne Password ist nichts zu retten.
**Deshalb**: Password JETZT in Passwort-Manager.
