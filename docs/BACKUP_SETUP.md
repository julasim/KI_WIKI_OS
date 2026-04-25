# Backup-Setup

Zwei Optionen, je nach Anspruch. **Default-Empfehlung: Option A (manuell via Telegram).**

---

## Option A — Manueller Backup via `/backup`-Command (empfohlen)

**Du tippst `/backup` in Telegram → Bot pusht Vault in privates GitHub-Repo. Fertig.**

### Vorteile
- ✅ Gratis (GitHub privates Repo)
- ✅ Du entscheidest wann
- ✅ Restore: `git clone <repo>` — kannst du schon
- ✅ Browsing & Diff in der GitHub-Web-UI
- ✅ Versionierung gratis (jede Push = Snapshot mit Datum)

### Trade-offs
- ⚠️ **Plain-Text** auf Microsoft-Servern. Wenn du dort sehr persönliche Tagebuch-Inhalte ablegst, Datenschutz-Trade-off (gleicher wie für Code).
- ⚠️ GitHub empfiehlt <1 GB pro Repo. Bei viel Foto-Anhängen evtl. eng. → dann Option B.
- ⚠️ Du musst dran denken `/backup` zu drücken (kein Cron). Empfehlung: nach jeder größeren Session.

### Setup (~5 Min)

#### 1. Privates GitHub-Repo erstellen

- github.com → "+" → "New repository"
- Name: z.B. `KI_WIKI_Vault_Backup`
- **Private** ✓
- "Initialize this repository" → **NICHTS ankreuzen** (leer lassen)
- → "Create repository"

#### 2. Personal Access Token (Fine-grained) erzeugen

- github.com → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → "Generate new token"
- **Name**: z.B. `ki-wiki-bot-backup`
- **Resource owner**: dein User
- **Repository access**: "Only select repositories" → wähle **nur dein Backup-Repo**
- **Repository permissions** → **Contents**: "Read and write"
- → "Generate token"
- 🔑 **Token jetzt kopieren** (wird nur 1× gezeigt — beginnt mit `github_pat_...`)

#### 3. In `.env` auf VPS eintragen

```bash
ssh -i ~/.ssh/vps_ki_wiki root@76.13.10.79
cd /opt/bot
nano .env
```

Folgendes ergänzen / einfügen:

```
GITHUB_BACKUP_REPO=julasim/KI_WIKI_Vault_Backup
GITHUB_BACKUP_TOKEN=github_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Speichern (Ctrl+O, Enter, Ctrl+X). Dann:

```bash
docker compose up -d --build   # Container muss neu gebaut werden (git+rsync wurden zur Image hinzugefügt)
```

#### 4. Erstes Backup auslösen

In Telegram:

```
/backup
```

Bot antwortet mit:
- `⏳ Backup läuft...`
- Dann nach ~10-30 Sek: `✓ Backup gepusht\nRepo: julasim/KI_WIKI_Vault_Backup@a1b2c3d\nFiles: 23 .md\nZeit: 14:32:05`

Auf github.com im Repo siehst du jetzt einen ersten Commit mit dem Vault-Inhalt unter `vault/`.

### Restore

#### Komplett neu auf VPS oder anderem Rechner

```bash
git clone https://github.com/julasim/KI_WIKI_Vault_Backup.git ~/restored
ls ~/restored/vault/
# alles da
```

#### Einzelne Datei zurückholen

GitHub-Web-UI → Repo → File browsen → "..." → "Download" oder "Raw" → kopieren.

Oder via History: jeder Commit ist ein Snapshot. → "Commits" anklicken → gewünschten Stand wählen → "Browse files at this point".

### Wartung

- **Vergessen ist okay**: Wenn du eine Woche lang nicht `/backup` machst, ist nichts kaputt — beim nächsten Push landet einfach alles aktuelle drin.
- **Nichts geändert seit letztem Backup?** Bot meldet "Keine Änderungen seit letztem Backup."
- **Token rotieren**: GitHub PAT alle paar Monate neu erstellen, in `.env` ändern, `docker compose restart`.

### Troubleshooting

| Fehler | Ursache | Fix |
|---|---|---|
| "Backup nicht konfiguriert" | `.env` Felder fehlen | GITHUB_BACKUP_REPO + GITHUB_BACKUP_TOKEN setzen + restart |
| "Push-Fehler: Permission denied" | PAT-Permissions falsch | Fine-grained PAT mit Contents=R/W für genau dieses Repo erstellen |
| "Push-Fehler: Repository not found" | REPO-String falsch | Format ist `username/reponame` ohne `https://` und `.git` |
| Bot antwortet gar nicht auf `/backup` | Container nicht neu gebaut nach Dockerfile-Update | `docker compose up -d --build` |
| Push lange & timeout | Repo wurde sehr groß (Bilder etc.) | Migrate zu Option B |

---

## Option B — Automatischer verschlüsselter Backup via restic + Backblaze B2

**Für wenn**: dein Vault wächst, du Verschlüsselung willst, oder Cron statt manuell.

Setup-Skripte liegen im Repo unter `scripts/install-backup.sh` und `scripts/backup-vault.sh`. Komplette Setup-Anleitung am Ende dieser Datei (siehe Sektion "Option B — Detail").

Kurzform:
- ~5 Cent/Monat bei Backblaze B2 für <10 GB
- Client-seitig AES-256 verschlüsselt (Backblaze sieht nur Blobs)
- Dedup + Snapshots + Retention-Policy
- Cron läuft nightly 03:00 UTC

→ Für die volle Anleitung: `bash scripts/install-backup.sh` ausführen, das Skript führt durch.

---

## Beide kombinieren?

Geht. Manueller Push für "ich will jetzt einen Save-Point" + Restic-Cron für "automatische Versicherung im Hintergrund". Doppelter Boden.

---

## Option B — Detail-Anleitung

(Nur lesen wenn du Option B nutzen willst.)

### Was wird gesichert
- `/opt/vault/` — komplettes Vault inkl. `.obsidian/`
- `/opt/bot/.env` — Bot-Credentials (verschlüsselt im Backup)

### Setup

#### 1. Backblaze-B2-Account
- [backblaze.com/cloud-storage](https://www.backblaze.com/cloud-storage)
- Sign up (10 GB Free Tier ohne Credit Card)
- Bucket erstellen (Private, kein Encryption, kein Object Lock)
- Application Key erstellen, Read+Write, restricted to bucket
- 3 Werte notieren: keyID, applicationKey, Bucket-Name

#### 2. Auf VPS

```bash
ssh -i ~/.ssh/vps_ki_wiki root@76.13.10.79
cd /opt/bot
bash scripts/install-backup.sh
```

Skript fragt B2-Credentials + Restic-Password (generiert eins, **JETZT in Passwort-Manager**).

Installiert: restic via apt, /etc/restic.env, /usr/local/bin/backup-vault.sh, Cron-Eintrag.

Macht direkt einen Test-Backup.

#### 3. Verifizieren

```bash
source /etc/restic.env && restic snapshots
```

### Restore mit restic

```bash
source /etc/restic.env

# Komplett restoren
restic restore latest --target /tmp/restored

# Einzelne Datei
restic restore latest --target /tmp/r --include /opt/vault/KI_WIKI_Vault/10_Life/daily/2026-04-22.md

# Browse-Modus (mount)
restic mount /tmp/restic-browse
ls /tmp/restic-browse/snapshots/
# Ctrl+C zum Beenden

# Auf neuem VPS recovern
apt install restic
cat > /etc/restic.env <<EOF
export B2_ACCOUNT_ID="<deine keyID>"
export B2_ACCOUNT_KEY="<dein appKey>"
export RESTIC_REPOSITORY="b2:<dein-bucket>:vault"
export RESTIC_PASSWORD="<dein password aus dem manager>"
EOF
chmod 600 /etc/restic.env && source /etc/restic.env
restic restore latest --target /
```

### Sicherheit
- **Restic-Password** ist alleinige Schlüssel zur Entschlüsselung. Verlust = Backups nutzlos.
- Application Key restricted-to-bucket, kein Master-Key.
- `/etc/restic.env` chmod 600.

### Kosten
| Vault-Größe | B2-Kosten/Monat |
|---|---|
| 1 GB | ~$0.005 |
| 10 GB | ~$0.05 (Free Tier) |
| 50 GB | ~$0.25 |
