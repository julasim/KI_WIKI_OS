# Backup-Setup

Drei Optionen, von simpel zu robust. Du kannst auch mehrere parallel nutzen.

---

## Option A — PC-lokal per Script (am einfachsten, empfohlen)

**Ein PowerShell-Script auf deinem PC zieht das Vault als datierten Ordner runter.**

### Vorteile
- ✅ Maximal simpel: ein Doppelklick = Backup
- ✅ Auf deiner eigenen Festplatte → keine Cloud, kein Account, keine Kosten
- ✅ Du entscheidest wann (kein Cron)
- ✅ Datierte Snapshots → "wie sah's letzten Mittwoch aus?" geht in 5 Sek
- ✅ Optional .zip + Auto-Cleanup (behält letzte N)

### Trade-offs
- ⚠️ Lokal = wenn dein PC stirbt, Backup auch weg. Kombinier mit Option B/C für off-site.
- ⚠️ Du musst dran denken zu drücken.

### Setup (Voraussetzung: rclone-Mount läuft, siehe `RCLONE_MOUNT_SETUP.md`)

#### 1. Code auf PC aktualisieren

In PowerShell:
```powershell
cd "C:\Users\juliu\OneDrive - Mag. Georg Sima\3_Unternehmen\KI-OS\KI_WIKI_Bot"
git pull
```

#### 2. Backup auslösen

**Variante A — Doppelklick:**
- In `scripts\backup-to-pc.bat` doppelklicken
- Default: speichert nach `%USERPROFILE%\Documents\Vault-Backups\KI_WIKI_Vault_<timestamp>\`

**Variante B — PowerShell mit Optionen:**
```powershell
# Standard (Ordner-Snapshot, behält letzte 10)
.\scripts\backup-to-pc.ps1

# Komprimiert als .zip
.\scripts\backup-to-pc.ps1 -Zip

# Eigenes Ziel-Verzeichnis
.\scripts\backup-to-pc.ps1 -Dest "D:\MyBrain"

# Nur letzte 5 behalten
.\scripts\backup-to-pc.ps1 -KeepN 5

# Alles kombiniert
.\scripts\backup-to-pc.ps1 -Dest "D:\Backups" -KeepN 20 -Zip
```

#### 3. Was du siehst

```
═══════════════════════════════════════════════
  Vault-Backup → PC
═══════════════════════════════════════════════
Quelle:  vps:/opt/vault/KI_WIKI_Vault
Ziel:    C:\Users\juliu\Documents\Vault-Backups\KI_WIKI_Vault_2026-04-25_193045

Transferred:    2.3 MiB / 2.3 MiB, 100%, 1.2 MiB/s, ETA 0s
Transferred:           42 / 42, 100%

✓ Backup fertig
  Pfad:  C:\Users\juliu\Documents\Vault-Backups\KI_WIKI_Vault_2026-04-25_193045
  Größe: 2.34 MB
  MD-Files: 23

Alle Backups in:  C:\Users\juliu\Documents\Vault-Backups
Name                                   Datum
----                                   -----
KI_WIKI_Vault_2026-04-25_193045        2026-04-25 19:30
KI_WIKI_Vault_2026-04-24_211200        2026-04-24 21:12
...
```

### Restore

Backup-Ordner auf machen, Files rauskopieren wohin du willst.
Oder bei `.zip`: entpacken und Files manuell zurückpacken.

Wenn du ein **komplettes Vault-Restore** auf den VPS machen willst: rsync den Backup-Ordner via SSH zurück nach `/opt/vault/`.

### Verbessern (optional)

**Auto-Backup täglich**: Windows Task Scheduler einrichten, der `backup-to-pc.bat` täglich ausführt. Anleitung wie für rclone-Mount in `RCLONE_MOUNT_SETUP.md`.

---

## Option B — `/backup` in Telegram (off-site, gratis)

**Telegram `/backup` → Bot pusht Vault in privates GitHub-Repo.**

### Vorteile
- ✅ Off-site (überlebt Hausbrand, Festplatten-Crash etc.)
- ✅ Gratis (privates GitHub-Repo)
- ✅ Versionierung gratis (jeder Push = Commit mit Datum)
- ✅ Browsing in GitHub-UI
- ✅ Restore: `git clone <repo>`

### Trade-offs
- ⚠️ Plain-Text auf Microsoft-Servern
- ⚠️ <1 GB empfohlen, problematisch wenn Vault sehr groß wird

### Setup (~5 Min)

#### 1. Privates GitHub-Repo erstellen
- github.com → "+" → "New repository"
- Name: z.B. `KI_WIKI_Vault_Backup`
- **Private** ✓
- Initialize → NICHTS ankreuzen
- Create

#### 2. Fine-grained PAT erzeugen
- github.com → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate
- Name: `ki-wiki-bot-backup`
- Repository access: **Only select repositories** → wähle dein Backup-Repo
- Permissions → **Contents: Read and write**
- Generate → 🔑 Token sofort kopieren (1× sichtbar)

#### 3. .env auf VPS ergänzen
```bash
ssh -i ~/.ssh/vps_ki_wiki root@76.13.10.79
cd /opt/bot && nano .env
```

Ergänzen:
```
GITHUB_BACKUP_REPO=julasim/KI_WIKI_Vault_Backup
GITHUB_BACKUP_TOKEN=github_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

```bash
docker compose up -d --build   # Container braucht git+rsync (im Dockerfile drin)
```

#### 4. In Telegram triggern
```
/backup
```
oder natürlich-sprachlich: *"mach ein backup"*

### Restore via Git
```bash
git clone https://github.com/julasim/KI_WIKI_Vault_Backup.git ~/restored
ls ~/restored/vault/
```

---

## Option C — Automatisch verschlüsselt (restic + Backblaze B2)

**Cron-basiert, client-seitig verschlüsselt, dedupliziert. Für wenn's ernst wird.**

Setup-Skripte: `scripts/install-backup.sh` + `scripts/backup-vault.sh`.

### Vorteile
- ✅ Voll automatisch (täglich 03:00 UTC)
- ✅ AES-256 verschlüsselt (Backblaze sieht nur Blobs)
- ✅ Dedup + Retention-Policy (7d/4w/6m)
- ✅ Skaliert auf beliebige Größe (~5 Cent/Mo bei <10 GB)

### Trade-offs
- ⚠️ Setup ~10 Min, B2-Account, Restic-Password im Manager halten

### Setup-Kurzform
```bash
ssh -i ~/.ssh/vps_ki_wiki root@76.13.10.79
cd /opt/bot
bash scripts/install-backup.sh
```
Das Skript fragt B2-Credentials + generiert Restic-Password. Anleitung im Skript-Output.

### Restore
```bash
source /etc/restic.env
restic snapshots                                  # alle anzeigen
restic restore latest --target /tmp/restored      # latest restoren
restic mount /tmp/browse                          # snapshots als Mount durchstöbern
```

Disaster Recovery auf neuem VPS: siehe Vorgängerversion dieser Doku im Git-History.

---

## Welche soll ich nehmen?

| Szenario | Empfehlung |
|---|---|
| **Erstmal "irgendwas haben"** | Option A (PC-lokal) — 5 Min, kein Account, fertig |
| **Off-site dazu** | Option A + B parallel — PC-lokal regelmäßig, GitHub gelegentlich |
| **"Production-Grade"** | Option C (restic) — wenn das Vault wirklich kritisch wird |

Doppelter Boden ist immer gut: ein **lokales Backup** + ein **off-site Backup**. Die einfachste Kombi: A + B, Aufwand insgesamt 10 Min.
