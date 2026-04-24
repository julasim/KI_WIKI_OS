# VPS als Netzlaufwerk auf Windows-PC mounten

Dokumentation: wie wir `/opt/` vom Hostinger-VPS als `Z:\` auf dem Windows-PC verfügbar gemacht haben — komplett unsichtbar, automatisch beim Login, mit Obsidian-tauglicher Performance.

## Ziel

Bot schreibt auf VPS in `/opt/vault/KI_WIKI_Vault/`, Obsidian auf dem PC editiert dieselben Files via `Z:\vault\KI_WIKI_Vault`. Ein File, zwei "Türen". Kein Sync, kein Conflict, single source of truth.

## Architektur

```
Bot (Container auf VPS) ──schreibt──▶ /opt/vault/KI_WIKI_Vault/
                                              ▲
                                              │ SFTP via SSH-Key
                                              │
PC ──rclone+WinFsp──▶ Z:\ ◀──liest/schreibt── Obsidian
```

## Voraussetzungen

- VPS mit SSH-Key-Auth (siehe `SSH_SETUP.md` falls vorhanden, oder im Hauptverzeichnis dokumentiert)
- Private-Key auf PC unter `%USERPROFILE%\.ssh\vps_ki_wiki`
- Public-Key auf VPS in `/root/.ssh/authorized_keys`

## Schritt 1 — Tools installieren

### rclone (Mount-Tool)

```powershell
winget install Rclone.Rclone
```

Falls `winget` nicht verfügbar: ZIP von [rclone.org/downloads](https://rclone.org/downloads) → `rclone.exe` nach `C:\rclone\` → zur PATH-Variable hinzufügen.

### WinFsp (Windows-Kernel-Treiber für Mount-Support)

`winget install WinFsp.WinFsp` braucht Admin und scheitert oft. Manuell:

1. [winfsp.dev/rel/](https://winfsp.dev/rel/) → "WinFsp Installer" (~5 MB MSI)
2. Doppelklick → durchklicken → fertig
3. PowerShell-Fenster neu öffnen

Verifizieren:

```powershell
rclone version
```

Sollte `rclone v1.7x.x` und Windows-Info zeigen.

## Schritt 2 — rclone-Config (SFTP-Backend)

PowerShell:

```powershell
$config = @"
[vps]
type = sftp
host = 76.13.10.79
user = root
key_file = $env:USERPROFILE\.ssh\vps_ki_wiki
shell_type = unix
md5sum_command = md5sum
sha1sum_command = sha1sum
"@
New-Item -ItemType Directory -Path "$env:APPDATA\rclone" -Force | Out-Null
Set-Content -Path "$env:APPDATA\rclone\rclone.conf" -Value $config -Encoding UTF8
```

Verbindung testen:

```powershell
rclone lsd vps:/opt
```

Erwartet: Liste der Verzeichnisse unter `/opt` (z.B. `bot`, `vault`, `containerd`, ...).

## Schritt 3 — Mount-Test (Foreground, mit Logs)

```powershell
rclone mount vps:/opt Z: --vfs-cache-mode writes --vfs-write-back 1s --dir-cache-time 30s --network-mode
```

Befehl bleibt laufend (kein Prompt). In neuem Fenster oder Explorer testen:

```powershell
ls Z:\vault\KI_WIKI_Vault
```

Sollte alle Vault-Ordner zeigen. Mit `Ctrl+C` im Mount-Fenster stoppen.

### Flag-Erklärung

| Flag | Wirkung |
|---|---|
| `--vfs-cache-mode writes` | Schreibt erst lokal, sync zum VPS im Hintergrund (für Obsidian-Speicher-Bursts) |
| `--vfs-write-back 1s` | Flusht Schreib-Cache innerhalb 1 Sek |
| `--dir-cache-time 30s` | Datei-Liste max 30 Sek alt (Bot-Schreibvorgänge erscheinen schnell auf PC) |
| `--network-mode` | Windows zeigt's als Netzlaufwerk statt lokales Drive |

## Schritt 4 — Persistenter Auto-Start (unsichtbar)

Damit `Z:\` automatisch beim Login da ist, ohne sichtbares Fenster:

### 4a) VBS-Launcher schreiben (versteckt die Konsole)

```powershell
$rclonePath = (Get-Command rclone).Source
$logPath    = "$env:USERPROFILE\rclone-mount.log"
$vbsPath    = "$env:USERPROFILE\rclone-mount-hidden.vbs"

$vbs = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """$rclonePath"" mount vps:/opt Z: --vfs-cache-mode writes --vfs-write-back 1s --dir-cache-time 30s --network-mode --log-file ""$logPath"" --log-level INFO", 0, False
"@
Set-Content -Path $vbsPath -Value $vbs -Encoding ASCII
```

Was macht das VBS: ruft `rclone.exe mount ...` auf mit `WindowStyle=0` (versteckt) und `WaitOnReturn=False` (asynchron).

### 4b) Scheduled Task registrieren

```powershell
$action    = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbsPath`""
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365)

Register-ScheduledTask `
    -TaskName "rclone-mount-vps" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Hidden rclone-Mount VPS:/opt → Z: via VBS-Launcher" `
    -Force
```

### 4c) Sofort starten

```powershell
Start-ScheduledTask -TaskName "rclone-mount-vps"
```

## Verifizieren

```powershell
# Task-Status
Get-ScheduledTask -TaskName "rclone-mount-vps" | Select-Object TaskName, State

# rclone-Prozess (sollte im Hintergrund laufen, ohne Fenster)
Get-Process rclone | Select-Object Id, MainWindowTitle, StartTime

# Vault sichtbar?
ls Z:\vault\KI_WIKI_Vault
```

Bei normalem PC-Boot/Login startet der Mount automatisch. Kein sichtbares Fenster, keine Aktion nötig.

## Wartung / Troubleshooting

### Mount ist plötzlich weg

Prozess gestorben — neu starten:

```powershell
Start-ScheduledTask -TaskName "rclone-mount-vps"
```

### Manuell stoppen

```powershell
Get-Process rclone -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Logs anschauen

```powershell
Get-Content "$env:USERPROFILE\rclone-mount.log" -Tail 50
```

### Task komplett entfernen

```powershell
Unregister-ScheduledTask -TaskName "rclone-mount-vps" -Confirm:$false
Get-Process rclone | Stop-Process -Force
Remove-Item "$env:USERPROFILE\rclone-mount-hidden.vbs"
```

### Performance-Optimierung (optional)

Falls Obsidian träge fühlt (große Vaults, viele Plugins), aggressiveres Caching aktivieren — im VBS `--vfs-cache-mode writes` ersetzen durch:

```
--vfs-cache-mode full --vfs-cache-max-age 24h --vfs-cache-max-size 1G
```

`full` cached auch Lese-Operationen (Obsidian liest tausende kleine Markdown-Files pro Operation).

### Drive-Buchstabe ändern

VBS editieren → `Z:` durch z.B. `V:` ersetzen → Task neu starten.

## Sicherheits-Hinweise

- Der SSH-Key (`vps_ki_wiki`) gibt Vollzugriff auf `/opt` und alles drum herum auf dem VPS. Schütze die Key-Datei wie ein Passwort.
- VBS-Launcher liegt im Klartext im Home-Verzeichnis — enthält den vollen Mount-Befehl, aber keinen Key.
- Beim Löschen einer Datei in `Z:\vault\...` wird sie **real auf dem VPS** gelöscht. Kein Papierkorb.

## Was wir damit erreicht haben

- ✅ `Z:\` ist immer da, ohne sichtbares Fenster
- ✅ `Z:\vault\KI_WIKI_Vault` öffnet sich in Obsidian wie ein lokales Vault
- ✅ Alle Plugins (Dataview, Templater, Advanced Slides) funktionieren
- ✅ Bot-Schreibvorgänge erscheinen innerhalb 30 Sek auf PC
- ✅ PC-Edits landen sofort auf VPS
- ✅ Single source of truth: `/opt/vault/KI_WIKI_Vault/`
- ✅ Auto-Recovery bei Verbindungsabbruch (3× Retry)
