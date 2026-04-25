# backup-to-pc.ps1
# Lokaler Backup des KI_WIKI_Vault vom VPS auf den PC.
# Nutzt rclone mit der bereits konfigurierten "vps:"-SFTP-Verbindung.
#
# Verwendung:
#   .\backup-to-pc.ps1                  → Snapshot in $env:USERPROFILE\Documents\Vault-Backups\
#   .\backup-to-pc.ps1 -Zip             → zusätzlich als .zip komprimieren
#   .\backup-to-pc.ps1 -KeepN 5         → nur letzte 5 Backups behalten (Default 10)
#   .\backup-to-pc.ps1 -Dest "D:\Brain" → eigener Backup-Ordner

[CmdletBinding()]
param(
    [string]$Dest    = "$env:USERPROFILE\Documents\Vault-Backups",
    [int]   $KeepN   = 10,
    [switch]$Zip,
    [string]$Remote  = "vps:/opt/vault/KI_WIKI_Vault"
)

$ErrorActionPreference = "Stop"

# ─── Pre-Checks ───
if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
    Write-Host "❌ rclone nicht im PATH. Erst rclone installieren (siehe RCLONE_MOUNT_SETUP.md)." -ForegroundColor Red
    exit 1
}

$confPath = "$env:APPDATA\rclone\rclone.conf"
if (-not (Test-Path $confPath) -or -not (Select-String -Path $confPath -Pattern "^\[vps\]" -Quiet)) {
    Write-Host "❌ rclone-Remote 'vps:' nicht konfiguriert. Erst RCLONE_MOUNT_SETUP.md durchgehen." -ForegroundColor Red
    exit 1
}

# ─── Backup ───
$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$backupDir = Join-Path $Dest "KI_WIKI_Vault_$timestamp"

Write-Host ""
Write-Host "═══════════════════════════════════════════════"
Write-Host "  Vault-Backup → PC"
Write-Host "═══════════════════════════════════════════════"
Write-Host "Quelle:  $Remote"
Write-Host "Ziel:    $backupDir"
Write-Host ""

New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

$rcloneArgs = @(
    "copy", $Remote, $backupDir,
    "--exclude", ".obsidian/workspace*",
    "--exclude", ".obsidian/cache/**",
    "--exclude", "*.tmp",
    "--exclude", "__pycache__/**",
    "--progress",
    "--stats", "1s",
    "--transfers", "8"
)

& rclone @rcloneArgs
$rcloneExit = $LASTEXITCODE

if ($rcloneExit -ne 0) {
    Write-Host ""
    Write-Host "❌ rclone copy fehlgeschlagen (exit $rcloneExit)." -ForegroundColor Red
    Remove-Item -Recurse -Force $backupDir -ErrorAction SilentlyContinue
    exit $rcloneExit
}

# ─── Optional: Zippen ───
if ($Zip) {
    Write-Host ""
    Write-Host "Komprimiere..."
    $zipPath = "$backupDir.zip"
    Compress-Archive -Path "$backupDir\*" -DestinationPath $zipPath -CompressionLevel Optimal -Force
    Remove-Item -Recurse -Force $backupDir
    $finalPath = $zipPath
} else {
    $finalPath = $backupDir
}

# ─── Stats ───
if ($Zip) {
    $size = "{0:N2} MB" -f ((Get-Item $finalPath).Length / 1MB)
} else {
    $size = "{0:N2} MB" -f ((Get-ChildItem $finalPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB)
}
$mdCount = (Get-ChildItem $finalPath -Recurse -Filter "*.md" -ErrorAction SilentlyContinue | Measure-Object).Count

Write-Host ""
Write-Host "✓ Backup fertig"
Write-Host "  Pfad:  $finalPath"
Write-Host "  Größe: $size"
Write-Host "  MD-Files: $mdCount"

# ─── Retention ───
if ($KeepN -gt 0) {
    $pattern = if ($Zip) { "KI_WIKI_Vault_*.zip" } else { "KI_WIKI_Vault_*" }
    $existing = Get-ChildItem $Dest -Filter $pattern |
                Sort-Object LastWriteTime -Descending
    $toDelete = $existing | Select-Object -Skip $KeepN
    if ($toDelete) {
        Write-Host ""
        Write-Host "Räume alte Backups auf (behalte letzte $KeepN):"
        foreach ($old in $toDelete) {
            Write-Host "  - $($old.Name)"
            Remove-Item -Recurse -Force $old.FullName
        }
    }
}

Write-Host ""
Write-Host "Alle Backups in:  $Dest"
Get-ChildItem $Dest -Filter "KI_WIKI_Vault_*" | Sort-Object LastWriteTime -Descending |
    Select-Object @{N='Name';E={$_.Name}}, @{N='Datum';E={$_.LastWriteTime.ToString('yyyy-MM-dd HH:mm')}} |
    Format-Table -AutoSize
