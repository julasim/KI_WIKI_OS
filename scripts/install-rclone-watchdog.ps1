# install-rclone-watchdog.ps1
# Macht den rclone-Mount selbstheilend:
#  1. Bessere rclone-Args (--retries 999, --low-level-retries 99)
#  2. Watchdog-Task der alle 5 Min checkt + neustartet bei Tot
#
# Verwendung (einmal): .\install-rclone-watchdog.ps1
# Idempotent: kann beliebig oft ausgeführt werden.

[CmdletBinding()]
param(
    [string]$Remote = "vps:/opt",
    [string]$DriveLetter = "Z",
    [int]$WatchdogIntervalMinutes = 5
)

$ErrorActionPreference = "Stop"

Write-Host "═══════════════════════════════════════════════"
Write-Host "  rclone-Mount: Self-Healing Setup"
Write-Host "═══════════════════════════════════════════════"

# ─── Pre-Checks ───
if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
    Write-Host "❌ rclone nicht im PATH. Erst rclone installieren." -ForegroundColor Red
    exit 1
}
$rclonePath = (Get-Command rclone).Source

# ─── Variablen ───
$logPath = "$env:USERPROFILE\rclone-mount.log"
$watchdogLog = "$env:USERPROFILE\rclone-watchdog.log"
$vbsPath = "$env:USERPROFILE\rclone-mount-hidden.vbs"
$watchdogPath = "$env:USERPROFILE\rclone-watchdog.ps1"

$mountTaskName = "rclone-mount-vps"
$watchdogTaskName = "rclone-mount-watchdog"

# ─── Cleanup: alle alten Sachen weg ───
Write-Host ""
Write-Host "──── Cleanup alter Setup ────"
Stop-ScheduledTask -TaskName $mountTaskName -ErrorAction SilentlyContinue
Stop-ScheduledTask -TaskName $watchdogTaskName -ErrorAction SilentlyContinue
Get-Process rclone -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
Write-Host "✓ Alte rclone-Prozesse beendet"

# ─── 1. VBS-Launcher mit RESILIENTEN rclone-Args ───
Write-Host ""
Write-Host "──── VBS-Launcher mit Retry-Flags ────"
$vbs = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """$rclonePath"" mount $Remote ${DriveLetter}: --vfs-cache-mode writes --vfs-write-back 1s --dir-cache-time 30s --network-mode --retries 999 --low-level-retries 99 --retries-sleep 30s --timeout 300s --contimeout 60s --log-file ""$logPath"" --log-level INFO", 0, False
"@
Set-Content -Path $vbsPath -Value $vbs -Encoding ASCII
Write-Host "✓ VBS gespeichert: $vbsPath"
Write-Host "  Neue Flags: --retries 999, --low-level-retries 99, --retries-sleep 30s, --timeout 300s"

# ─── 2. Watchdog-Script ───
Write-Host ""
Write-Host "──── Watchdog-Script ────"
$watchdog = @"
# Watchdog: prüft alle $WatchdogIntervalMinutes Min ob ${DriveLetter}: lebt, restartet sonst
`$ErrorActionPreference = 'Continue'
`$logFile = '$watchdogLog'
`$now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

function Write-WatchdogLog([string]`$msg) {
    Add-Content -Path `$logFile -Value "`$now  `$msg"
}

# Quick-Check: Drive existiert + zugreifbar (mit 8s Timeout)
`$mountWorks = `$false
try {
    `$job = Start-Job -ScriptBlock {
        if (Test-Path '${DriveLetter}:\') {
            Get-ChildItem '${DriveLetter}:\' -ErrorAction Stop | Select-Object -First 1
        }
    }
    `$completed = Wait-Job -Job `$job -Timeout 8
    if (`$completed -and (`$job.State -eq 'Completed')) {
        `$result = Receive-Job -Job `$job
        if (`$result) { `$mountWorks = `$true }
    }
    Remove-Job -Job `$job -Force -ErrorAction SilentlyContinue
} catch {
    `$mountWorks = `$false
}

if (`$mountWorks) {
    # Alles gut, kein Log (sonst spamt's)
    exit 0
}

# Mount tot — heilen
Write-WatchdogLog "Mount nicht erreichbar → restart"

# Stale rclone-Prozesse killen
Get-Process rclone -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Mount-Task neu starten
try {
    Start-ScheduledTask -TaskName '$mountTaskName' -ErrorAction Stop
    Write-WatchdogLog "✓ Mount-Task neu gestartet"
} catch {
    Write-WatchdogLog "✗ Task-Start-Fehler: `$(`$_.Exception.Message)"
}
"@
Set-Content -Path $watchdogPath -Value $watchdog -Encoding UTF8
Write-Host "✓ Watchdog: $watchdogPath"

# ─── 3. Mount-Task neu registrieren ───
Write-Host ""
Write-Host "──── Mount-Task ────"
Unregister-ScheduledTask -TaskName $mountTaskName -Confirm:$false -ErrorAction SilentlyContinue

$mountAction = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbsPath`""
$mountTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$mountPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$mountSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365)

Register-ScheduledTask `
    -TaskName $mountTaskName `
    -Action $mountAction `
    -Trigger $mountTrigger `
    -Principal $mountPrincipal `
    -Settings $mountSettings `
    -Description "rclone-Mount VPS:/opt → ${DriveLetter}: (resilient: 999 retries, 30s sleep)" `
    -Force | Out-Null
Write-Host "✓ Mount-Task: $mountTaskName"

# ─── 4. Watchdog-Task registrieren ───
Write-Host ""
Write-Host "──── Watchdog-Task ────"
Unregister-ScheduledTask -TaskName $watchdogTaskName -Confirm:$false -ErrorAction SilentlyContinue

$wdAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$watchdogPath`""

# Watchdog: läuft beim Login + alle X Min
$wdTriggerLogon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$wdTriggerRepeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $WatchdogIntervalMinutes)

$wdPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$wdSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2)

Register-ScheduledTask `
    -TaskName $watchdogTaskName `
    -Action $wdAction `
    -Trigger @($wdTriggerLogon, $wdTriggerRepeat) `
    -Principal $wdPrincipal `
    -Settings $wdSettings `
    -Description "Watchdog: prüft alle $WatchdogIntervalMinutes Min ob ${DriveLetter}: lebt, restartet sonst" `
    -Force | Out-Null
Write-Host "✓ Watchdog-Task: $watchdogTaskName (alle $WatchdogIntervalMinutes Min)"

# ─── 5. Mount jetzt starten ───
Write-Host ""
Write-Host "──── Mount starten ────"
Start-ScheduledTask -TaskName $mountTaskName
Start-Sleep -Seconds 6

$rcloneProc = Get-Process rclone -ErrorAction SilentlyContinue
if ($rcloneProc) {
    Write-Host "✓ rclone läuft (PID $($rcloneProc.Id))"
} else {
    Write-Host "⚠️  rclone nicht gestartet — schau Logs: $logPath"
}

if (Test-Path "${DriveLetter}:\") {
    Write-Host "✓ ${DriveLetter}:\ gemountet"
    try {
        $items = Get-ChildItem "${DriveLetter}:\" -ErrorAction Stop | Select-Object -First 3
        Write-Host "  Inhalt: $($items.Name -join ', ')"
    } catch {
        Write-Host "  ⚠️ Drive da, aber nicht zugreifbar (evtl. noch im Aufbau)"
    }
} else {
    Write-Host "⚠️  ${DriveLetter}:\ noch nicht da. Watchdog wird in $WatchdogIntervalMinutes Min nochmal probieren."
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════"
Write-Host "  Setup fertig ✓"
Write-Host "═══════════════════════════════════════════════"
Write-Host ""
Write-Host "Was passiert ab jetzt:"
Write-Host "  • Beim Login startet rclone automatisch (mit 999 Retries)"
Write-Host "  • Wenn Verbindung temporär weg → rclone wartet 30s, retried"
Write-Host "  • Alle $WatchdogIntervalMinutes Min checkt Watchdog ob ${DriveLetter}: lebt"
Write-Host "  • Bei totem Mount: rclone wird gekillt + neu gestartet"
Write-Host ""
Write-Host "Manuelle Befehle:"
Write-Host "  Mount-Logs:    Get-Content $logPath -Tail 30"
Write-Host "  Watchdog-Logs: Get-Content $watchdogLog -Tail 30"
Write-Host "  Status:        Get-ScheduledTask -TaskName $mountTaskName, $watchdogTaskName | Format-Table"
Write-Host "  Stop alles:    Stop-ScheduledTask $watchdogTaskName; Stop-ScheduledTask $mountTaskName; Get-Process rclone | Stop-Process -Force"
