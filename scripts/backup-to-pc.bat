@echo off
REM Doppelklick-Wrapper fuer backup-to-pc.ps1
REM Speichert Backup nach %USERPROFILE%\Documents\Vault-Backups\

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backup-to-pc.ps1"
echo.
pause
