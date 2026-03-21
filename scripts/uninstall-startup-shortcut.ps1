param(
    [string]$ShortcutName = "Convocortex-STT.lnk"
)

$ErrorActionPreference = "Stop"

$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$shortcutPath = Join-Path $startupDir $ShortcutName

if (Test-Path $shortcutPath) {
    Remove-Item $shortcutPath -Force
    Write-Host "Removed startup shortcut at $shortcutPath"
}
else {
    Write-Host "Startup shortcut not found at $shortcutPath"
}
