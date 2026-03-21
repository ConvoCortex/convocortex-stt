param(
    [string]$ShortcutName = "Convocortex-STT.lnk"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$shortcutPath = Join-Path $startupDir $ShortcutName

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe. Run 'uv sync' first."
}

if (-not (Test-Path $startupDir)) {
    throw "Startup directory not found at $startupDir."
}

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonExe
$shortcut.Arguments = "stt.py --background"
$shortcut.WorkingDirectory = $repoRoot
$shortcut.WindowStyle = 7
$shortcut.Description = "Launch Convocortex STT in background mode at logon"
$shortcut.IconLocation = "$pythonExe,0"
$shortcut.Save()

Write-Host "Installed startup shortcut at $shortcutPath"
Write-Host "Target: $pythonExe stt.py --background"
