param(
    [string]$ShortcutName = "Convocortex-STT.lnk"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$launcherVbs = Join-Path $PSScriptRoot "launch-stt-hidden.vbs"
$wscriptExe = Join-Path $env:WINDIR "System32\wscript.exe"
$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$shortcutPath = Join-Path $startupDir $ShortcutName

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe. Run 'uv sync' first."
}

if (-not (Test-Path $launcherVbs)) {
    throw "Launcher not found at $launcherVbs."
}

if (-not (Test-Path $wscriptExe)) {
    throw "wscript.exe not found at $wscriptExe."
}

if (-not (Test-Path $startupDir)) {
    throw "Startup directory not found at $startupDir."
}

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $wscriptExe
$shortcut.Arguments = '"' + $launcherVbs + '"'
$shortcut.WorkingDirectory = $repoRoot
$shortcut.WindowStyle = 7
$shortcut.Description = "Launch Convocortex STT at logon using the console startup mode from config.toml"
$shortcut.IconLocation = "$pythonExe,0"
$shortcut.Save()

Write-Host "Installed startup shortcut at $shortcutPath"
Write-Host "Target: $wscriptExe `"$launcherVbs`""
Write-Host "Console startup mode comes from config.toml."
