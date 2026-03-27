param()

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $repoRoot "stt.py"
$launcherVbs = Join-Path $PSScriptRoot "launch-stt-hidden.vbs"
$wscriptExe = Join-Path $env:WINDIR "System32\wscript.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe. Run 'uv sync' first."
}

if (-not (Test-Path $scriptPath)) {
    throw "STT entrypoint not found at $scriptPath."
}

if (-not (Test-Path $launcherVbs)) {
    throw "Hidden launcher not found at $launcherVbs."
}

if (-not (Test-Path $wscriptExe)) {
    throw "wscript.exe not found at $wscriptExe."
}

$normalizedRepoRoot = [System.IO.Path]::GetFullPath($repoRoot)
$targetProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(?:w)?(?:\.exe)?$' -and
    $_.CommandLine -like '*stt.py*' -and
    $_.CommandLine -like "*$normalizedRepoRoot*"
}

if ($targetProcesses) {
    $pids = @($targetProcesses | ForEach-Object { $_.ProcessId })
    Write-Host "Stopping existing STT process(es): $($pids -join ', ')"
    $targetProcesses | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

    $deadline = (Get-Date).AddSeconds(10)
    do {
        Start-Sleep -Milliseconds 200
        $stillRunning = Get-Process -Id $pids -ErrorAction SilentlyContinue
    } while ($stillRunning -and (Get-Date) -lt $deadline)
}

$null = Start-Process -FilePath $wscriptExe -ArgumentList ('"' + $launcherVbs + '"') -WindowStyle Hidden -PassThru

Write-Host "Started STT hidden in background mode."
