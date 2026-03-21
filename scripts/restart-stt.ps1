param()

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $repoRoot "stt.py"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe. Run 'uv sync' first."
}

if (-not (Test-Path $scriptPath)) {
    throw "STT entrypoint not found at $scriptPath."
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

$wsh = New-Object -ComObject WScript.Shell
$command = '"' + $pythonExe + '" "' + $scriptPath + '" --background'
$null = $wsh.Run($command, 7, $false)

Write-Host "Started STT in background mode."
