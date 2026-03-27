param()

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$normalizedRepoRoot = [System.IO.Path]::GetFullPath($repoRoot)

$targetProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(?:w)?(?:\.exe)?$' -and
    $_.CommandLine -like '*stt.py*' -and
    $_.CommandLine -like "*$normalizedRepoRoot*"
}

if (-not $targetProcesses) {
    Write-Host "No STT process found for $normalizedRepoRoot"
    exit 0
}

$pids = @($targetProcesses | ForEach-Object { $_.ProcessId })
Write-Host "Stopping STT process(es): $($pids -join ', ')"
$targetProcesses | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

$deadline = (Get-Date).AddSeconds(10)
do {
    Start-Sleep -Milliseconds 200
    $stillRunning = Get-Process -Id $pids -ErrorAction SilentlyContinue
} while ($stillRunning -and (Get-Date) -lt $deadline)

$stillRunning = Get-Process -Id $pids -ErrorAction SilentlyContinue
if ($stillRunning) {
    throw "Timed out waiting for STT process(es) to exit: $($stillRunning.Id -join ', ')"
}

Write-Host "STT stopped."
