<#
Installs a self-contained nightly shutdown scheduled task.

Why this script exists here:
- You might want the PC itself to behave more like part of an ambient STT setup.
- This is tangentially related to Convocortex STT, but useful enough to keep nearby.
- The installed scheduled task does not depend on any repo file at runtime.

What it is for:
- It automates a daily shutdown prompt loop for the PC.
- The installed task is self-contained after setup.
- If you also want Convocortex STT to start automatically at logon, use
  `.\scripts\install-startup-shortcut.ps1` separately.
#>

param(
    [string]$TaskName = "Nightly Shutdown",
    [string]$StartTime = "00:00",
    [int]$DelayMinutes = 20
)

$ErrorActionPreference = "Stop"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    throw "Run this script in an elevated PowerShell window."
}

if ($DelayMinutes -lt 1) {
    throw "DelayMinutes must be 1 or greater."
}

$powershellExe = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"
if (-not (Test-Path $powershellExe)) {
    throw "powershell.exe not found at $powershellExe"
}

$scheduledTaskUser = "$env:USERDOMAIN\$env:USERNAME"

$payloadLines = @(
'param(',
'    [int]$DelayMinutes = 20',
')',
'',
'$ErrorActionPreference = "Stop"',
'',
'if ($DelayMinutes -lt 1) {',
'    throw "DelayMinutes must be 1 or greater."',
'}',
'',
'Add-Type -AssemblyName System.Windows.Forms',
'Add-Type -AssemblyName System.Drawing',
'',
'$shutdownExe = Join-Path $env:WINDIR "System32\shutdown.exe"',
'if (-not (Test-Path $shutdownExe)) {',
'    throw "shutdown.exe not found at $shutdownExe"',
'}',
'',
'$title = "Nightly Shutdown"',
'',
'function Show-ShutdownPrompt {',
'    param(',
'        [int]$PromptMinutes',
'    )',
'',
'    $script:remainingSeconds = $PromptMinutes * 60',
'    $script:choice = "timeout"',
'',
'    $form = New-Object System.Windows.Forms.Form',
'    $form.Text = $title',
'    $form.Size = New-Object System.Drawing.Size(440, 240)',
'    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen',
'    $form.TopMost = $true',
'    $form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog',
'    $form.MaximizeBox = $false',
'    $form.MinimizeBox = $false',
'',
'    $label = New-Object System.Windows.Forms.Label',
'    $label.Location = New-Object System.Drawing.Point(20, 20)',
'    $label.Size = New-Object System.Drawing.Size(390, 110)',
'    $label.Font = New-Object System.Drawing.Font("Segoe UI", 10)',
'    $form.Controls.Add($label)',
'',
'    $yesButton = New-Object System.Windows.Forms.Button',
'    $yesButton.Text = "Shut Down Now"',
'    $yesButton.Location = New-Object System.Drawing.Point(20, 145)',
'    $yesButton.Size = New-Object System.Drawing.Size(120, 32)',
'    $yesButton.Add_Click({',
'        $script:choice = "yes"',
'        $form.Close()',
'    })',
'    $form.Controls.Add($yesButton)',
'',
'    $noButton = New-Object System.Windows.Forms.Button',
'    $noButton.Text = "Ask Again Later"',
'    $noButton.Location = New-Object System.Drawing.Point(155, 145)',
'    $noButton.Size = New-Object System.Drawing.Size(120, 32)',
'    $noButton.Add_Click({',
'        $script:choice = "no"',
'        $form.Close()',
'    })',
'    $form.Controls.Add($noButton)',
'',
'    $cancelButton = New-Object System.Windows.Forms.Button',
'    $cancelButton.Text = "Cancel Tonight"',
'    $cancelButton.Location = New-Object System.Drawing.Point(290, 145)',
'    $cancelButton.Size = New-Object System.Drawing.Size(120, 32)',
'    $cancelButton.Add_Click({',
'        $script:choice = "cancel"',
'        $form.Close()',
'    })',
'    $form.Controls.Add($cancelButton)',
'',
'    $updateLabel = {',
'        $shutdownAt = (Get-Date).AddSeconds($script:remainingSeconds).ToString("h:mm tt")',
'        $minutes = [Math]::Floor($script:remainingSeconds / 60)',
'        $seconds = $script:remainingSeconds % 60',
'        $label.Text = "Shut down the PC now?`r`n`r`nIf you do nothing, the PC will shut down automatically in $minutes minute(s) and $seconds second(s), around $shutdownAt.`r`n`r`nAsk Again Later: snooze for $PromptMinutes minutes`r`nCancel Tonight: stop asking for tonight"',
'    }',
'',
'    & $updateLabel',
'',
'    $timer = New-Object System.Windows.Forms.Timer',
'    $timer.Interval = 1000',
'    $timer.Add_Tick({',
'        $script:remainingSeconds -= 1',
'        if ($script:remainingSeconds -le 0) {',
'            $script:choice = "timeout"',
'            $timer.Stop()',
'            $form.Close()',
'            return',
'        }',
'        & $updateLabel',
'    })',
'',
'    $form.Add_Shown({',
'        $timer.Start()',
'        $form.Activate()',
'    })',
'    $form.Add_FormClosed({',
'        $timer.Stop()',
'        $timer.Dispose()',
'    })',
'',
'    [void]$form.ShowDialog()',
'    $form.Dispose()',
'    return $script:choice',
'}',
'',
'while ($true) {',
'    $choice = Show-ShutdownPrompt -PromptMinutes $DelayMinutes',
'',
'    if ($choice -eq "yes" -or $choice -eq "timeout") {',
'        Start-Process -FilePath $shutdownExe -ArgumentList "/s /t 0"',
'        exit 0',
'    }',
'',
'    if ($choice -eq "cancel") {',
'        exit 0',
'    }',
'',
'    Start-Sleep -Seconds ($DelayMinutes * 60)',
'}'
)

$payload = [string]::Join("`r`n", $payloadLines)
$wrappedPayload = "& {`r`n$payload`r`n} -DelayMinutes $DelayMinutes"
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($wrappedPayload))
$arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -EncodedCommand $encodedCommand"

$action = New-ScheduledTaskAction -Execute $powershellExe -Argument $arguments
$trigger = New-ScheduledTaskTrigger -Daily -At $StartTime
$principal = New-ScheduledTaskPrincipal -UserId $scheduledTaskUser -LogonType Interactive -RunLevel Highest

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Force | Out-Null

Write-Host "Created scheduled task: $TaskName"
Write-Host "Starts prompt loop at: $StartTime"
Write-Host "Delay between prompts: $DelayMinutes minutes"
Write-Host "Task payload is self-contained inside the scheduled task."
Write-Host ""
Write-Host "Nightly behavior:"
Write-Host "  Shut Down Now = shut down now"
Write-Host "  Ask Again Later = snooze for $DelayMinutes minutes"
Write-Host "  Cancel Tonight = stop asking for tonight"
Write-Host "  No response = auto-shut down when the countdown ends"
Write-Host ""
Write-Host "Related setup:"
Write-Host "  If you want Convocortex STT to launch at logon too, run .\scripts\install-startup-shortcut.ps1"
Write-Host ""
Write-Host "Installed task details:"
schtasks /Query /TN $TaskName /V /FO LIST

