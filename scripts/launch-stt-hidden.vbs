Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

Function PsQuote(value)
    PsQuote = "'" & Replace(value, "'", "''") & "'"
End Function

repoRoot = fso.GetParentFolderName(WScript.ScriptFullName)
repoRoot = fso.GetParentFolderName(repoRoot)
shell.CurrentDirectory = repoRoot
pythonExePath = fso.BuildPath(repoRoot, ".venv\Scripts\pythonw.exe")
scriptPathValue = fso.BuildPath(repoRoot, "stt.py")
configPath = fso.BuildPath(repoRoot, "config.toml")
powershellExe = Chr(34) & fso.BuildPath(shell.ExpandEnvironmentStrings("%WINDIR%"), "System32\WindowsPowerShell\v1.0\powershell.exe") & Chr(34)
consoleMode = "foreground"

If fso.FileExists(configPath) Then
    Set configFile = fso.OpenTextFile(configPath, 1)
    inStartupSection = False

    Do Until configFile.AtEndOfStream
        line = Trim(configFile.ReadLine)

        If Left(line, 1) = "[" And Right(line, 1) = "]" Then
            inStartupSection = (LCase(line) = "[startup]")
        ElseIf inStartupSection Then
            If InStr(line, "=") > 0 Then
                key = LCase(Trim(Split(line, "=")(0)))
                If key = "console_startup_mode" Then
                    value = Trim(Split(line, "=")(1))
                    If InStr(value, "#") > 0 Then
                        value = Trim(Split(value, "#")(0))
                    End If
                    value = LCase(Replace(value, Chr(34), ""))
                    If value = "background" Then
                        consoleMode = "background"
                    Else
                        consoleMode = "foreground"
                    End If
                    Exit Do
                End If
            End If
        End If
    Loop

    configFile.Close
End If

psCommand = "$env:CONVOCORTEX_OWN_CONSOLE='1'; $env:CONVOCORTEX_CONSOLE_STARTUP_MODE='" & consoleMode & "'; Start-Process -FilePath " & PsQuote(pythonExePath) & _
    " -ArgumentList @(" & PsQuote(scriptPathValue) & ")" & _
    " -WorkingDirectory " & PsQuote(repoRoot)

command = powershellExe & " -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command " & Chr(34) & psCommand & Chr(34)

shell.Run command, 0, False
