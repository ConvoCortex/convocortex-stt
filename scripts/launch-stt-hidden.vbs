Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

repoRoot = fso.GetParentFolderName(WScript.ScriptFullName)
repoRoot = fso.GetParentFolderName(repoRoot)
shell.CurrentDirectory = repoRoot
pythonExe = Chr(34) & fso.BuildPath(repoRoot, ".venv\Scripts\python.exe") & Chr(34)
scriptPath = Chr(34) & fso.BuildPath(repoRoot, "stt.py") & Chr(34)
configPath = fso.BuildPath(repoRoot, "config.toml")
consoleMode = "foreground"
windowStyle = 1
command = pythonExe & " " & scriptPath

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

If consoleMode = "background" Then
    command = command & " --background"
    windowStyle = 0
End If

' 0 = hidden window, 1 = normal window, False = do not wait
shell.Run command, windowStyle, False
