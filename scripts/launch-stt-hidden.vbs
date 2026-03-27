Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

repoRoot = fso.GetParentFolderName(WScript.ScriptFullName)
repoRoot = fso.GetParentFolderName(repoRoot)
shell.CurrentDirectory = repoRoot
pythonExe = Chr(34) & fso.BuildPath(repoRoot, ".venv\Scripts\python.exe") & Chr(34)
scriptPath = Chr(34) & fso.BuildPath(repoRoot, "stt.py") & Chr(34)
command = pythonExe & " " & scriptPath & " --background"

' 0 = hidden window, False = do not wait
shell.Run command, 0, False
