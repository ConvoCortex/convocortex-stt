using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        string processPath = Application.ExecutablePath;
        string repoRoot = !string.IsNullOrWhiteSpace(processPath)
            ? Path.GetDirectoryName(processPath) ?? AppDomain.CurrentDomain.BaseDirectory
            : AppDomain.CurrentDomain.BaseDirectory;
        string pythonExe = Path.Combine(repoRoot, ".venv", "Scripts", "pythonw.exe");
        string scriptPath = Path.Combine(repoRoot, "stt.py");
        string configPath = Path.Combine(repoRoot, "config.toml");

        if (!File.Exists(pythonExe))
        {
            MessageBox.Show("Python executable not found at " + pythonExe + ". Run 'uv sync' first.", "Convocortex STT", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        if (!File.Exists(scriptPath))
        {
            MessageBox.Show("STT entrypoint not found at " + scriptPath + ".", "Convocortex STT", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        KillExistingInstances(repoRoot);

        var launch = new ProcessStartInfo();
        launch.FileName = pythonExe;
        launch.Arguments = "\"" + scriptPath + "\"";
        launch.WorkingDirectory = repoRoot;
        launch.UseShellExecute = false;
        launch.CreateNoWindow = true;
        launch.EnvironmentVariables["CONVOCORTEX_OWN_CONSOLE"] = "1";
        launch.EnvironmentVariables["CONVOCORTEX_CONSOLE_STARTUP_MODE"] = ReadConsoleStartupMode(configPath, repoRoot);
        Process.Start(launch);
    }

    private static string UnquoteTomlString(string value)
    {
        string trimmed = value.Trim();
        if (trimmed.Length >= 2 && trimmed.StartsWith("\"") && trimmed.EndsWith("\""))
        {
            return trimmed.Substring(1, trimmed.Length - 2);
        }
        return trimmed;
    }

    private static string ReadConsoleStartupMode(string configPath, string repoRoot)
    {
        if (!File.Exists(configPath)) return "background";
        string deviceProfilesFile = "device-profiles.toml";
        bool deviceSetupInitialized = false;
        bool interactiveStartupPending = false;
        bool inStartup = false;
        bool inAudio = false;

        foreach (string rawLine in File.ReadLines(configPath))
        {
            string line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith("#")) continue;
            if (line.StartsWith("[") && line.EndsWith("]"))
            {
                inStartup = string.Equals(line, "[startup]", StringComparison.OrdinalIgnoreCase);
                inAudio = string.Equals(line, "[audio]", StringComparison.OrdinalIgnoreCase);
                continue;
            }

            int equals = line.IndexOf('=');
            if (equals < 0) continue;
            string key = line.Substring(0, equals).Trim();
            string value = line.Substring(equals + 1);
            int comment = value.IndexOf('#');
            if (comment >= 0) value = value.Substring(0, comment);
            value = value.Trim();

            if (inStartup && string.Equals(key, "device_setup_initialized", StringComparison.OrdinalIgnoreCase))
            {
                deviceSetupInitialized = string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
                if (!deviceSetupInitialized)
                {
                    interactiveStartupPending = true;
                }
                continue;
            }

            if (inStartup && key.EndsWith("_setup_initialized", StringComparison.OrdinalIgnoreCase))
            {
                bool initialized = string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
                if (!initialized)
                {
                    interactiveStartupPending = true;
                }
                continue;
            }

            if (inAudio)
            {
                if (string.Equals(key, "device_profiles_file", StringComparison.OrdinalIgnoreCase))
                {
                    string parsed = UnquoteTomlString(value);
                    if (!string.IsNullOrWhiteSpace(parsed))
                    {
                        deviceProfilesFile = parsed;
                    }
                }
                continue;
            }

            if (!inStartup) continue;
            if (!string.Equals(key, "console_startup_mode", StringComparison.OrdinalIgnoreCase)) continue;

            value = UnquoteTomlString(value).ToLowerInvariant();
            string configuredMode = value == "foreground" ? "foreground" : "background";
            string deviceProfilesPath = Path.IsPathRooted(deviceProfilesFile)
                ? deviceProfilesFile
                : Path.Combine(repoRoot, deviceProfilesFile);
            if (interactiveStartupPending || !deviceSetupInitialized || !File.Exists(deviceProfilesPath))
            {
                return "foreground";
            }
            return configuredMode;
        }

        string fallbackProfilesPath = Path.IsPathRooted(deviceProfilesFile)
            ? deviceProfilesFile
            : Path.Combine(repoRoot, deviceProfilesFile);
        if (interactiveStartupPending || !deviceSetupInitialized || !File.Exists(fallbackProfilesPath))
        {
            return "foreground";
        }
        return "background";
    }

    private static void KillExistingInstances(string repoRoot)
    {
        string repoLiteral = repoRoot.Replace("'", "''");
        string ps =
            "$repo = [System.IO.Path]::GetFullPath('" + repoLiteral + "'); " +
            "Get-CimInstance Win32_Process | Where-Object { " +
            "  $_.Name -match '^python(?:w)?(?:\\.exe)?$' -and " +
            "  $_.CommandLine -like '*stt.py*' -and " +
            "  $_.CommandLine -like ('*' + $repo + '*') " +
            "} | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }";

        var psi = new ProcessStartInfo();
        psi.FileName = "powershell.exe";
        psi.Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command \"" + ps + "\"";
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        using (Process proc = Process.Start(psi))
        {
            if (proc != null)
            {
                proc.WaitForExit(10000);
            }
        }
    }
}
