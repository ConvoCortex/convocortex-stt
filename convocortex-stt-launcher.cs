using System.Diagnostics;
using System.Windows.Forms;

static string UnquoteTomlString(string value)
{
    var trimmed = value.Trim();
    if (trimmed.Length >= 2 && trimmed.StartsWith('"') && trimmed.EndsWith('"'))
    {
        return trimmed[1..^1];
    }
    return trimmed;
}

static string ReadConsoleStartupMode(string configPath, string repoRoot)
{
    if (!File.Exists(configPath)) return "background";
    var deviceProfilesFile = "device-profiles.toml";
    var deviceSetupInitialized = false;
    bool inStartup = false;
    bool inAudio = false;
    foreach (var rawLine in File.ReadLines(configPath))
    {
        var line = rawLine.Trim();
        if (line.Length == 0 || line.StartsWith("#")) continue;
        if (line.StartsWith("[") && line.EndsWith("]"))
        {
            inStartup = string.Equals(line, "[startup]", StringComparison.OrdinalIgnoreCase);
            inAudio = string.Equals(line, "[audio]", StringComparison.OrdinalIgnoreCase);
            continue;
        }
        var equals = line.IndexOf('=');
        if (equals < 0) continue;
        var key = line[..equals].Trim();
        var value = line[(equals + 1)..];
        var comment = value.IndexOf('#');
        if (comment >= 0) value = value[..comment];
        value = value.Trim();
        if (inAudio)
        {
            if (string.Equals(key, "device_setup_initialized", StringComparison.OrdinalIgnoreCase))
            {
                deviceSetupInitialized = string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
            }
            else if (string.Equals(key, "device_profiles_file", StringComparison.OrdinalIgnoreCase))
            {
                var parsed = UnquoteTomlString(value);
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
        var configuredMode = value == "foreground" ? "foreground" : "background";
        var deviceProfilesPath = Path.IsPathRooted(deviceProfilesFile)
            ? deviceProfilesFile
            : Path.Combine(repoRoot, deviceProfilesFile);
        if (!deviceSetupInitialized || !File.Exists(deviceProfilesPath))
        {
            return "foreground";
        }
        return configuredMode;
    }
    var fallbackProfilesPath = Path.IsPathRooted(deviceProfilesFile)
        ? deviceProfilesFile
        : Path.Combine(repoRoot, deviceProfilesFile);
    if (!deviceSetupInitialized || !File.Exists(fallbackProfilesPath))
    {
        return "foreground";
    }
    return "background";
}

static void KillExistingInstances(string repoRoot)
{
    var repoLiteral = repoRoot.Replace("'", "''");
    var ps = "$repo = [System.IO.Path]::GetFullPath('" + repoLiteral + "'); " +
             "Get-CimInstance Win32_Process | Where-Object { " +
             "  $_.Name -match '^python(?:w)?(?:\\.exe)?$' -and " +
             "  $_.CommandLine -like '*stt.py*' -and " +
             "  $_.CommandLine -like ('*' + $repo + '*') " +
             "} | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }";
    var psi = new ProcessStartInfo
    {
        FileName = "powershell.exe",
        Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command \"" + ps + "\"",
        UseShellExecute = false,
        CreateNoWindow = true,
    };
    using var proc = Process.Start(psi);
    proc?.WaitForExit(10000);
}

var processPath = Environment.ProcessPath;
var repoRoot = !string.IsNullOrWhiteSpace(processPath)
    ? Path.GetDirectoryName(processPath) ?? AppContext.BaseDirectory
    : AppContext.BaseDirectory;
var pythonExe = Path.Combine(repoRoot, ".venv", "Scripts", "pythonw.exe");
var scriptPath = Path.Combine(repoRoot, "stt.py");
var configPath = Path.Combine(repoRoot, "config.toml");

if (!File.Exists(pythonExe))
{
    MessageBox.Show($"Python executable not found at {pythonExe}. Run 'uv sync' first.", "Convocortex STT", MessageBoxButtons.OK, MessageBoxIcon.Error);
    return;
}

if (!File.Exists(scriptPath))
{
    MessageBox.Show($"STT entrypoint not found at {scriptPath}.", "Convocortex STT", MessageBoxButtons.OK, MessageBoxIcon.Error);
    return;
}

KillExistingInstances(repoRoot);

var psiLaunch = new ProcessStartInfo
{
    FileName = pythonExe,
    Arguments = '"' + scriptPath + '"',
    WorkingDirectory = repoRoot,
    UseShellExecute = false,
    CreateNoWindow = true,
};
psiLaunch.Environment["CONVOCORTEX_OWN_CONSOLE"] = "1";
psiLaunch.Environment["CONVOCORTEX_CONSOLE_STARTUP_MODE"] = ReadConsoleStartupMode(configPath, repoRoot);
Process.Start(psiLaunch);
