using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

internal static class Program
{
    private static string _launcherLogPath = "";
    private static Mutex? _singleInstanceMutex;
    private static EventWaitHandle? _restartSignal;

    [STAThread]
    private static void Main()
    {
        try
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            string repoRoot = ResolveRepoRoot();
            _launcherLogPath = Path.Combine(repoRoot, "launcher-debug.log");
            Log("launcher start");
            string instanceKey = BuildInstanceKey(repoRoot);
            string pythonExe = Path.Combine(repoRoot, ".venv", "Scripts", "python.exe");
            string scriptPath = Path.Combine(repoRoot, "stt.py");
            string configPath = Path.Combine(repoRoot, "config.toml");
            if (!AcquireSingleInstance(instanceKey))
            {
                Log("restart signal sent to existing launcher");
                return;
            };

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

            string startupMode = ReadConsoleStartupMode(configPath, repoRoot);
            string consoleToggleHotkey = ReadConsoleToggleHotkey(configPath);

            KillExistingInstances(repoRoot);
            Application.Run(new LauncherContext(repoRoot, pythonExe, scriptPath, startupMode, consoleToggleHotkey, _restartSignal!));
        }
        catch (Exception exc)
        {
            Log("fatal launcher error: " + exc);
            MessageBox.Show(exc.ToString(), "Convocortex STT Launcher", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private static string BuildInstanceKey(string repoRoot)
    {
        string normalized = Path.GetFullPath(repoRoot).ToLowerInvariant();
        var builder = new StringBuilder();
        foreach (char ch in normalized)
        {
            if (char.IsLetterOrDigit(ch))
            {
                builder.Append(ch);
            }
            else
            {
                builder.Append('_');
            }
        }
        return builder.ToString();
    }

    private static bool AcquireSingleInstance(string instanceKey)
    {
        string mutexName = "Local\\ConvocortexSttLauncherMutex_" + instanceKey;
        string eventName = "Local\\ConvocortexSttLauncherRestart_" + instanceKey;
        bool createdNew;
        _singleInstanceMutex = new Mutex(true, mutexName, out createdNew);
        if (createdNew)
        {
            _restartSignal = new EventWaitHandle(false, EventResetMode.AutoReset, eventName);
            return true;
        }
        try
        {
            using EventWaitHandle signal = EventWaitHandle.OpenExisting(eventName);
            signal.Set();
        }
        catch
        {
        }
        _singleInstanceMutex.Dispose();
        _singleInstanceMutex = null;
        return false;
    }

    private static void Log(string message)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(_launcherLogPath))
            {
                return;
            }
            File.AppendAllText(_launcherLogPath, DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff ") + message + Environment.NewLine);
        }
        catch
        {
        }
    }

    private static string ResolveRepoRoot()
    {
        string startDir = !string.IsNullOrWhiteSpace(Application.ExecutablePath)
            ? Path.GetDirectoryName(Application.ExecutablePath) ?? AppDomain.CurrentDomain.BaseDirectory
            : AppDomain.CurrentDomain.BaseDirectory;
        string current = Path.GetFullPath(startDir);
        while (!string.IsNullOrWhiteSpace(current))
        {
            if (File.Exists(Path.Combine(current, "stt.py")) && File.Exists(Path.Combine(current, "config.toml")))
            {
                return current;
            }
            string? parent = Directory.GetParent(current)?.FullName;
            if (string.IsNullOrWhiteSpace(parent) || string.Equals(parent, current, StringComparison.OrdinalIgnoreCase))
            {
                break;
            }
            current = parent;
        }
        return Path.GetFullPath(startDir);
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

    private static string ReadConsoleToggleHotkey(string configPath)
    {
        if (!File.Exists(configPath)) return "shift+f8";
        bool inHotkeys = false;
        foreach (string rawLine in File.ReadLines(configPath))
        {
            string line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith("#")) continue;
            if (line.StartsWith("[") && line.EndsWith("]"))
            {
                inHotkeys = string.Equals(line, "[hotkeys]", StringComparison.OrdinalIgnoreCase);
                continue;
            }
            if (!inHotkeys) continue;
            int equals = line.IndexOf('=');
            if (equals < 0) continue;
            string key = line.Substring(0, equals).Trim();
            if (!string.Equals(key, "console_toggle", StringComparison.OrdinalIgnoreCase)) continue;
            string value = line.Substring(equals + 1);
            int comment = value.IndexOf('#');
            if (comment >= 0) value = value.Substring(0, comment);
            string parsed = UnquoteTomlString(value).Trim();
            return string.IsNullOrWhiteSpace(parsed) ? "shift+f8" : parsed;
        }
        return "shift+f8";
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

    private sealed class LauncherContext : ApplicationContext
    {
        private const string TerminalWindowTitle = "Convocortex STT";

        private readonly string _repoRoot;
        private readonly string _pythonExe;
        private readonly string _scriptPath;
        private readonly string _startupMode;
        private readonly NotifyIcon _trayIcon;
        private readonly ToolStripMenuItem _showConsoleMenuItem;
        private readonly ToolStripMenuItem _restartRuntimeMenuItem;
        private readonly ToolStripMenuItem _openFolderMenuItem;
        private readonly ToolStripMenuItem _openLogMenuItem;
        private readonly ToolStripMenuItem _exitMenuItem;
        private readonly LauncherHotkeyWindow _hotkeyWindow;
        private readonly System.Windows.Forms.Timer _timer;
        private readonly EventWaitHandle _restartSignal;
        private readonly RegisteredWaitHandle _restartWait;
        private readonly SynchronizationContext _uiContext;
        private readonly string _runtimeConsoleLogPath;
        private readonly string _runtimeDebugLogPath;

        private IntPtr _terminalHwnd = IntPtr.Zero;
        private Process? _runtimeProcess;
        private bool _exitRequested;

        public LauncherContext(string repoRoot, string pythonExe, string scriptPath, string startupMode, string consoleToggleHotkey, EventWaitHandle restartSignal)
        {
            _repoRoot = repoRoot;
            _pythonExe = pythonExe;
            _scriptPath = scriptPath;
            _startupMode = startupMode;
            _restartSignal = restartSignal;
            _uiContext = SynchronizationContext.Current ?? new WindowsFormsSynchronizationContext();
            _runtimeConsoleLogPath = Path.Combine(_repoRoot, "runtime-console.log");
            _runtimeDebugLogPath = Path.Combine(_repoRoot, "stt-debug.log");

            _showConsoleMenuItem = new ToolStripMenuItem("Show Console", null, (_, __) => ShowConsole());
            _restartRuntimeMenuItem = new ToolStripMenuItem("Restart Runtime", null, (_, __) => RestartStt());
            _openFolderMenuItem = new ToolStripMenuItem("Open Folder", null, (_, __) => OpenFolder());
            _openLogMenuItem = new ToolStripMenuItem("Open Log", null, (_, __) => OpenLog());
            _exitMenuItem = new ToolStripMenuItem("Exit", null, (_, __) => ExitStt());
            var menu = new ContextMenuStrip();
            menu.Items.Add(_showConsoleMenuItem);
            menu.Items.Add(_restartRuntimeMenuItem);
            menu.Items.Add(_openFolderMenuItem);
            menu.Items.Add(_openLogMenuItem);
            menu.Items.Add(new ToolStripSeparator());
            menu.Items.Add(_exitMenuItem);

            _trayIcon = new NotifyIcon
            {
                Text = "Convocortex STT",
                Icon = SystemIcons.Application,
                ContextMenuStrip = menu,
                Visible = true,
            };
            _trayIcon.MouseClick += (_, args) =>
            {
                if (args.Button == MouseButtons.Left)
                {
                    ToggleConsole();
                }
            };

            _hotkeyWindow = new LauncherHotkeyWindow(this);
            _hotkeyWindow.Register(consoleToggleHotkey);
            _restartWait = ThreadPool.RegisterWaitForSingleObject(
                _restartSignal,
                (_, __) => _uiContext.Post(_ =>
                {
                    Program.Log("restart signal received");
                    RestartStt();
                }, null),
                null,
                -1,
                false
            );

            _timer = new System.Windows.Forms.Timer();
            _timer.Interval = 300;
            _timer.Tick += (_, __) => OnTick();
            _timer.Start();

            StartRuntime();
            if (string.Equals(_startupMode, "foreground", StringComparison.OrdinalIgnoreCase))
            {
                ShowConsole();
            }
        }

        private void StartRuntime()
        {
            Directory.CreateDirectory(_repoRoot);
            using (File.Open(_runtimeConsoleLogPath, FileMode.Create, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete))
            {
            }
            var psi = new ProcessStartInfo
            {
                FileName = _pythonExe,
                Arguments = "\"" + _scriptPath + "\"",
                WorkingDirectory = _repoRoot,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            psi.Environment["CONVOCORTEX_DISABLE_INTERNAL_TRAY"] = "1";
            var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
            if (!process.Start())
            {
                throw new InvalidOperationException("Could not start STT runtime.");
            }
            _runtimeProcess = process;
            BeginPipeCapture(process, process.StandardOutput, _runtimeConsoleLogPath);
            BeginPipeCapture(process, process.StandardError, _runtimeConsoleLogPath);
            Program.Log("runtime started pid=" + process.Id);
        }

        private static void BeginPipeCapture(Process process, StreamReader reader, string destinationPath)
        {
            Thread thread = new Thread(() =>
            {
                try
                {
                    using var writer = new StreamWriter(new FileStream(destinationPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete))
                    {
                        AutoFlush = true,
                    };
                    string? line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        writer.WriteLine(line);
                    }
                }
                catch
                {
                }
            });
            thread.IsBackground = true;
            thread.Name = "runtime-log-" + process.Id;
            thread.Start();
        }

        private void LaunchConsoleViewer()
        {
            EnsureConsoleViewerLogExists();
            string viewerScriptPath = EnsureConsoleViewerScript();
            string arguments =
                "-w new " +
                "nt " +
                "--title \"" + TerminalWindowTitle + "\" " +
                "--suppressApplicationTitle " +
                "powershell.exe -NoLogo -NoExit -ExecutionPolicy Bypass -File \"" + viewerScriptPath + "\"";
            Program.Log("launch console viewer requested");
            if (TryLaunchWithWtAlias(arguments))
            {
                Program.Log("launch console viewer via wt.exe");
                return;
            }
            if (TryLaunchWithTerminalProcessPath(arguments))
            {
                Program.Log("launch console viewer via existing WindowsTerminal.exe path");
                return;
            }
            throw new InvalidOperationException("Could not launch Windows Terminal viewer.");
        }

        private void EnsureConsoleViewerLogExists()
        {
            Directory.CreateDirectory(_repoRoot);
            using (File.Open(_runtimeConsoleLogPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete))
            {
            }
        }

        private string EnsureConsoleViewerScript()
        {
            string launcherTempDir = Path.Combine(_repoRoot, ".tmp", "launcher");
            Directory.CreateDirectory(launcherTempDir);
            string scriptPath = Path.Combine(launcherTempDir, "tail-runtime-console.ps1");
            string escapedLogPath = _runtimeConsoleLogPath.Replace("'", "''");
            string script =
                "$path = '" + escapedLogPath + "'" + Environment.NewLine +
                "if (!(Test-Path $path)) { New-Item -ItemType File -Path $path -Force | Out-Null }" + Environment.NewLine +
                "Get-Content -Path $path -Wait -Tail 200" + Environment.NewLine;
            File.WriteAllText(scriptPath, script, Encoding.UTF8);
            return scriptPath;
        }

        private static string EscapeCmd(string value)
        {
            return value.Replace("\"", "\"\"");
        }

        private static bool TryLaunchWithWtAlias(string arguments)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "wt.exe",
                    Arguments = arguments,
                    UseShellExecute = true,
                    WorkingDirectory = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                };
                Process.Start(psi);
                return true;
            }
            catch (Exception exc)
            {
                Program.Log("wt.exe launch failed: " + exc.Message);
                return false;
            }
        }

        private static bool TryLaunchWithTerminalProcessPath(string arguments)
        {
            try
            {
                foreach (Process process in Process.GetProcessesByName("WindowsTerminal"))
                {
                    string path = process.MainModule?.FileName ?? "";
                    if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                    {
                        continue;
                    }
                    var psi = new ProcessStartInfo
                    {
                        FileName = "cmd.exe",
                        Arguments = "/c start \"\" \"" + path + "\" " + arguments,
                        UseShellExecute = false,
                        CreateNoWindow = true,
                        WorkingDirectory = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    };
                    Process.Start(psi);
                    return true;
                }
            }
            catch (Exception exc)
            {
                Program.Log("WindowsTerminal.exe path launch failed: " + exc.Message);
            }
            return false;
        }

        private void OnTick()
        {
            if (_exitRequested)
            {
                return;
            }

            if (_terminalHwnd != IntPtr.Zero && !IsWindow(_terminalHwnd))
            {
                _terminalHwnd = IntPtr.Zero;
            }

            if (_runtimeProcess != null)
            {
                try
                {
                    if (_runtimeProcess.HasExited)
                    {
                        Program.Log("runtime exited code=" + _runtimeProcess.ExitCode);
                        _runtimeProcess.Dispose();
                        _runtimeProcess = null;
                    }
                }
                catch
                {
                    _runtimeProcess = null;
                }
            }
        }

        public void ShowConsole()
        {
            if (_terminalHwnd == IntPtr.Zero)
            {
                _terminalHwnd = FindWindowByTitle(TerminalWindowTitle);
            }
            if (_terminalHwnd == IntPtr.Zero)
            {
                LaunchConsoleViewer();
                return;
            }
            ShowWindow(_terminalHwnd, ShowWindowCommands.Restore);
            ShowWindow(_terminalHwnd, ShowWindowCommands.Show);
            SetForegroundWindow(_terminalHwnd);
        }

        public void ToggleConsole()
        {
            if (_terminalHwnd == IntPtr.Zero)
            {
                _terminalHwnd = FindWindowByTitle(TerminalWindowTitle);
                if (_terminalHwnd == IntPtr.Zero)
                {
                    ShowConsole();
                    return;
                }
            }

            if (IsWindowVisible(_terminalHwnd))
            {
                ShowWindow(_terminalHwnd, ShowWindowCommands.Hide);
            }
            else
            {
                ShowConsole();
            }
        }

        public void RestartStt()
        {
            Program.Log("restart requested");
            try
            {
                if (_runtimeProcess != null && !_runtimeProcess.HasExited)
                {
                    _runtimeProcess.Kill(entireProcessTree: true);
                    _runtimeProcess.WaitForExit(5000);
                }
            }
            catch
            {
            }
            try
            {
                KillExistingInstances(_repoRoot);
            }
            catch
            {
            }
            _runtimeProcess?.Dispose();
            _runtimeProcess = null;
            StartRuntime();
        }

        private bool IsConsoleVisible()
        {
            return _terminalHwnd != IntPtr.Zero && IsWindowVisible(_terminalHwnd);
        }

        private void OpenFolder()
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = "\"" + _repoRoot + "\"",
                UseShellExecute = true,
            });
        }

        private void OpenLog()
        {
            string path = File.Exists(_runtimeDebugLogPath) ? _runtimeDebugLogPath : _runtimeConsoleLogPath;
            if (!File.Exists(path))
            {
                EnsureConsoleViewerLogExists();
                path = _runtimeConsoleLogPath;
            }
            Process.Start(new ProcessStartInfo
            {
                FileName = path,
                UseShellExecute = true,
            });
        }

        private void ExitStt()
        {
            _exitRequested = true;
            _timer.Stop();
            try
            {
                if (_runtimeProcess != null && !_runtimeProcess.HasExited)
                {
                    _runtimeProcess.Kill(entireProcessTree: true);
                    _runtimeProcess.WaitForExit(5000);
                }
            }
            catch
            {
            }
            try
            {
                KillExistingInstances(_repoRoot);
            }
            catch
            {
            }
            try
            {
                if (_terminalHwnd != IntPtr.Zero && IsWindow(_terminalHwnd))
                {
                    PostMessage(_terminalHwnd, WM_CLOSE, IntPtr.Zero, IntPtr.Zero);
                }
            }
            catch
            {
            }
            ExitThread();
        }

        protected override void ExitThreadCore()
        {
            _timer.Stop();
            _restartWait.Unregister(null);
            _hotkeyWindow.Dispose();
            _trayIcon.Visible = false;
            _trayIcon.Dispose();
            _runtimeProcess?.Dispose();
            base.ExitThreadCore();
        }

        private static IntPtr FindWindowByTitle(string title)
        {
            IntPtr found = IntPtr.Zero;
            EnumWindows((hwnd, _) =>
            {
                if (!IsWindow(hwnd))
                {
                    return true;
                }
                int length = GetWindowTextLength(hwnd);
                if (length <= 0)
                {
                    return true;
                }
                var builder = new StringBuilder(length + 1);
                GetWindowText(hwnd, builder, builder.Capacity);
                if (string.Equals(builder.ToString(), title, StringComparison.Ordinal))
                {
                    found = hwnd;
                    return false;
                }
                return true;
            }, IntPtr.Zero);
            return found;
        }
    }

    private sealed class LauncherHotkeyWindow : NativeWindow, IDisposable
    {
        private const int WM_HOTKEY = 0x0312;
        private readonly LauncherContext _context;
        private int _hotkeyId = 1;
        private bool _registered;

        public LauncherHotkeyWindow(LauncherContext context)
        {
            _context = context;
            CreateHandle(new CreateParams());
        }

        public void Register(string hotkey)
        {
            if (!TryParseHotkey(hotkey, out uint modifiers, out uint virtualKey))
            {
                return;
            }
            _registered = RegisterHotKey(Handle, _hotkeyId, modifiers, virtualKey);
        }

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == WM_HOTKEY && m.WParam.ToInt32() == _hotkeyId)
            {
                _context.ToggleConsole();
                return;
            }
            base.WndProc(ref m);
        }

        public void Dispose()
        {
            if (_registered)
            {
                UnregisterHotKey(Handle, _hotkeyId);
                _registered = false;
            }
            DestroyHandle();
        }

        private static bool TryParseHotkey(string hotkey, out uint modifiers, out uint virtualKey)
        {
            modifiers = 0;
            virtualKey = 0;
            if (string.IsNullOrWhiteSpace(hotkey))
            {
                return false;
            }

            string[] parts = hotkey.ToLowerInvariant().Split('+');
            foreach (string rawPart in parts)
            {
                string part = rawPart.Trim();
                if (part == "shift")
                {
                    modifiers |= MOD_SHIFT;
                    continue;
                }
                if (part == "ctrl" || part == "control")
                {
                    modifiers |= MOD_CONTROL;
                    continue;
                }
                if (part == "alt")
                {
                    modifiers |= MOD_ALT;
                    continue;
                }
                if (part == "win" || part == "windows")
                {
                    modifiers |= MOD_WIN;
                    continue;
                }
                if (part.StartsWith("f", StringComparison.Ordinal) && int.TryParse(part.Substring(1), out int fn) && fn >= 1 && fn <= 24)
                {
                    virtualKey = (uint)(0x70 + fn - 1);
                    continue;
                }
                if (part.Length == 1)
                {
                    virtualKey = (uint)char.ToUpperInvariant(part[0]);
                    continue;
                }
                if (part == "`")
                {
                    virtualKey = 0xC0;
                    continue;
                }
                return false;
            }

            return virtualKey != 0;
        }
    }

    private enum ShowWindowCommands
    {
        Hide = 0,
        Show = 5,
        Restore = 9,
    }

    private delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);

    private const uint MOD_ALT = 0x0001;
    private const uint MOD_CONTROL = 0x0002;
    private const uint MOD_SHIFT = 0x0004;
    private const uint MOD_WIN = 0x0008;
    private const int WM_CLOSE = 0x0010;

    [DllImport("user32.dll")]
    private static extern bool RegisterHotKey(IntPtr hWnd, int id, uint fsModifiers, uint vk);

    [DllImport("user32.dll")]
    private static extern bool UnregisterHotKey(IntPtr hWnd, int id);

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    private static extern int GetWindowTextLength(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool IsWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool ShowWindow(IntPtr hWnd, ShowWindowCommands nCmdShow);

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool PostMessage(IntPtr hWnd, int msg, IntPtr wParam, IntPtr lParam);
}
