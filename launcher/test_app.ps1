# Smoke test for the built AutoResearch.exe (PowerShell 5.1+).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File launcher\test_app.ps1
#
# Stops any manual webapp.server process, launches dist\AutoResearch\
# AutoResearch.exe twice, and asserts the dashboard acceptance criteria over
# HTTP: healthz, the token guard, the user's workspace (parity with Runtime
# state.json), the live catalog status, a chart feed, the window title, the
# tailnet binding, a clean shutdown (exit 0, port freed) and that a relaunch
# after a hard kill still passes (no stale single-instance mutex).
#
# At the end no server is left running; restart the manual server the way it
# was started if you need one.

$ErrorActionPreference = "Stop"
$dev = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $dev "dist\AutoResearch\AutoResearch.exe"
if (-not (Test-Path $exe)) { throw "Build first with launcher\build.ps1 (missing $exe)" }

$config = Get-Content (Join-Path $env:APPDATA "AutoResearch\config.json") -Raw | ConvertFrom-Json
$port = [int]$config.port
$token = $config.token
$workspaceRoot = $config.workspaceRoot
if (Test-Path (Join-Path $workspaceRoot "Runtime")) { $workspaceRoot = Join-Path $workspaceRoot "Runtime" }
$stateFile = Join-Path $workspaceRoot "state.json"
$tailnetAddress = "100.75.183.120"
$expectedCatalog = "10 repositories $([char]0xB7) 5 built-in projects"

$script:failures = 0
$script:passes = 0

function Assert([string]$name, $condition, [string]$detail) {
    if ($condition) {
        $script:passes++
        Write-Output ("PASS  {0}  {1}" -f $name, $detail)
    } else {
        $script:failures++
        Write-Output ("FAIL  {0}  {1}" -f $name, $detail)
    }
}

function Invoke-Status([string]$url) {
    # (status, body) without failing on 4xx/5xx (Invoke-WebRequest 5.1 throws).
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 25
        return @{ Status = [int]$response.StatusCode; Body = $response.Content }
    } catch {
        if ($_.Exception.Response) {
            return @{ Status = [int]$_.Exception.Response.StatusCode; Body = "" }
        }
        return @{ Status = 0; Body = $_.Exception.Message }
    }
}

function Test-PortFree([int]$p) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $null = $client.Connect("127.0.0.1", $p)
        return $false
    } catch {
        return $true
    } finally {
        $client.Close()
    }
}

function Wait-ForServer([int]$p) {
    # Polls /healthz until the dashboard answers; true when it did.
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        $health = Invoke-Status "http://127.0.0.1:$p/healthz"
        if ($health.Status -eq 200) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

Write-Output "== AutoResearch.exe smoke test =="

# --- first-run shortcut choice: seed it, restore the user's config at the end -
# The exe asks once (a MessageBox), then records shortcutChoiceMade and
# desktopShortcut in %APPDATA%\AutoResearch\config.json. Seed "made, no
# shortcut" so the smoke launches never prompt, and restore the user's config
# in the finally block below. The seed is written with Python (JSON, UTF-8
# without BOM): ConvertTo-Json 5.1 mangles strings and Out-File adds a BOM.
$exeConfig = Join-Path $env:APPDATA "AutoResearch\config.json"
$exeConfigBackup = "$exeConfig.smoke.bak"
Copy-Item $exeConfig $exeConfigBackup -Force
$seedPy = @'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
data["shortcutChoiceMade"] = True
data["desktopShortcut"] = False
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
'@
$seedFile = Join-Path ([System.IO.Path]::GetTempPath()) "autoresearch-smoke-seed.py"
[IO.File]::WriteAllText($seedFile, $seedPy, [Text.UTF8Encoding]::new($false))
try {
    python -B $seedFile $exeConfig
    if ($LASTEXITCODE -ne 0) { throw "Could not seed the shortcut choice in the smoke config." }

# --- stop the manual python server so the exe can bind the port ---------------
# 'webapp.server' matches the manual server's 'webapp\server.py' command line.
$manual = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -match "webapp.server" }
foreach ($process in $manual) {
    Write-Output ("Stopping the manual server (pid {0})" -f $process.ProcessId)
    Stop-Process -Id $process.ProcessId -Force
}
$deadline = (Get-Date).AddSeconds(10)
$freed = $false
while ((Get-Date) -lt $deadline -and -not $freed) {
    $freed = Test-PortFree $port
    if (-not $freed) { Start-Sleep -Milliseconds 500 }
}
if (-not $freed) {
    throw "Port $port is still busy after stopping every webapp.server process; the assertions below would test the wrong server."
}
Write-Output ("PASS  port {0} free after stopping webapp.server" -f $port)

# --- first launch --------------------------------------------------------------
$started = Get-Date
$first = Start-Process -FilePath $exe -PassThru
$up = Wait-ForServer $port $token
$startupSeconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
Assert "launch 1: /healthz 200" $up ("startup {0}s, pid {1}" -f $startupSeconds, $first.Id)

$guard = Invoke-Status "http://127.0.0.1:$port/api/state"
Assert "untokenized /api/state -> 401" ($guard.Status -eq 401) ("got {0}" -f $guard.Status)

$state = Invoke-Status "http://127.0.0.1:$port/api/state?token=$token"
Assert "tokenized /api/state -> 200" ($state.Status -eq 200) ("got {0}" -f $state.Status)
$dashboard = $state.Body | ConvertFrom-Json
$saved = Get-Content $stateFile -Raw | ConvertFrom-Json
Assert "workspace parity: projects" ($dashboard.projects.Count -eq $saved.projects.Count) ("{0} projects" -f $dashboard.projects.Count)
Assert "workspace parity: sessions" ($dashboard.sessions.Count -eq $saved.sessions.Count) ("{0} sessions" -f $dashboard.sessions.Count)
Assert "catalog status live" ($dashboard.server.catalogStatus -eq $expectedCatalog) $dashboard.server.catalogStatus
Assert "server access is lan" ($dashboard.server.access -eq "lan") ("workspace " + $dashboard.server.workspaceRoot)
Assert "provider stays opencode" ($dashboard.provider -eq "opencode") $dashboard.provider

$tracked = $dashboard.projects | Where-Object { $_.id -notlike "builtin:*" }
$chart = $null
foreach ($project in $tracked) {
    $query = [uri]::EscapeDataString($project.id)
    $feed = Invoke-Status "http://127.0.0.1:$port/api/dashboard?projectID=$query&token=$token"
    if ($feed.Status -ne 200) { continue }
    $feedJson = $feed.Body | ConvertFrom-Json
    if ($feedJson.scope -and $feedJson.scope.repo) {
        $chart = @{ json = $feedJson; id = $project.id }
        break
    }
}
if ($chart) {
    Assert "chart feed returns scope/groups/pulls" ($chart.json.scope.repo -eq $chart.id -and $chart.json.groups.Count -ge 1 -and $chart.json.pulls.Count -ge 1) ("{0}: {1} groups, {2} pulls" -f $chart.id, $chart.json.groups.Count, $chart.json.pulls.Count)
} else {
    Assert "chart feed returns scope/groups/pulls" $false "no tracked project answered"
}

$title = ""
$deadline = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $deadline -and -not $title) {
    $running = Get-Process -Id $first.Id -ErrorAction SilentlyContinue
    if ($running -and $running.MainWindowTitle) { $title = $running.MainWindowTitle }
    Start-Sleep -Milliseconds 250
}
Assert "MainWindowTitle == 'Auto Research'" ($title -eq "Auto Research") "got '$title'"

$lan = Invoke-Status "http://${tailnetAddress}:$port/healthz"
Assert "healthz 200 over the tailnet ($tailnetAddress)" ($lan.Status -eq 200) ("got {0}" -f $lan.Status)

# --- hard kill, port freed, relaunch ------------------------------------------
Stop-Process -Id $first.Id -Force
$deadline = (Get-Date).AddSeconds(15)
$freed = $false
while ((Get-Date) -lt $deadline -and -not $freed) {
    $freed = Test-PortFree $port
    if (-not $freed) { Start-Sleep -Milliseconds 500 }
}
Assert "port $port freed after kill" $freed "relaunching"

$second = Start-Process -FilePath $exe -PassThru
$up = Wait-ForServer $port $token
Assert "launch 2: /healthz 200 (no stale mutex)" $up "relaunch after a hard kill passed"
$guard2 = Invoke-Status "http://127.0.0.1:$port/api/state"
Assert "launch 2: untokenized /api/state -> 401" ($guard2.Status -eq 401) ("got {0}" -f $guard2.Status)
$title2 = ""
$deadline = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $deadline -and -not $title2) {
    $running = Get-Process -Id $second.Id -ErrorAction SilentlyContinue
    if ($running -and $running.MainWindowTitle) { $title2 = $running.MainWindowTitle }
    Start-Sleep -Milliseconds 250
}
Assert "launch 2: MainWindowTitle == 'Auto Research'" ($title2 -eq "Auto Research") "got '$title2'"

# --- graceful window close: server stops cleanly, process exits 0 -------------
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Win32Close {
    [DllImport("user32.dll")]
    public static extern bool PostMessage(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
}
"@
$handle = (Get-Process -Id $second.Id).MainWindowHandle
$null = [Win32Close]::PostMessage($handle, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)  # WM_CLOSE
$exited = $second.WaitForExit(20000)
Assert "window close exits 0" ($exited -and $second.ExitCode -eq 0) ("exit code " + $(if ($exited) { $second.ExitCode } else { "still running" }))
$deadline = (Get-Date).AddSeconds(15)
$freed = $false
while ((Get-Date) -lt $deadline -and -not $freed) {
    $freed = Test-PortFree $port
    if (-not $freed) { Start-Sleep -Milliseconds 500 }
}
Assert "port $port freed after the window closed" $freed "server stopped cleanly"

$after = Get-Content $exeConfig -Raw | ConvertFrom-Json
Assert "shortcut choice survives the launches" ($after.shortcutChoiceMade -eq $true -and $after.desktopShortcut -eq $false) ("the launcher neither re-prompted nor rewrote the choice")
}
finally {
    Copy-Item $exeConfigBackup $exeConfig -Force
    Remove-Item $exeConfigBackup -Force -ErrorAction SilentlyContinue
    Remove-Item $seedFile -Force -ErrorAction SilentlyContinue
    Write-Output "user config restored from the smoke backup"
}

Write-Output ("== {0} passed, {1} failed ==" -f $script:passes, $script:failures)
if ($script:failures -gt 0) { exit 1 }
exit 0
