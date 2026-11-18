# Build AutoResearch.exe (the Auto Research Windows desktop app) in one command.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File launcher\build.ps1
#
# Wraps the existing webapp server unchanged: webapp/ and Resources/ are added
# as data files, the pywebview shell is collected, and dist\AutoResearch\
# holds the onedir build. The script also zips the folder for distribution.
#
# Requirements (launcher\requirements-packaging.txt): pywebview, pyinstaller,
# pillow — installed into the system Python. webapp/ itself stays stdlib-only.

$ErrorActionPreference = "Stop"
$dev = Split-Path $PSScriptRoot -Parent
$python = "C:\Program Files\Python313\python.exe"
$ico = Join-Path $PSScriptRoot "app.ico"
$iconset = Join-Path $dev "Resources\AppIcon.iconset\icon_256x256.png"

Push-Location $dev
try {
    Write-Output "Dev copy: $dev"

    # The icon: generate from the repo's iconset when missing (pillow).
    if (-not (Test-Path $ico)) {
        Write-Output "Generating launcher\app.ico from Resources\AppIcon.iconset ..."
        & $python -c "import sys; from PIL import Image; source = Image.open(sys.argv[1]).convert('RGBA'); source.save(sys.argv[2], format='ICO', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])" $iconset $ico
        if ($LASTEXITCODE -ne 0) { throw "Icon generation failed." }
    }

    # Verify the window shell imports before packaging (Python 3.13 needs a
    # pywebview version that supports it).
    & $python -c "import webview; print('pywebview import ok')"
    if ($LASTEXITCODE -ne 0) { throw "pywebview is not importable; pip install -r launcher\requirements-packaging.txt" }

    # Reproducible onedir build from a clean dev copy.
    & $python -m PyInstaller launcher\launcher.py `
        --name AutoResearch `
        --windowed `
        --icon launcher\app.ico `
        --noconfirm `
        --clean `
        --add-data "$dev\webapp;webapp" `
        --add-data "$dev\Resources;Resources" `
        --collect-all webview
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

    $exe = Join-Path $dev "dist\AutoResearch\AutoResearch.exe"
    if (-not (Test-Path $exe)) { throw "Build output is missing: $exe" }
    $size = [int]((Get-Item $exe).Length / 1KB)
    Write-Output ("Built {0} ({1} KiB)" -f $exe, $size)

    $warnFile = Join-Path $dev "build\AutoResearch\warn-AutoResearch.txt"
    if (Test-Path $warnFile) {
        $missing = Select-String -Path $warnFile -Pattern "no module named ([a-z0-9_.]+)" -AllMatches |
            ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object -Unique
        Write-Output ("PyInstaller warnings: {0} unresolved module names" -f ($missing | Measure-Object).Count)
        if ($missing) { Write-Output ("  not bundled (expected for gui-only backends): " + (($missing | Select-Object -First 8) -join ", ")) }
    }

    # Distribution zip.
    $zip = Join-Path $dev "dist\AutoResearch-windows-x64.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path (Join-Path $dev "dist\AutoResearch") -DestinationPath $zip
    $zipSize = [int]((Get-Item $zip).Length / 1MB)
    Write-Output ("Zipped {0} ({1} MB)" -f $zip, $zipSize)
    Write-Output "Done."
} finally {
    Pop-Location
}
