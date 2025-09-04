Param(
  [string]$Python = "python",
  [switch]$RebuildVenv,
  [switch]$Debug
)

$ErrorActionPreference = "Stop"

Write-Host "[bridge] Packaging NavMap Bridge GUI..." -ForegroundColor Cyan

if ($RebuildVenv) {
  if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
}

if (-not (Test-Path .venv)) {
  & $Python -m venv .venv
}

& .\.venv\Scripts\pip.exe install --upgrade pip
& .\.venv\Scripts\pip.exe install -r client\requirements.txt

# Build
$console = "--noconsole"
if ($Debug) { $console = "--console" }

& .\.venv\Scripts\pyinstaller.exe --clean $console --onefile --name NavMapBridge \
  --hidden-import SimConnect \
  --collect-all SimConnect \
  --collect-submodules websockets \
  --collect-submodules requests \
  --collect-data certifi \
  client\msfs_bridge_gui.pyw --specpath client

Write-Host "[bridge] Built dist\NavMapBridge.exe" -ForegroundColor Green
