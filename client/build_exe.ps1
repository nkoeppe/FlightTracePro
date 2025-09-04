Param(
  [string]$Python = "python",
  [switch]$RebuildVenv
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
& .\.venv\Scripts\pyinstaller.exe --clean --noconsole --onefile --name NavMapBridge client\msfs_bridge_gui.pyw --specpath client

Write-Host "[bridge] Built dist\NavMapBridge.exe" -ForegroundColor Green
