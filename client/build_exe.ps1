Param(
  [string]$Python = "python",
  [string]$BuildRoot = $env:LOCALAPPDATA + "\FlightTraceProBuild",
  [switch]$RebuildVenv,
  [switch]$Debug
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "[bridge] Packaging FlightTracePro GUI..." -ForegroundColor Cyan

if ($RebuildVenv) {
  if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
}

# Use a local (trusted) build root for venv and outputs to avoid UNC execution issues
if (-not (Test-Path $BuildRoot)) { New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null }
$VenvPath = Join-Path $BuildRoot ".venv"
if ($RebuildVenv -and (Test-Path $VenvPath)) { Remove-Item -Recurse -Force $VenvPath }
if (-not (Test-Path $VenvPath)) {
  & $Python -m venv --copies $VenvPath
}

$pip = Join-Path $VenvPath "Scripts\pip.exe"
$pyi = Join-Path $VenvPath "Scripts\pyinstaller.exe"

& $pip install --upgrade pip
& $pip install -r (Join-Path $PSScriptRoot "requirements.txt")

$argsList = @('--clean')
if ($Debug) { $argsList += '--console' } else { $argsList += '--noconsole' }
$DistPath = Join-Path $BuildRoot 'dist'
$WorkPath = Join-Path $BuildRoot 'build'
$SpecPath = Join-Path $BuildRoot 'spec'
New-Item -ItemType Directory -Force -Path $DistPath,$WorkPath,$SpecPath | Out-Null

$argsList += @(
  '--onefile','--name','FlightTracePro',
  '--hidden-import','SimConnect',
  '--collect-all','SimConnect',
  '--collect-submodules','websockets',
  '--collect-submodules','requests',
  '--collect-data','certifi',
  (Join-Path $PSScriptRoot 'msfs_bridge_gui.pyw'),
  '--distpath', $DistPath,
  '--workpath', $WorkPath,
  '--specpath', $SpecPath
)

# Optionally bundle SimConnect.dll if present (local builds only)
$vendorDll = Join-Path $PSScriptRoot 'vendor\SimConnect\x64\SimConnect.dll'
if (Test-Path $vendorDll) {
  Write-Host "[bridge] Bundling vendor SimConnect.dll: $vendorDll" -ForegroundColor Yellow
  $argsList += @('--add-binary', "$vendorDll;.")
} elseif ($env:FLIGHTTRACEPRO_SIMCONNECT_DLL_DIR) {
  $cand = Join-Path $env:FLIGHTTRACEPRO_SIMCONNECT_DLL_DIR 'SimConnect.dll'
  if (Test-Path $cand) {
    Write-Host "[bridge] Bundling SimConnect.dll from env dir: $cand" -ForegroundColor Yellow
    $argsList += @('--add-binary', "$cand;.")
  }
}

& $pyi @argsList

Write-Host "[bridge] Built $DistPath\FlightTracePro.exe" -ForegroundColor Green
