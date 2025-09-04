@echo off
setlocal

set PYTHON=python
if not "%1"=="" set PYTHON=%1

if not exist .venv (
  %PYTHON% -m venv .venv
)

.venv\Scripts\pip.exe install --upgrade pip
.venv\Scripts\pip.exe install -r client\requirements.txt

.venv\Scripts\pyinstaller.exe --clean --noconsole --onefile --name FlightTracePro client\msfs_bridge_gui.pyw --specpath client

echo Built dist\FlightTracePro.exe
exit /b 0
