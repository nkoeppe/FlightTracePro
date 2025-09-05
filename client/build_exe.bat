@echo off
setlocal enabledelayedexpansion

echo [FlightTracePro] Building Windows executable...

set PYTHON=python
if not "%1"=="" set PYTHON=%1

REM Create virtual environment in temp location to avoid path issues
set VENV_DIR=%LOCALAPPDATA%\FlightTraceProBuild\.venv
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
%PYTHON% -m venv --copies "%VENV_DIR%"

REM Install dependencies
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\pip.exe" install -r client\requirements.txt

REM Clean previous builds
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM Build using spec file for better DLL handling
echo [FlightTracePro] Running PyInstaller with spec file...
"%VENV_DIR%\Scripts\pyinstaller.exe" --clean client\FlightTracePro.spec

if exist dist\FlightTracePro.exe (
    echo [FlightTracePro] ✅ Built dist\FlightTracePro.exe successfully!
    echo [FlightTracePro] Testing executable...
    
    REM Test version flag - simulates update process startup
    echo [FlightTracePro] Testing --version flag (simulates update restart)...
    dist\FlightTracePro.exe --version
    if !errorlevel! equ 0 (
        echo [FlightTracePro] ✅ Version test passed - DLL issues should be fixed
    ) else (
        echo [FlightTracePro] ❌ Version test failed - possible DLL issue
        echo [FlightTracePro] This may cause problems during auto-update restart
    )
) else (
    echo [FlightTracePro] ❌ Build failed - executable not found
    exit /b 1
)

REM Cleanup
rmdir /s /q "%VENV_DIR%"

exit /b 0
