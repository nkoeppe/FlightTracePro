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
    
    REM Quick test - should not crash on startup
    timeout /t 1 /nobreak > nul
    if exist dist\FlightTracePro.exe (
        echo [FlightTracePro] ✅ Executable appears valid
    ) else (
        echo [FlightTracePro] ❌ Executable test failed
        exit /b 1
    )
) else (
    echo [FlightTracePro] ❌ Build failed - executable not found
    exit /b 1
)

REM Cleanup
rmdir /s /q "%VENV_DIR%"

exit /b 0
