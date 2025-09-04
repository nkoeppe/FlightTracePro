# FlightTracePro Bridge - Windows Executable Packaging

This guide provides comprehensive instructions for building a standalone Windows executable of the FlightTracePro Bridge GUI application using PyInstaller.

## Overview

The packaging process creates a single, self-contained `.exe` file that includes:
- Python runtime environment
- All required dependencies (PySide6, SimConnect, websockets, etc.)
- Application resources and assets
- No external Python installation required on target systems

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11 (64-bit)
- **Python**: Version 3.10 or higher
- **Build Tools**: Visual Studio Build Tools (for some dependencies)
- **Disk Space**: ~500MB for build environment and dependencies

### Development Environment Setup

1. **Install Python Dependencies**
   ```bash
   pip install -r client/requirements.txt
   ```

2. **Install PyInstaller**
   ```bash
   pip install pyinstaller
   ```

## Build Methods

### Method 1: Automated PowerShell Build (Recommended)

#### Release Build (Production)
```powershell
./client/build_exe.ps1
```
- Creates optimized executable without console window
- Suitable for end-user distribution
- Minimal file size and startup time

#### Debug Build (Development)
```powershell
./client/build_exe.ps1 -Debug
```
- Includes console window for debugging output
- Shows import errors and SimConnect diagnostics
- Useful for troubleshooting build issues

### Method 2: Batch File Build (Windows CMD)
```cmd
client\build_exe.bat
```
- Alternative build script for Command Prompt users
- Equivalent to PowerShell release build

### Method 3: Manual PyInstaller Command

For advanced users or custom build configurations:

```bash
# Ensure virtual environment is activated
.venv\Scripts\activate

# Install PyInstaller in virtual environment
.venv\Scripts\pip.exe install pyinstaller

# Build executable with complete dependency collection
.venv\Scripts\pyinstaller.exe --clean --noconsole --onefile --name FlightTracePro ^
  --hidden-import SimConnect ^
  --collect-all SimConnect ^
  --collect-submodules websockets ^
  --collect-submodules requests ^
  --collect-data certifi ^
  --add-data "static;static" ^
  client\msfs_bridge_gui.pyw --specpath client
```

## Build Output

### Generated Files
- **Executable**: `dist/FlightTracePro.exe` (primary distribution file)
- **Spec File**: `client/FlightTracePro.spec` (build configuration)
- **Build Cache**: `build/` directory (temporary files, can be deleted)

### File Size and Performance
- **Executable Size**: ~50-80MB (includes Python runtime and all dependencies)
- **Startup Time**: 2-5 seconds on modern systems
- **Memory Usage**: ~100-200MB during operation

## Deployment

### Distribution Checklist
- [ ] Test executable on clean Windows system (without Python installed)
- [ ] Verify SimConnect functionality with MSFS 2020
- [ ] Test network connectivity and server communication
- [ ] Validate system tray functionality and GUI responsiveness
- [ ] Check Windows Defender/antivirus compatibility

### Installation Instructions for End Users

1. **Download**: Obtain `FlightTracePro.exe` from distribution source
2. **Security**: Windows may show security warning for unsigned executable
   - Click "More info" → "Run anyway" if from trusted source
3. **First Run**: Application will appear in system tray (look for airplane icon)
4. **Configuration**: Right-click tray icon to configure server settings

## Advanced Configuration

### PyInstaller Spec File Customization

The generated `.spec` file can be modified for advanced build scenarios:

```python
# Example spec file modifications
a = Analysis(['client/msfs_bridge_gui.pyw'],
             pathex=[],
             binaries=[],
             datas=[('static', 'static')],  # Include static assets
             hiddenimports=['SimConnect', 'PySide6.QtCore'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=['matplotlib', 'numpy'],  # Exclude unnecessary packages
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=None,
             noarchive=False)
```

### Build Optimization

#### Reduce Executable Size
```bash
# Exclude unused modules
--exclude-module matplotlib
--exclude-module numpy
--exclude-module tkinter

# Use UPX compression (requires UPX.exe in PATH)
--upx-dir /path/to/upx
```

#### Improve Startup Performance
```bash
# Create directory-based distribution (faster startup)
--onedir

# Enable binary caching
--distpath ./dist-cache
```

## Troubleshooting

### Common Build Issues

**SimConnect Import Errors**
```bash
# Ensure SimConnect is properly installed
pip install SimConnect==0.4.26 --force-reinstall
```

**Missing Dependencies**
```bash
# Rebuild with verbose output
pyinstaller --log-level DEBUG client/msfs_bridge_gui.pyw
```

**Runtime Errors**
- Use debug build to see console output
- Check Windows Event Viewer for application errors
- Verify all required DLLs are included

### Testing and Validation

#### Pre-Distribution Testing
1. **Clean Environment**: Test on VM or system without development tools
2. **MSFS Integration**: Verify SimConnect communication works correctly
3. **Network Connectivity**: Test with various server configurations
4. **System Tray**: Ensure application starts and tray icon appears
5. **GUI Functionality**: Test all dialog boxes and configuration options

#### Performance Verification
```bash
# Monitor resource usage during operation
# Task Manager → Details → FlightTracePro.exe
# Watch CPU, Memory, and Network usage
```

## Security Considerations

### Code Signing (Optional)
For professional distribution, consider code signing:
```bash
# Using SignTool (requires certificate)
signtool sign /f certificate.p12 /p password /t http://timestamp.server dist/FlightTracePro.exe
```

### Antivirus Compatibility
- Test with major antivirus solutions
- Consider submitting to Microsoft Defender for analysis
- Document any false positive reports

## Continuous Integration

### Automated Build Pipeline
```yaml
# Example GitHub Actions workflow
name: Build Windows Executable
on: [push, pull_request]
jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r client/requirements.txt pyinstaller
      - name: Build executable
        run: ./client/build_exe.ps1
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: FlightTracePro-Windows
          path: dist/FlightTracePro.exe
```

## Support and Maintenance

### Version Management
- Update version strings in spec file and application code
- Include version information in executable properties
- Maintain changelog for distribution updates

### User Support
- Provide clear installation instructions
- Document system requirements and compatibility
- Offer troubleshooting guides for common issues
