# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Get the current directory (client folder). In spec files, use SPECPATH.
client_dir = Path(globals().get('SPECPATH', '.'))

data_files = collect_data_files('certifi')
# Include VERSION file for version checking
data_files.append((str(client_dir / 'VERSION'), '.'))

# Ensure Python runtime DLLs are included
python_dll_binaries = []
if hasattr(sys, 'base_exec_prefix'):
    python_dll_path = os.path.join(sys.base_exec_prefix, f'python{sys.version_info.major}{sys.version_info.minor}.dll')
    if os.path.exists(python_dll_path):
        python_dll_binaries.append((python_dll_path, '.'))

a = Analysis(
    [str(client_dir / 'msfs_bridge_gui.pyw')],
    pathex=[str(client_dir)],
    binaries=python_dll_binaries,
    datas=data_files,
    hiddenimports=[
        'SimConnect',
        'websockets',
        'websockets.client',
        'websockets.exceptions',
        'requests',
        'certifi',
        'PySide6.QtCore',
        'PySide6.QtWidgets', 
        'PySide6.QtGui',
        'json',
        'asyncio',
        'threading',
        'queue',
        'time',
        'sys',
        'os',
        'logging',
        'argparse',
        'random',
        'math',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'PIL',
        'tkinter',
        'PyQt5',
        'PyQt6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate entries and optimize, but keep Python DLLs
a.binaries = [x for x in a.binaries if not x[0].lower().startswith('api-ms-win')]
a.binaries = [x for x in a.binaries if not x[0].lower().startswith('ucrtbase')]
# Ensure Python DLL is preserved (check by filename in binary name)
python_dll_name = f'python{sys.version_info.major}{sys.version_info.minor}.dll'
has_python_dll = any(python_dll_name.lower() in x[0].lower() for x in a.binaries)
if not has_python_dll:
    print(f"Adding Python DLL: {python_dll_name}")
    for dll_path, target_dir in python_dll_binaries:
        a.binaries.append((os.path.basename(dll_path), dll_path, 'BINARY'))

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FlightTracePro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    # Remove runtime_tmpdir to use default temp extraction - fixes DLL issues with batch restart
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    # Optimize for Windows 10+
    manifest="""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity
    version="1.0.0.0"
    processorArchitecture="*"
    name="FlightTracePro"
    type="win32"
  />
  <description>FlightTracePro Bridge Client</description>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity
        type="win32"
        name="Microsoft.Windows.Common-Controls"
        version="6.0.0.0"
        processorArchitecture="*"
        publicKeyToken="6595b64144ccf1df"
        language="*"
      />
    </dependentAssembly>
  </dependency>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
    </application>
  </compatibility>
</assembly>
""",
)
