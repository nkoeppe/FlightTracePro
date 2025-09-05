#!/usr/bin/env python3
"""
Test script for the improved update dialogs
"""
import sys
import tempfile
import os
from pathlib import Path

# Add client directory to path
client_dir = Path(__file__).parent / "client"
sys.path.insert(0, str(client_dir))

try:
    from PySide6.QtCore import Qt, QThread, Signal, Slot, QDateTime, QSettings, QTimer
    from PySide6.QtGui import QIcon, QAction
    from PySide6.QtWidgets import QApplication
    
    # Import our dialog classes
    from msfs_bridge_gui import UpdateDialog, DeferredUpdateCompletionDialog, DeferredUpdateRestartDialog
    
    def test_update_dialog():
        """Test the main update dialog"""
        print("Testing UpdateDialog...")
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        dialog = UpdateDialog("1.5.0", "https://github.com/test/test/releases/download/v1.5.0/FlightTracePro.exe")
        result = dialog.exec()
        
        print(f"User choice: {result}")
        print(f"INSTALL_NOW = {UpdateDialog.UpdateAction.INSTALL_NOW}")
        print(f"INSTALL_LATER = {UpdateDialog.UpdateAction.INSTALL_LATER}")
        print(f"CANCEL = {UpdateDialog.UpdateAction.CANCEL}")
        
        return result

    def test_deferred_completion_dialog():
        """Test the deferred update completion dialog"""
        print("\nTesting DeferredUpdateCompletionDialog...")
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        dialog = DeferredUpdateCompletionDialog("1.5.0")
        result = dialog.exec()
        print(f"Dialog completed: {result}")

    def test_deferred_restart_dialog():
        """Test the deferred update restart dialog"""
        print("\nTesting DeferredUpdateRestartDialog...")
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create a dummy batch file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as f:
            f.write('@echo off\necho Test batch file\npause\n')
            batch_file = f.name
        
        try:
            dialog = DeferredUpdateRestartDialog("1.5.0", batch_file)
            result = dialog.exec()
            
            print(f"User choice: {result}")
            print(f"RESTART_NOW = {DeferredUpdateRestartDialog.RestartAction.RESTART_NOW}")
            print(f"RESTART_LATER = {DeferredUpdateRestartDialog.RestartAction.RESTART_LATER}")
            
            return result
        finally:
            # Clean up temp file
            if os.path.exists(batch_file):
                os.remove(batch_file)

    def test_marker_file_creation():
        """Test creating and reading marker files"""
        print("\nTesting marker file functionality...")
        
        tmpdir = tempfile.gettempdir()
        marker_file = os.path.join(tmpdir, 'flighttracepro_test_deferred_update.marker')
        
        # Create test marker file
        with open(marker_file, 'w') as mf:
            mf.write("version=1.5.0\n")
            mf.write("batch_file=/tmp/test_update.bat\n")
            mf.write("log_file=/tmp/test_update.log\n")
            mf.write("new_exe=/tmp/FlightTracePro_new.exe\n")
        
        print(f"Created marker file: {marker_file}")
        
        # Read it back
        marker_data = {}
        with open(marker_file, 'r') as mf:
            for line in mf:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    marker_data[key] = value
        
        print("Marker data read back:")
        for key, value in marker_data.items():
            print(f"  {key}: {value}")
        
        # Clean up
        os.remove(marker_file)
        print("Marker file test completed successfully!")

    if __name__ == "__main__":
        print("FlightTracePro Update Dialog Test Suite")
        print("=" * 50)
        
        # Test marker file functionality (non-GUI)
        test_marker_file_creation()
        
        print("\nGUI tests require user interaction:")
        print("1. UpdateDialog - Choose install now/later/cancel")
        print("2. DeferredUpdateCompletionDialog - Click OK")
        print("3. DeferredUpdateRestartDialog - Choose restart now/later")
        
        response = input("\nRun GUI tests? (y/n): ").lower().strip()
        
        if response == 'y':
            print("\nStarting GUI tests...")
            
            # Test the main update dialog
            update_result = test_update_dialog()
            
            # If user chose install later, test the completion dialog
            if update_result == UpdateDialog.UpdateAction.INSTALL_LATER:
                test_deferred_completion_dialog()
            
            # Test the restart dialog
            test_deferred_restart_dialog()
            
        print("\nTest suite completed!")

except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires PySide6 to be installed.")
    print("Run: pip install PySide6")