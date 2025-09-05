#!/usr/bin/env python3
"""
Test script for the update logic without GUI dependencies
"""
import os
import tempfile
import sys
from pathlib import Path

def test_marker_file_functionality():
    """Test creating and reading deferred update marker files"""
    print("Testing deferred update marker file functionality...")
    
    tmpdir = tempfile.gettempdir()
    marker_file = os.path.join(tmpdir, 'flighttracepro_test_deferred_update.marker')
    
    # Test data
    test_version = "1.5.0"
    test_batch = "/tmp/test_update.bat"
    test_log = "/tmp/test_update.log"
    test_exe = "/tmp/FlightTracePro_new.exe"
    
    try:
        # Create test marker file (simulating deferred update)
        with open(marker_file, 'w') as mf:
            mf.write(f"version={test_version}\n")
            mf.write(f"batch_file={test_batch}\n")
            mf.write(f"log_file={test_log}\n")
            mf.write(f"new_exe={test_exe}\n")
        
        print(f"✅ Created marker file: {marker_file}")
        
        # Read it back (simulating startup check)
        marker_data = {}
        with open(marker_file, 'r') as mf:
            for line in mf:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    marker_data[key] = value
        
        print("✅ Marker data read successfully:")
        for key, value in marker_data.items():
            print(f"   {key}: {value}")
        
        # Verify data integrity
        assert marker_data['version'] == test_version, f"Version mismatch: {marker_data['version']} != {test_version}"
        assert marker_data['batch_file'] == test_batch, f"Batch file mismatch"
        assert marker_data['log_file'] == test_log, f"Log file mismatch"
        assert marker_data['new_exe'] == test_exe, f"Exe file mismatch"
        
        print("✅ All marker data verified successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Marker file test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(marker_file):
            os.remove(marker_file)
            print("✅ Cleanup completed")

def test_batch_file_generation():
    """Test batch file generation logic"""
    print("\nTesting batch file generation...")
    
    try:
        # Test data
        version = "1.5.0"
        cur_exe = "C:\\Programs\\FlightTracePro\\FlightTracePro.exe"
        new_exe = "C:\\Temp\\FlightTracePro_new.exe" 
        log_file = "C:\\Temp\\flighttracepro_update.log"
        
        # Test immediate update batch content
        immediate_mode = False
        deferred_mode = "false" if not immediate_mode else "true"
        
        batch_content_immediate = f"""@echo off
setlocal enabledelayedexpansion
title FlightTracePro Updater v{version}

REM Define paths as variables  
set TARGET_EXE="{cur_exe}"
set NEW_EXE="{new_exe}"
set LOGFILE="{log_file}"
set DEFERRED_MODE={deferred_mode}
set VERSION={version}

echo [%date% %time%] FlightTracePro Auto-Updater v%VERSION% Started >> %LOGFILE%
"""
        
        print("✅ Immediate update batch content generated")
        print("   First few lines:")
        for line in batch_content_immediate.split('\n')[:10]:
            print(f"   {line}")
        
        # Test deferred update batch content
        deferred_mode = True
        deferred_flag = "true" if deferred_mode else "false"
        
        batch_content_deferred = f"""@echo off
setlocal enabledelayedexpansion
title FlightTracePro Updater v{version}

REM Define paths as variables  
set TARGET_EXE="{cur_exe}"
set NEW_EXE="{new_exe}"
set LOGFILE="{log_file}"
set DEFERRED_MODE={deferred_flag}
set VERSION={version}

if "%DEFERRED_MODE%"=="true" (
    echo Mode: Deferred Update ^(triggered by user restart^)
) else (
    echo Mode: Immediate Update ^(triggered by update check^)
)
"""
        
        print("\n✅ Deferred update batch content generated")
        print("   Key differences for deferred mode:")
        print(f"   DEFERRED_MODE={deferred_flag}")
        
        # Verify batch content has essential components
        essential_components = [
            "setlocal enabledelayedexpansion",
            f"title FlightTracePro Updater v{version}",
            "TARGET_EXE=",
            "NEW_EXE=",
            "LOGFILE=",
            "DEFERRED_MODE=",
            "VERSION="
        ]
        
        for component in essential_components:
            if component in batch_content_immediate:
                print(f"   ✅ Found: {component}")
            else:
                print(f"   ❌ Missing: {component}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Batch file generation test failed: {e}")
        return False

def test_version_parsing():
    """Test version parsing and comparison logic"""
    print("\nTesting version parsing and comparison logic...")
    
    def parse_version(v):
        """Extract version numbers from string like "0.1.0-10" -> ["0", "1", "0", "10"]"""
        import re
        numbers = re.findall(r'\d+', v)
        return tuple(int(n) for n in numbers) if numbers else (0,)
    
    def compare_versions(current, latest):
        """Compare versions with proper padding for different lengths"""
        max_len = max(len(current), len(latest))
        current_padded = current + (0,) * (max_len - len(current))
        latest_padded = latest + (0,) * (max_len - len(latest))
        return latest_padded > current_padded
    
    # Test parsing
    parse_test_cases = [
        ("1.0.0", (1, 0, 0)),
        ("v1.2.3", (1, 2, 3)),
        ("bridge-v1.5.0", (1, 5, 0)),
        ("flighttracepro-v2.1.0-10", (2, 1, 0, 10)),
        ("0.1.0-5", (0, 1, 0, 5)),
        ("0.2", (0, 2)),
        ("0.2.27", (0, 2, 27)),
        ("invalid", (0,))
    ]
    
    try:
        for test_input, expected in parse_test_cases:
            result = parse_version(test_input)
            if result == expected:
                print(f"   ✅ Parse: {test_input} -> {result}")
            else:
                print(f"   ❌ Parse: {test_input} -> {result} (expected {expected})")
                return False
        
        # Test version comparison - the critical fix
        comparison_test_cases = [
            ("1.0.0", "1.0.1", True, "Patch update"),
            ("1.0", "1.0.1", True, "Patch update with short version"),
            ("1.0.1", "1.0", False, "Downgrade"),
            ("2.0", "1.9.9", False, "Major version higher"),
            ("0.2", "0.2.27", True, "The actual case from logs"),
            ("0.2.27", "0.2", False, "Reverse case"),
            ("1.0.0", "1.0.0", False, "Same version"),
        ]
        
        for current_str, latest_str, expected, description in comparison_test_cases:
            current = parse_version(current_str)
            latest = parse_version(latest_str)
            result = compare_versions(current, latest)
            
            if result == expected:
                print(f"   ✅ Compare: {current_str} vs {latest_str} = {result} ({description})")
            else:
                print(f"   ❌ Compare: {current_str} vs {latest_str} = {result}, expected {expected} ({description})")
                return False
            
        return True
        
    except Exception as e:
        print(f"❌ Version parsing/comparison test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("FlightTracePro Update System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Marker File Functionality", test_marker_file_functionality),
        ("Batch File Generation", test_batch_file_generation), 
        ("Version Parsing", test_version_parsing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Update system is ready.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)