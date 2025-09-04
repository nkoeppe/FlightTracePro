#!/usr/bin/env python3
"""
SimConnect Diagnostic Tool
Tests SimConnect installation and connection
"""
import sys
import os

def main():
    print("=== SimConnect Diagnostic Tool ===")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Current Directory: {os.getcwd()}")
    print()
    
    # Test 1: Basic Python imports
    print("1. Testing basic imports...")
    try:
        import time, json, asyncio
        print("   ✓ Basic imports OK")
    except Exception as e:
        print(f"   ✗ Basic imports failed: {e}")
        return
    
    # Test 2: Check if SimConnect package is available
    print("2. Testing SimConnect package...")
    try:
        import SimConnect
        print("   ✓ SimConnect package found")
        print(f"   SimConnect location: {SimConnect.__file__}")
        try:
            print(f"   SimConnect version: {SimConnect.__version__}")
        except:
            print("   SimConnect version: unknown")
    except ImportError as e:
        print(f"   ✗ SimConnect package not found: {e}")
        print("   Install with: pip install SimConnect==0.4.26")
        return
    except Exception as e:
        print(f"   ✗ SimConnect package error: {e}")
        return
    
    # Test 3: Test SimConnect import specifics
    print("3. Testing SimConnect classes...")
    try:
        from SimConnect import SimConnect as SC, AircraftRequests
        print("   ✓ SimConnect and AircraftRequests imported")
    except Exception as e:
        print(f"   ✗ SimConnect classes import failed: {e}")
        return
    
    # Test 3.5: Check if MSFS is running
    print("3.5 Checking if MSFS 2020 is running...")
    try:
        import subprocess
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq FlightSimulator.exe'], 
                              capture_output=True, text=True, shell=True)
        if 'FlightSimulator.exe' in result.stdout:
            print("   ✓ Flight Simulator process found")
        else:
            print("   ⚠ Flight Simulator process NOT found")
            print("   Please start MSFS 2020 first!")
    except Exception as e:
        print(f"   Could not check MSFS process: {e}")
    
    # Test 4: Try to create SimConnect instance with different methods
    print("4. Testing SimConnect connection...")
    
    connection_methods = [
        ("Default connection", lambda: SC()),
        ("Auto-connect False", lambda: SC(auto_connect=False)),  
        ("Auto-connect True", lambda: SC(auto_connect=True)),
    ]
    
    sim = None
    for method_name, connect_func in connection_methods:
        try:
            print(f"   Trying {method_name}...")
            sim = connect_func()
            print(f"   ✓ {method_name} successful!")
            break
        except Exception as e:
            print(f"   ✗ {method_name} failed: {e}")
            continue
    
    if sim:
        # Test 5: Try AircraftRequests
        print("5. Testing AircraftRequests...")
        try:
            areq = AircraftRequests(sim, _time=50)
            print("   ✓ AircraftRequests created successfully!")
            
            # Test 6: Try to read some data
            print("6. Testing data reading...")
            try:
                lat = areq.get("PLANE_LATITUDE")
                print(f"   ✓ Data read successful! Latitude: {lat}")
                print("   🎉 SimConnect is working perfectly!")
            except Exception as e:
                print(f"   ⚠ Data read failed (expected if not in flight): {e}")
                print("   ✓ SimConnect connection is working!")
            
        except Exception as e:
            print(f"   ✗ AircraftRequests failed: {e}")
        
        # Clean up
        try:
            sim.quit()
        except:
            pass
    else:
        print("   ✗ All SimConnect connection methods failed!")
        print("   Troubleshooting checklist:")
        print("   1. Is MSFS 2020 running and in a flight?")
        print("   2. Check MSFS Options > General > Developers > Enable SimConnect")
        print("   3. Try restarting MSFS completely") 
        print("   4. Run this as Administrator")
        print("   5. Check Windows Firewall settings")
    
    print()
    print("=== Diagnostic Complete ===")

if __name__ == "__main__":
    main()