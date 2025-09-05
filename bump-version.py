#!/usr/bin/env python3
"""
Simple version bumping script for FlightTracePro
Usage: python bump-version.py [patch|minor|major]
"""
import sys
import os
from pathlib import Path

def parse_version(version_str):
    """Parse version string like '0.2.0' into [0, 2, 0]"""
    return [int(x) for x in version_str.split('.')]

def format_version(version_list):
    """Format version list [0, 2, 0] back to '0.2.0'"""
    return '.'.join(str(x) for x in version_list)

def bump_version(current_version, bump_type):
    """Bump version according to semver rules"""
    v = parse_version(current_version)
    
    if bump_type == 'major':
        v[0] += 1
        v[1] = 0
        v[2] = 0
    elif bump_type == 'minor':
        v[1] += 1
        v[2] = 0
    elif bump_type == 'patch':
        v[2] += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    return format_version(v)

def main():
    if len(sys.argv) != 2:
        print("Usage: python bump-version.py [minor|major]")
        print("Current usage:")
        print("  minor: 0.2.0 -> 0.3.0 (new features)")
        print("  major: 0.2.0 -> 1.0.0 (breaking changes)")
        print("")
        print("Note: Patch versions are auto-generated from GitHub run numbers.")
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    if bump_type not in ['minor', 'major']:
        print(f"Error: Invalid bump type '{bump_type}'")
        print("Valid options: minor, major")
        print("(patch versions are auto-generated from build numbers)")
        sys.exit(1)
    
    # Read current version
    version_file = Path('client/VERSION')
    if not version_file.exists():
        print("Error: client/VERSION file not found!")
        sys.exit(1)
    
    current_version = version_file.read_text().strip()
    print(f"Current version: {current_version}")
    
    # Bump version
    try:
        new_version = bump_version(current_version, bump_type)
        print(f"New version: {new_version}")
        
        # Confirm with user
        response = input(f"Update version from {current_version} to {new_version}? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            sys.exit(0)
        
        # Write new version
        version_file.write_text(new_version)
        print(f"✅ Version updated to {new_version}")
        print("\nNext steps:")
        print(f"1. Commit the changes: git add client/VERSION && git commit -m 'Bump version to v{new_version}'")
        print("2. Push to trigger release: git push")
        print(f"\nNote: The actual release will be {new_version}.X where X is the GitHub run number")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()