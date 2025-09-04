# FlightTracePro Windows Bridge Client

A high-performance Windows bridge application that streams real-time flight data from Microsoft Flight Simulator 2020 to the FlightTracePro server via SimConnect API.

## Features

- **Real-time Data Streaming**: Low-latency flight data transmission via WebSocket or HTTP
- **SimConnect Integration**: Direct interface with MSFS 2020 simulation engine
- **Dual Protocol Support**: WebSocket (recommended) and HTTP fallback modes
- **GUI & CLI Interfaces**: System tray application or command-line operation
- **Demo Mode**: Test functionality without running MSFS 2020
- **Secure Authentication**: Optional API key support for protected servers

## Quick Start

### Prerequisites
- **Operating System**: Windows 10/11
- **Python**: Version 3.10 or higher
- **Microsoft Flight Simulator 2020**: Running instance required (except demo mode)

### Installation

#### Method 1: Minimal Dependencies (CLI Only)
```bash
pip install SimConnect==0.4.26 websockets requests
```

#### Method 2: Full Installation (GUI + CLI)
```bash
pip install -r client/requirements.txt
```

> **Note**: PySide6 is a large package; initial installation may take several minutes on Windows.

## Usage

### Command Line Interface

#### WebSocket Mode (Recommended)
```bash
python msfs_bridge.py --server ws://YOUR_HOST:8000 --channel default --callsign N123AB --mode ws
```

#### HTTP Mode (Fallback)
```bash
python msfs_bridge.py --server http://YOUR_HOST:8000 --channel default --callsign N123AB --mode http
```

#### Demo Mode (No MSFS Required)
```bash
python msfs_bridge.py --server ws://localhost:8000 --channel default --callsign TEST --mode ws --demo
```

### Graphical User Interface
```bash
python msfs_bridge_gui.pyw
```

The GUI application runs in the system tray. Right-click the tray icon to access:
- Server configuration
- Connection status
- Demo mode toggle
- Update rate settings
- Log viewer

## Configuration

### Command Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--server` | Server URL (ws:// or http://) | Required | `ws://localhost:8000` |
| `--channel` | Channel name for multi-session support | `default` | `flight-training` |
| `--callsign` | Aircraft identifier | `N123AB` | `VH-ABC` |
| `--mode` | Protocol mode | `ws` | `ws` or `http` |
| `--key` | Optional authentication key | None | `secure-key-123` |
| `--rate` | Update frequency in Hz | `2.0` | `10.0` |
| `--demo` | Enable simulation mode | `false` | - |

### Security Configuration

For secured server deployments:

1. **Server Setup**: Set `LIVE_POST_KEY` environment variable
2. **Client Setup**: Use `--key YOUR_KEY` argument

Example:
```bash
python msfs_bridge.py --server ws://secure-server:8000 --key secret-api-key --callsign N123AB --mode ws
```

## Performance Optimization

### Update Rates
- **Standard**: 2 Hz (every 0.5 seconds) - Default setting
- **High Performance**: 10 Hz (every 0.1 seconds) - For high-precision tracking
- **Low Bandwidth**: 0.5 Hz (every 2 seconds) - For limited network conditions

### Network Protocols
- **WebSocket**: Lower latency, persistent connection, automatic reconnection
- **HTTP**: Simpler setup, higher overhead, suitable for proxy environments

## Troubleshooting

### Common Issues

**SimConnect Installation**
```bash
# If SimConnect package fails to install
pip install SimConnect==0.4.26 --force-reinstall
```

**MSFS 2020 Connection**
- Ensure Microsoft Flight Simulator 2020 is running and active
- Verify SimConnect is enabled in MSFS settings
- Check Windows Defender/Firewall permissions

**Network Connectivity**
- Verify server URL and port accessibility
- Test connection with demo mode first
- Check firewall settings on both client and server

**Performance Issues**
- Reduce update rate for slower systems: `--rate 1`
- Use HTTP mode if WebSocket connections are unstable
- Monitor system resources during operation

### Debug Logging

Enable verbose logging by modifying the script or using the GUI debug mode:
```bash
python msfs_bridge.py --server ws://localhost:8000 --callsign DEBUG --mode ws --rate 1
```

## System Requirements

### Minimum Requirements
- Windows 10/11 (64-bit)
- Python 3.10+
- 100 MB available RAM
- Network connectivity to FlightTracePro server

### Recommended Requirements
- Windows 11 (64-bit)
- Python 3.11+
- 500 MB available RAM
- Stable broadband internet connection
- MSFS 2020 with SimConnect enabled

## Integration Notes

- **Little Navmap**: Not required - FlightTracePro Bridge connects directly to MSFS 2020
- **SimConnect API**: Uses official Microsoft SimConnect interface
- **Multi-Instance**: Multiple bridge clients can connect to different channels
- **Persistence**: GUI version stores recent server configurations via Windows Registry
