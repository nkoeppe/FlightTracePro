# NavMap

A professional real-time flight tracking system with GPX to KML conversion capabilities, designed for Microsoft Flight Simulator 2020 integration and multi-user live tracking.

## Features

- **Real-time Flight Tracking**: Live aircraft positioning with 2D/3D visualization
- **GPX to KML Conversion**: Professional-grade file conversion with preview capabilities
- **Multi-user Support**: Real-time multiplayer tracking across channels
- **Cross-platform**: Web-based interface with Windows bridge client
- **Professional UI**: Modern Material Design-compliant interface

## Architecture

### Server Components
- **FastAPI Backend**: High-performance web server with WebSocket support
- **Live Tracking System**: Real-time data broadcasting with channel isolation
- **File Conversion Engine**: GPX to KML transformation with validation

### Client Components
- **Windows Bridge**: SimConnect integration for MSFS 2020 data streaming
- **Web Interface**: Responsive 2D/3D mapping with real-time updates
- **Multi-protocol Support**: WebSocket and HTTP fallback modes

## Quick Start

### Development Environment (Recommended)

```bash
# Prerequisites: Docker and Docker Compose
git clone <repository-url>
cd navmap

# Start development environment with hot-reload
docker compose -f docker-compose.dev.yml up --build

# Access the application
open http://localhost:8000
```

### Production Deployment

```bash
# Production deployment
docker compose up --build -d

# Ensure external network and secrets are configured
# - Network: proxy
# - Secret: ./secrets/cesium_ion_token
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LIVE_POST_KEY` | Authentication key for data feeders | None | No |
| `LIVE_TTL_SEC` | Stale data timeout in seconds | 60 | No |
| `CESIUM_ION_TOKEN` | Cesium Ion access token for 3D terrain | None | No |
| `CESIUM_ION_TOKEN_FILE` | Path to Cesium token file | None | No |

### Cesium Ion Integration (Optional)

For enhanced 3D terrain and imagery:

1. Obtain a [Cesium Ion](https://cesium.com/ion/) access token
2. Set `CESIUM_ION_TOKEN` environment variable
3. Select "Real" terrain mode in the 3D interface

## Installation

### Local Python Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install server dependencies
pip install -r server/requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Windows Bridge Client

#### Command Line Interface

```bash
# Install minimal dependencies
pip install SimConnect==0.4.26 websockets requests

# WebSocket mode (recommended)
python client/msfs_bridge.py --server ws://YOUR_HOST:8000 --channel default --callsign N123AB --mode ws

# HTTP mode (fallback)
python client/msfs_bridge.py --server http://YOUR_HOST:8000 --channel default --callsign N123AB --mode http

# Demo mode (no MSFS required)
python client/msfs_bridge.py --server ws://localhost:8000 --channel default --callsign TEST --mode ws --demo
```

#### Graphical Interface

```bash
# Install full dependencies
pip install -r client/requirements.txt

# Launch GUI application
python client/msfs_bridge_gui.pyw

# Build standalone executable
pyinstaller --noconsole --name NavMapBridge --onefile --icon NONE client/msfs_bridge_gui.pyw
```

## API Reference

### Live Tracking Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| WebSocket | `/ws/live/{channel}` | Real-time data streaming |
| POST | `/api/live/{channel}` | HTTP data ingestion |
| GET | `/api/live/{channel}/recent` | Current aircraft states |
| GET | `/api/live/{channel}/history` | Flight track history |
| GET | `/healthz` | Service health check |

### Data Format

```json
{
  "lat": 47.6062,
  "lon": -122.3321,
  "alt_m": 1500,
  "spd_kt": 120,
  "vsi_ms": 2.5,
  "hdg_deg": 270,
  "pitch_deg": 5,
  "roll_deg": -2,
  "callsign": "N123AB",
  "aircraft": "Cessna 172",
  "ts": "2024-01-01T12:00:00Z"
}
```

## Usage

### GPX Conversion
1. Navigate to the main interface
2. Select "Converter" tab
3. Upload GPX file
4. Preview in 2D/3D view
5. Download generated KML

### Live Tracking
1. Start Windows bridge client with MSFS 2020
2. Navigate to "Live Map" tab
3. Select 2D Map or 3D Globe view
4. Monitor real-time aircraft positions
5. Click aircraft markers for detailed information

### Multi-user Features
- **Channels**: Separate flight sessions using different channel names
- **Real-time Updates**: Automatic position broadcasting to all viewers
- **Player Management**: Join/leave notifications and presence tracking
- **Trail Visualization**: Historical flight paths with telemetry data

## Development

### Project Structure
```
navmap/
├── server/          # FastAPI application
│   ├── app.py      # Main application file
│   └── requirements.txt
├── client/          # Windows bridge client
│   ├── msfs_bridge.py
│   ├── msfs_bridge_gui.pyw
│   └── requirements.txt
├── static/          # Web assets
│   ├── icons/      # UI icons
│   ├── models/     # 3D models
│   └── country-bboxes.json
└── docker-compose*.yml
```

### Technology Stack
- **Backend**: FastAPI, WebSockets, asyncio
- **Frontend**: Embedded HTML/CSS/JS with Material Design
- **3D Rendering**: Cesium.js
- **Windows Integration**: SimConnect API
- **Containerization**: Docker & Docker Compose

## Troubleshooting

### Common Issues

**WebSocket Disconnections**
- Ensure proxy compatibility with WebSocket compression
- Use HTTP mode as fallback for testing

**3D Terrain Not Loading**
- Verify `CESIUM_ION_TOKEN` is properly configured
- Select "Real" terrain mode in 3D interface

**Incorrect Heading Display**
- Check SimConnect radian/degree conversion
- Enable DEBUG logging in GUI for raw data inspection

**Connection Issues**
- Verify firewall settings allow traffic on port 8000
- Check network connectivity between client and server

## Support

For issues and feature requests, please refer to the project documentation or contact the development team.

## License

[Specify license here]
