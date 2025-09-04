# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlightTracePro is a real-time flight tracking system with GPX to KML conversion capabilities. The system consists of:

- **FastAPI server** (`server/`) - Web application with live map UI and GPX/KML conversion
- **Windows bridge client** (`client/`) - Streams MSFS 2020 flight data to the server via SimConnect
- **Static assets** (`static/`) - Icons, 3D models, and configuration files for the web interface

## Development Commands

### Local Development (Docker - Recommended)
```bash
# Development environment with hot-reload
docker compose -f docker-compose.dev.yml up --build

# Production environment
docker compose up --build -d
```

### Local Python Development
```bash
# Server setup
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Windows client dependencies (requires Windows)
pip install -r client/requirements.txt
```

### Windows Bridge Client
```bash
# CLI mode (minimal dependencies)
pip install SimConnect==0.4.26 websockets requests

# WebSocket mode (recommended)
python client/msfs_bridge.py --server ws://YOUR_HOST:8000 --channel default --callsign N123AB --mode ws

# HTTP mode (fallback)
python client/msfs_bridge.py --server http://YOUR_HOST:8000 --channel default --callsign N123AB --mode http

# Demo mode (no MSFS required)
python client/msfs_bridge.py --server ws://localhost:8000 --channel default --callsign TEST --mode ws --demo

# GUI mode
python client/msfs_bridge_gui.pyw
```

### Build Windows EXE
```bash
# From project root
pyinstaller --noconsole --name FlightTraceProBridge --onefile --icon NONE client/msfs_bridge_gui.pyw
```

## Architecture Overview

### Server Architecture (`server/app.py`)
- **FastAPI application** with WebSocket and HTTP endpoints
- **In-memory data structures** for live tracking:
  - `ChannelState` - manages WebSocket connections, viewers, and feeders per channel
  - `LiveSample` - Pydantic model for aircraft position/attitude data
  - `KML_STORE` - temporary storage for generated KML files
- **Real-time broadcasting** system using asyncio and WebSockets
- **Multi-channel support** - different flight sessions can use separate channels
- **Stale data pruning** - removes inactive aircraft after configurable TTL

### Client Architecture (`client/msfs_bridge.py`)
- **SimConnect integration** - reads flight data from MSFS 2020
- **Dual protocol support** - WebSocket (preferred) and HTTP modes
- **Demo mode** - simulated flight data for testing without MSFS
- **GUI wrapper** (`msfs_bridge_gui.pyw`) - PySide6 system tray application

### Key Data Flow
1. Windows client connects to MSFS via SimConnect
2. Client streams position data (lat/lon/alt/heading/etc.) to server
3. Server broadcasts to all connected viewers via WebSocket
4. Web UI displays real-time aircraft positions on 2D/3D maps

## Environment Variables

### Server
- `LIVE_POST_KEY` - Optional security key for feeder authentication
- `LIVE_TTL_SEC` - Stale data timeout in seconds (default: 60)
- `CESIUM_ION_TOKEN` - Enables Cesium Ion terrain and imagery for 3D globe
- `CESIUM_ION_TOKEN_FILE` - Alternative token source from Docker secrets

### Client
- Uses CLI arguments rather than environment variables
- Key arguments: `--server`, `--channel`, `--callsign`, `--mode`, `--key`

## API Endpoints

### Live Tracking
- `GET /` - Main web interface
- `WebSocket /ws/live/{channel}` - Real-time data streaming
- `POST /api/live/{channel}` - HTTP data ingestion
- `GET /api/live/{channel}/recent` - Current aircraft states
- `GET /api/live/{channel}/history` - Flight tracks since server start

### GPX/KML Conversion
- `POST /upload` - Convert GPX to KML with preview
- `GET /kml/{uuid}` - Download generated KML files

### Health
- `GET /healthz` - Service health check

## File Structure Notes

- Frontend code is embedded in `server/app.py` as `INDEX_HTML` constant
- 3D aircraft model is served from `static/models/glider/Glider.glb`
- Icons are in `static/icons/` (airplane.svg, aircraft.svg)
- Country boundaries data in `static/country-bboxes.json`

## Development Notes

- The development Docker setup mounts source files for live editing
- WebSocket connections use compression=None for proxy compatibility
- SimConnect values may be in radians - client handles degree conversion
- The server maintains flight history per callsign for trail visualization
- Multi-player support through real-time WebSocket broadcasting

## Testing and Quality Assurance

### Manual Testing Checklist
- [ ] GPX file upload and conversion functionality
- [ ] KML file download and Google Earth compatibility
- [ ] Live map 2D view with real-time updates
- [ ] Live map 3D globe rendering and navigation
- [ ] WebSocket connectivity and reconnection handling
- [ ] Multi-channel support and isolation
- [ ] Bridge client GUI system tray functionality
- [ ] Demo mode operation without MSFS

### Performance Benchmarks
- WebSocket latency: < 50ms for local connections
- GPX conversion: < 2 seconds for files up to 10MB
- Live map updates: 2Hz default, up to 10Hz supported
- Concurrent users: Tested up to 50 simultaneous viewers per channel

## Security Considerations

- Optional authentication via `LIVE_POST_KEY` environment variable
- No persistent data storage - all tracking data is in-memory only
- CORS enabled for cross-origin web interface access
- WebSocket connections support secure WSS protocol
- Docker secrets integration for production token management

## Deployment Configurations

### Development
- Docker Compose with volume mounts for hot-reload
- Direct Python execution for rapid iteration
- SQLite or in-memory data storage

### Production
- Multi-stage Docker builds for optimized images
- External network integration for reverse proxy setups
- Secret management for Cesium Ion tokens
- Health check endpoints for monitoring

## Monitoring and Observability

### Health Endpoints
- `GET /healthz` - Basic service health check
- WebSocket connection metrics via server logs
- Client connection status in bridge GUI

### Logging
- Structured logging in server application
- Bridge client debug modes for troubleshooting
- Docker container logs for deployment monitoring

## Future Enhancement Opportunities

### Planned Features
- Historical flight data persistence (database integration)
- Advanced flight analytics and reporting
- Integration with additional flight simulation platforms
- Mobile-responsive web interface improvements
- Real-time weather overlay integration

### SimConnect Data Expansion
- Engine performance monitoring (RPM, fuel flow, temperatures)
- Aircraft system status (gear, flaps, electrical, hydraulic)
- Environmental data integration (weather, wind conditions)
- Fuel consumption and range estimation
- Emergency and warning system alerts