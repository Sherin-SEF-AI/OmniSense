"""
FastAPI REST API for OMNISENSE platform.

Provides HTTP endpoints for querying system state, tracked persons,
activities, and system configuration.
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import json

from omnisense.utils.logger import get_logger

logger = get_logger(__name__)


# Pydantic models for request/response validation
class Person3DResponse(BaseModel):
    """Response model for a tracked person."""
    global_id: int
    position_3d: List[float] = Field(..., description="[x, y, z] position in meters")
    velocity_3d: List[float] = Field(..., description="[vx, vy, vz] velocity in m/s")
    confidence: float = Field(..., ge=0.0, le=1.0)
    cameras_visible: List[int]
    last_seen: float
    head_pose: Optional[Dict[str, float]] = None
    attention_state: Optional[str] = None
    activity: Optional[str] = None


class WorldStateResponse(BaseModel):
    """Response model for complete world state."""
    timestamp: float
    person_count: int
    persons: List[Person3DResponse]


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    running: bool
    uptime_seconds: float
    cameras: Dict[int, Dict[str, Any]]
    sync_fps: Optional[float]
    person_count: int
    gpu_available: bool
    model_info: Optional[Dict[str, Any]]


class AlertResponse(BaseModel):
    """Response model for alerts."""
    alert_id: int
    timestamp: float
    alert_type: str
    severity: str
    message: str
    person_id: Optional[int]
    camera_id: Optional[int]
    acknowledged: bool


class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates."""
    parameter_path: str = Field(..., description="Dot-notation path like 'tracking.detection_confidence'")
    value: Any


class OmniSenseAPI:
    """
    REST API for OMNISENSE platform.

    Provides endpoints for querying and controlling the system.
    """

    def __init__(self, app_instance):
        """
        Initialize API.

        Args:
            app_instance: OMNISENSE application instance
        """
        self.app_instance = app_instance
        self.fastapi_app = FastAPI(
            title="OMNISENSE API",
            description="Multi-Camera Spatial Intelligence Platform API",
            version="1.0.0"
        )

        # Add CORS middleware
        self.fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # WebSocket connections
        self.active_websockets: List[WebSocket] = []

        # Setup routes
        self._setup_routes()

        logger.info("OmniSense API initialized")

    def _setup_routes(self):
        """Setup all API routes."""

        @self.fastapi_app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "OMNISENSE API",
                "version": "1.0.0",
                "status": "operational",
                "endpoints": {
                    "world_state": "/api/v1/world-state",
                    "persons": "/api/v1/persons",
                    "status": "/api/v1/status",
                    "alerts": "/api/v1/alerts",
                    "config": "/api/v1/config",
                    "websocket": "ws://host:port/ws"
                }
            }

        @self.fastapi_app.get("/api/v1/world-state", response_model=WorldStateResponse)
        async def get_world_state():
            """Get current world state with all tracked persons."""
            if not self.app_instance.sensor_fusion:
                raise HTTPException(status_code=503, detail="Sensor fusion not initialized")

            world_state = self.app_instance.sensor_fusion.get_world_state()

            persons = []
            for person in world_state.persons.values():
                persons.append(Person3DResponse(
                    global_id=person.global_id,
                    position_3d=person.position_3d.tolist(),
                    velocity_3d=person.velocity_3d.tolist(),
                    confidence=person.confidence,
                    cameras_visible=person.cameras_visible,
                    last_seen=person.last_seen,
                    head_pose=person.head_pose,
                    attention_state=person.attention_state,
                    activity=person.activity
                ))

            return WorldStateResponse(
                timestamp=world_state.timestamp,
                person_count=len(persons),
                persons=persons
            )

        @self.fastapi_app.get("/api/v1/persons/{person_id}", response_model=Person3DResponse)
        async def get_person(person_id: int):
            """Get specific person by global ID."""
            if not self.app_instance.sensor_fusion:
                raise HTTPException(status_code=503, detail="Sensor fusion not initialized")

            person = self.app_instance.sensor_fusion.get_person(person_id)

            if person is None:
                raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

            return Person3DResponse(
                global_id=person.global_id,
                position_3d=person.position_3d.tolist(),
                velocity_3d=person.velocity_3d.tolist(),
                confidence=person.confidence,
                cameras_visible=person.cameras_visible,
                last_seen=person.last_seen,
                head_pose=person.head_pose,
                attention_state=person.attention_state,
                activity=person.activity
            )

        @self.fastapi_app.get("/api/v1/persons", response_model=List[Person3DResponse])
        async def list_persons():
            """List all currently tracked persons."""
            if not self.app_instance.sensor_fusion:
                raise HTTPException(status_code=503, detail="Sensor fusion not initialized")

            persons = []
            for person in self.app_instance.sensor_fusion.get_all_persons():
                persons.append(Person3DResponse(
                    global_id=person.global_id,
                    position_3d=person.position_3d.tolist(),
                    velocity_3d=person.velocity_3d.tolist(),
                    confidence=person.confidence,
                    cameras_visible=person.cameras_visible,
                    last_seen=person.last_seen,
                    head_pose=person.head_pose,
                    attention_state=person.attention_state,
                    activity=person.activity
                ))

            return persons

        @self.fastapi_app.get("/api/v1/status", response_model=SystemStatusResponse)
        async def get_status():
            """Get system status."""
            import time
            status = self.app_instance.get_status()

            # Calculate uptime
            uptime = time.time() - self.app_instance.start_time if hasattr(self.app_instance, 'start_time') else 0

            return SystemStatusResponse(
                running=status.get('running', False),
                uptime_seconds=uptime,
                cameras=status.get('cameras', {}),
                sync_fps=status.get('sync_fps'),
                person_count=status.get('person_count', 0),
                gpu_available=self.app_instance.config.performance.gpu_enabled,
                model_info=self.app_instance.model_manager.get_model_info() if self.app_instance.model_manager else None
            )

        @self.fastapi_app.get("/api/v1/cameras")
        async def get_cameras():
            """Get camera status."""
            if not self.app_instance.camera_manager:
                raise HTTPException(status_code=503, detail="Camera manager not initialized")

            return self.app_instance.camera_manager.get_status()

        @self.fastapi_app.get("/api/v1/config")
        async def get_config():
            """Get current configuration."""
            from dataclasses import asdict
            return asdict(self.app_instance.config)

        @self.fastapi_app.post("/api/v1/config")
        async def update_config(request: ConfigUpdateRequest):
            """Update configuration parameter."""
            try:
                self.app_instance.config_manager.set(request.parameter_path, request.value)
                return {"status": "success", "parameter": request.parameter_path, "value": request.value}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.fastapi_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_websockets.append(websocket)
            logger.info(f"WebSocket client connected. Total: {len(self.active_websockets)}")

            try:
                # Send initial state
                if self.app_instance.sensor_fusion:
                    world_state = self.app_instance.sensor_fusion.get_world_state()
                    await websocket.send_json({
                        "type": "world_state",
                        "data": {
                            "timestamp": world_state.timestamp,
                            "person_count": len(world_state.persons)
                        }
                    })

                # Keep connection alive and send updates
                while True:
                    # Wait for messages or send periodic updates
                    try:
                        # Receive with timeout
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                        # Handle client messages if needed
                    except asyncio.TimeoutError:
                        # Send periodic update
                        if self.app_instance.sensor_fusion:
                            world_state = self.app_instance.sensor_fusion.get_world_state()
                            await websocket.send_json({
                                "type": "update",
                                "timestamp": world_state.timestamp,
                                "person_count": len(world_state.persons)
                            })

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if websocket in self.active_websockets:
                    self.active_websockets.remove(websocket)
                logger.info(f"WebSocket client removed. Total: {len(self.active_websockets)}")

    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients."""
        if not self.active_websockets:
            return

        message = json.dumps(data)
        disconnected = []

        for websocket in self.active_websockets:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")
                disconnected.append(websocket)

        # Remove disconnected websockets
        for ws in disconnected:
            self.active_websockets.remove(ws)

    def get_app(self) -> FastAPI:
        """Get FastAPI application instance."""
        return self.fastapi_app
