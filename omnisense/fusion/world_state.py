"""
Unified world state representation for OMNISENSE platform.

Maintains a consistent 3D model of the environment integrating data from
all cameras and perception modules.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from filterpy.kalman import ExtendedKalmanFilter
import time

from omnisense.tracking.tracker import Track
from omnisense.utils.logger import get_logger
from omnisense.utils.geometry import triangulate_multi_view

logger = get_logger(__name__)


@dataclass
class Person3D:
    """3D representation of a tracked person."""
    global_id: int
    position_3d: np.ndarray  # (x, y, z) in global frame
    velocity_3d: np.ndarray  # (vx, vy, vz)
    confidence: float
    last_seen: float  # timestamp
    cameras_visible: List[int]  # Camera IDs where person is visible
    trajectory_3d: List[Tuple[float, np.ndarray]]  # (timestamp, position) history
    head_pose: Optional[Dict] = None
    body_pose: Optional[Dict] = None
    gaze_direction: Optional[np.ndarray] = None
    attention_state: Optional[str] = None
    activity: Optional[str] = None


@dataclass
class WorldState:
    """Complete world state representation."""
    persons: Dict[int, Person3D] = field(default_factory=dict)
    occupancy_grid: Optional[np.ndarray] = None
    map_points: Optional[np.ndarray] = None
    camera_poses: Dict[int, Dict] = field(default_factory=dict)
    zones: Dict[str, Dict] = field(default_factory=dict)
    timestamp: float = 0.0


class SensorFusion:
    """
    Fuses multi-camera observations into unified 3D world state.

    Performs triangulation, Kalman filtering, and state estimation.
    """

    def __init__(self, camera_calibrations: Dict = None):
        """
        Initialize sensor fusion.

        Args:
            camera_calibrations: Dictionary of camera calibration parameters
        """
        self.camera_calibrations = camera_calibrations or {}

        # World state
        self.world_state = WorldState()

        # Kalman filters for each person
        self.person_filters: Dict[int, ExtendedKalmanFilter] = {}

        logger.info("SensorFusion initialized")

    def update(
        self,
        tracks_per_camera: Dict[int, List[Track]],
        timestamp: Optional[float] = None
    ) -> WorldState:
        """
        Update world state from multi-camera tracks.

        Args:
            tracks_per_camera: Dictionary mapping camera_id to list of tracks
            timestamp: Current timestamp

        Returns:
            Updated WorldState
        """
        if timestamp is None:
            timestamp = time.time()

        self.world_state.timestamp = timestamp

        # Group tracks by global ID
        tracks_by_global_id = self._group_tracks_by_global_id(tracks_per_camera)

        # Update each person's 3D position
        for global_id, camera_tracks in tracks_by_global_id.items():
            person_3d = self._estimate_person_3d(global_id, camera_tracks, timestamp)

            if person_3d is not None:
                self.world_state.persons[global_id] = person_3d

        # Remove stale persons
        current_time = time.time()
        stale_ids = [
            pid for pid, person in self.world_state.persons.items()
            if current_time - person.last_seen > 5.0  # 5 second timeout
        ]

        for pid in stale_ids:
            del self.world_state.persons[pid]
            if pid in self.person_filters:
                del self.person_filters[pid]

        return self.world_state

    def _group_tracks_by_global_id(
        self,
        tracks_per_camera: Dict[int, List[Track]]
    ) -> Dict[int, Dict[int, Track]]:
        """
        Group tracks by their global ID.

        Returns:
            Dictionary mapping global_id to {camera_id: track}
        """
        grouped = {}

        for camera_id, tracks in tracks_per_camera.items():
            for track in tracks:
                if track.global_id is not None:
                    if track.global_id not in grouped:
                        grouped[track.global_id] = {}
                    grouped[track.global_id][camera_id] = track

        return grouped

    def _estimate_person_3d(
        self,
        global_id: int,
        camera_tracks: Dict[int, Track],
        timestamp: float
    ) -> Optional[Person3D]:
        """
        Estimate 3D position of a person from multi-view tracks.

        Args:
            global_id: Global person ID
            camera_tracks: Dictionary of camera_id -> track
            timestamp: Current timestamp

        Returns:
            Person3D object or None
        """
        if len(camera_tracks) < 1:
            return None

        # For simplicity, use centroid of available camera observations
        # In production, use proper triangulation with camera calibration
        positions_2d = []
        camera_ids = []

        for camera_id, track in camera_tracks.items():
            # Get bounding box center
            x1, y1, x2, y2 = track.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            positions_2d.append((center_x, center_y))
            camera_ids.append(camera_id)

        # Simplified 3D position estimation
        # In production: use triangulation with camera matrices
        if len(positions_2d) >= 2:
            # Estimate depth from bbox size (very simplified)
            avg_height = np.mean([track.bbox[3] - track.bbox[1]
                                 for track in camera_tracks.values()])
            estimated_depth = 1000.0 / (avg_height + 1e-6)  # Inverse relationship

            # Use first camera's view for X, Y
            x_2d, y_2d = positions_2d[0]
            position_3d = np.array([x_2d / 100, y_2d / 100, estimated_depth])
        else:
            # Single camera - estimate rough 3D position
            x_2d, y_2d = positions_2d[0]
            position_3d = np.array([x_2d / 100, y_2d / 100, 2.0])

        # Initialize or update Kalman filter
        if global_id not in self.person_filters:
            self._initialize_person_filter(global_id, position_3d)

        # Update filter
        velocity_3d = self._update_person_filter(global_id, position_3d)

        # Calculate average confidence
        avg_confidence = np.mean([track.confidence for track in camera_tracks.values()])

        # Create or update Person3D
        if global_id in self.world_state.persons:
            person = self.world_state.persons[global_id]
            person.position_3d = position_3d
            person.velocity_3d = velocity_3d
            person.confidence = avg_confidence
            person.last_seen = timestamp
            person.cameras_visible = camera_ids
            person.trajectory_3d.append((timestamp, position_3d.copy()))

            # Keep trajectory limited
            if len(person.trajectory_3d) > 100:
                person.trajectory_3d = person.trajectory_3d[-100:]

        else:
            person = Person3D(
                global_id=global_id,
                position_3d=position_3d,
                velocity_3d=velocity_3d,
                confidence=avg_confidence,
                last_seen=timestamp,
                cameras_visible=camera_ids,
                trajectory_3d=[(timestamp, position_3d.copy())]
            )

        return person

    def _initialize_person_filter(self, person_id: int, initial_position: np.ndarray):
        """Initialize Kalman filter for a person."""
        # Simple constant velocity model
        # State: [x, y, z, vx, vy, vz]
        kf = ExtendedKalmanFilter(dim_x=6, dim_z=3)

        # Initial state
        kf.x = np.hstack([initial_position, np.zeros(3)])

        # State transition matrix (constant velocity)
        dt = 1.0 / 30.0  # Assuming 30 FPS
        kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix (observe position only)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Covariances
        kf.R = np.eye(3) * 0.1  # Measurement noise
        kf.Q = np.eye(6) * 0.01  # Process noise
        kf.P = np.eye(6) * 1.0  # Initial uncertainty

        self.person_filters[person_id] = kf

    def _update_person_filter(
        self,
        person_id: int,
        measurement: np.ndarray
    ) -> np.ndarray:
        """
        Update Kalman filter with new measurement.

        Returns:
            Estimated velocity
        """
        kf = self.person_filters[person_id]

        # Predict
        kf.predict()

        # Update
        kf.update(measurement)

        # Extract velocity estimate
        velocity = kf.x[3:6]

        return velocity

    def get_world_state(self) -> WorldState:
        """Get current world state."""
        return self.world_state

    def get_person(self, global_id: int) -> Optional[Person3D]:
        """Get specific person by global ID."""
        return self.world_state.persons.get(global_id)

    def get_all_persons(self) -> List[Person3D]:
        """Get all tracked persons."""
        return list(self.world_state.persons.values())

    def get_person_count(self) -> int:
        """Get current number of tracked persons."""
        return len(self.world_state.persons)
