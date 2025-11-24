"""
Attention Analysis Module for OMNISENSE platform.

Analyzes visual attention, gaze patterns, and focus states for tracked persons.
Generates attention heatmaps and detects distraction events.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from collections import deque
import time

from omnisense.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GazePoint:
    """3D gaze intersection point."""
    position_3d: np.ndarray  # [x, y, z] in world coordinates
    confidence: float
    timestamp: float
    person_id: int


@dataclass
class AttentionTarget:
    """Attention target object or region."""
    target_id: str
    name: str
    position_3d: np.ndarray  # Center position
    bbox_3d: Optional[np.ndarray] = None  # [min_x, min_y, min_z, max_x, max_y, max_z]
    radius: float = 1.0  # Radius for point targets


@dataclass
class AttentionState:
    """Person's attention state."""
    person_id: int
    timestamp: float

    # Gaze information
    gaze_direction_3d: Optional[np.ndarray] = None
    gaze_target: Optional[str] = None  # Target ID
    gaze_confidence: float = 0.0

    # Focus metrics
    is_focused: bool = False
    focus_duration: float = 0.0  # Seconds
    distraction_level: float = 0.0  # 0.0 (focused) to 1.0 (distracted)

    # Attention patterns
    gaze_stability: float = 0.0  # 0.0 (unstable) to 1.0 (stable)
    saccade_rate: float = 0.0  # Saccades per second
    fixation_count: int = 0

    # Zone attention
    current_zone: Optional[str] = None
    zone_dwell_time: float = 0.0


class AttentionAnalyzer:
    """
    Analyzes visual attention patterns for tracked persons.

    Features:
    - Gaze tracking and 3D intersection
    - Attention target detection
    - Focus/distraction classification
    - Attention heatmap generation
    - Zone-based attention analysis
    """

    # Attention thresholds
    FOCUS_GAZE_STABILITY_THRESHOLD = 0.7
    FOCUS_MIN_DURATION = 2.0  # Seconds
    DISTRACTION_SACCADE_RATE = 3.0  # Saccades per second

    # Gaze parameters
    GAZE_HISTORY_SECONDS = 10.0
    FIXATION_THRESHOLD = 5.0  # Degrees
    SACCADE_THRESHOLD = 20.0  # Degrees

    def __init__(self):
        """Initialize attention analyzer."""
        self.attention_targets: Dict[str, AttentionTarget] = {}
        self.attention_zones: Dict[str, Dict] = {}

        # Person attention states
        self.person_states: Dict[int, AttentionState] = {}

        # Gaze history for each person
        self.gaze_history: Dict[int, deque] = {}

        # Heatmap data
        self.heatmap_resolution = (100, 100, 50)  # X, Y, Z bins
        self.heatmap_bounds = np.array([[-10, 10], [-10, 10], [0, 5]])  # World bounds
        self.attention_heatmap = np.zeros(self.heatmap_resolution)

        logger.info("Attention analyzer initialized")

    def add_target(self, target: AttentionTarget):
        """
        Add attention target.

        Args:
            target: Attention target to add
        """
        self.attention_targets[target.target_id] = target
        logger.info(f"Added attention target: {target.name} ({target.target_id})")

    def remove_target(self, target_id: str):
        """Remove attention target."""
        if target_id in self.attention_targets:
            del self.attention_targets[target_id]
            logger.info(f"Removed attention target: {target_id}")

    def define_zone(self, zone_id: str, zone_name: str, bounds_3d: np.ndarray):
        """
        Define attention zone.

        Args:
            zone_id: Zone identifier
            zone_name: Human-readable name
            bounds_3d: Zone bounds [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        self.attention_zones[zone_id] = {
            'name': zone_name,
            'bounds': bounds_3d,
            'attention_time': 0.0,
            'visit_count': 0
        }
        logger.info(f"Defined attention zone: {zone_name} ({zone_id})")

    def update(self, person_id: int, head_pose: Dict, position_3d: np.ndarray,
               timestamp: Optional[float] = None) -> AttentionState:
        """
        Update attention analysis for person.

        Args:
            person_id: Person ID
            head_pose: Head pose dict with yaw, pitch, roll
            position_3d: Person's 3D position
            timestamp: Timestamp (uses current time if None)

        Returns:
            Updated attention state
        """
        if timestamp is None:
            timestamp = time.time()

        # Initialize gaze history if needed
        if person_id not in self.gaze_history:
            self.gaze_history[person_id] = deque(maxlen=int(self.GAZE_HISTORY_SECONDS * 30))  # Assume 30 FPS

        # Compute gaze direction from head pose
        gaze_direction = self._head_pose_to_gaze_direction(head_pose)

        # Find gaze target
        gaze_target, gaze_point, gaze_confidence = self._find_gaze_target(
            position_3d, gaze_direction
        )

        # Store gaze point in history
        if gaze_point is not None:
            self.gaze_history[person_id].append(GazePoint(
                position_3d=gaze_point,
                confidence=gaze_confidence,
                timestamp=timestamp,
                person_id=person_id
            ))

            # Update heatmap
            self._update_heatmap(gaze_point)

        # Analyze gaze patterns
        gaze_stability = self._compute_gaze_stability(person_id)
        saccade_rate = self._compute_saccade_rate(person_id)
        fixation_count = self._count_fixations(person_id)

        # Determine focus state
        is_focused = self._is_focused(gaze_stability, saccade_rate, fixation_count)
        distraction_level = self._compute_distraction_level(gaze_stability, saccade_rate)

        # Compute focus duration
        focus_duration = self._compute_focus_duration(person_id, is_focused, timestamp)

        # Determine current zone
        current_zone = self._find_current_zone(position_3d)
        zone_dwell_time = self._compute_zone_dwell_time(person_id, current_zone, timestamp)

        # Create attention state
        state = AttentionState(
            person_id=person_id,
            timestamp=timestamp,
            gaze_direction_3d=gaze_direction,
            gaze_target=gaze_target,
            gaze_confidence=gaze_confidence,
            is_focused=is_focused,
            focus_duration=focus_duration,
            distraction_level=distraction_level,
            gaze_stability=gaze_stability,
            saccade_rate=saccade_rate,
            fixation_count=fixation_count,
            current_zone=current_zone,
            zone_dwell_time=zone_dwell_time
        )

        self.person_states[person_id] = state

        return state

    def _head_pose_to_gaze_direction(self, head_pose: Dict) -> np.ndarray:
        """
        Convert head pose to 3D gaze direction vector.

        Args:
            head_pose: Dict with yaw, pitch, roll in degrees

        Returns:
            Normalized 3D gaze direction vector
        """
        yaw = np.radians(head_pose.get('yaw', 0))
        pitch = np.radians(head_pose.get('pitch', 0))

        # Compute direction vector from yaw and pitch
        # Yaw: rotation around Y axis
        # Pitch: rotation around X axis
        direction = np.array([
            np.cos(pitch) * np.sin(yaw),
            -np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])

        # Normalize
        direction = direction / np.linalg.norm(direction)

        return direction

    def _find_gaze_target(self, origin: np.ndarray, direction: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray], float]:
        """
        Find what the person is looking at.

        Args:
            origin: Gaze origin (person position + eye offset)
            direction: Normalized gaze direction vector

        Returns:
            (target_id, gaze_point, confidence)
        """
        # Add eye offset (approximate)
        eye_offset = np.array([0, 0, 1.6])  # 1.6m for standing person
        ray_origin = origin + eye_offset

        best_target = None
        best_distance = float('inf')
        best_point = None

        # Check intersection with each target
        for target_id, target in self.attention_targets.items():
            # Ray-sphere intersection for point/sphere targets
            if target.bbox_3d is None:
                distance, point = self._ray_sphere_intersection(
                    ray_origin, direction, target.position_3d, target.radius
                )
            else:
                # Ray-box intersection for box targets
                distance, point = self._ray_box_intersection(
                    ray_origin, direction, target.bbox_3d
                )

            if distance is not None and distance < best_distance:
                best_distance = distance
                best_target = target_id
                best_point = point

        # If no target found, project gaze to ground plane
        if best_target is None:
            ground_point = self._ray_ground_intersection(ray_origin, direction)
            if ground_point is not None:
                best_point = ground_point

        # Compute confidence based on distance
        confidence = 1.0 / (1.0 + best_distance * 0.1) if best_distance != float('inf') else 0.5

        return best_target, best_point, confidence

    def _ray_sphere_intersection(self, origin: np.ndarray, direction: np.ndarray,
                                  center: np.ndarray, radius: float) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """Ray-sphere intersection test."""
        oc = origin - center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None, None

        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t < 0:
            t = (-b + np.sqrt(discriminant)) / (2.0 * a)

        if t < 0:
            return None, None

        point = origin + t * direction
        return t, point

    def _ray_box_intersection(self, origin: np.ndarray, direction: np.ndarray,
                              bbox: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """Ray-AABB intersection test."""
        tmin = (bbox[:3] - origin) / (direction + 1e-8)
        tmax = (bbox[3:] - origin) / (direction + 1e-8)

        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)

        tnear = np.max(t1)
        tfar = np.min(t2)

        if tnear > tfar or tfar < 0:
            return None, None

        t = tnear if tnear >= 0 else tfar
        point = origin + t * direction

        return t, point

    def _ray_ground_intersection(self, origin: np.ndarray, direction: np.ndarray,
                                  ground_z: float = 0.0) -> Optional[np.ndarray]:
        """Ray intersection with ground plane."""
        if abs(direction[2]) < 1e-6:  # Parallel to ground
            return None

        t = (ground_z - origin[2]) / direction[2]
        if t < 0:
            return None

        return origin + t * direction

    def _compute_gaze_stability(self, person_id: int) -> float:
        """Compute gaze stability from recent history."""
        if person_id not in self.gaze_history or len(self.gaze_history[person_id]) < 10:
            return 0.0

        points = [gp.position_3d for gp in self.gaze_history[person_id]]
        points = np.array(points)

        # Compute standard deviation of gaze points
        std_dev = np.std(points, axis=0)
        total_std = np.linalg.norm(std_dev)

        # Map to 0-1 range (lower std = higher stability)
        stability = 1.0 / (1.0 + total_std)

        return stability

    def _compute_saccade_rate(self, person_id: int) -> float:
        """Compute saccade rate (rapid eye movements per second)."""
        if person_id not in self.gaze_history or len(self.gaze_history[person_id]) < 2:
            return 0.0

        history = list(self.gaze_history[person_id])
        saccade_count = 0

        for i in range(1, len(history)):
            # Angular distance between consecutive gaze points
            v1 = history[i-1].position_3d
            v2 = history[i].position_3d

            angle = self._angular_distance(v1, v2)

            if angle > self.SACCADE_THRESHOLD:
                saccade_count += 1

        # Compute rate
        if len(history) > 1:
            duration = history[-1].timestamp - history[0].timestamp
            if duration > 0:
                return saccade_count / duration

        return 0.0

    def _count_fixations(self, person_id: int) -> int:
        """Count distinct fixation periods."""
        if person_id not in self.gaze_history or len(self.gaze_history[person_id]) < 2:
            return 0

        history = list(self.gaze_history[person_id])
        fixation_count = 0
        in_fixation = False

        for i in range(1, len(history)):
            v1 = history[i-1].position_3d
            v2 = history[i].position_3d

            angle = self._angular_distance(v1, v2)

            if angle < self.FIXATION_THRESHOLD:
                if not in_fixation:
                    fixation_count += 1
                    in_fixation = True
            else:
                in_fixation = False

        return fixation_count

    def _angular_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute angular distance between two points (in degrees)."""
        v1 = p1 / (np.linalg.norm(p1) + 1e-8)
        v2 = p2 / (np.linalg.norm(p2) + 1e-8)

        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def _is_focused(self, stability: float, saccade_rate: float, fixation_count: int) -> bool:
        """Determine if person is focused."""
        return (stability >= self.FOCUS_GAZE_STABILITY_THRESHOLD and
                saccade_rate < self.DISTRACTION_SACCADE_RATE)

    def _compute_distraction_level(self, stability: float, saccade_rate: float) -> float:
        """Compute distraction level (0=focused, 1=distracted)."""
        # Combine stability and saccade rate
        distraction = (1.0 - stability) * 0.6 + min(saccade_rate / 5.0, 1.0) * 0.4
        return np.clip(distraction, 0.0, 1.0)

    def _compute_focus_duration(self, person_id: int, is_focused: bool, timestamp: float) -> float:
        """Compute continuous focus duration."""
        if person_id not in self.person_states:
            return 0.0

        prev_state = self.person_states[person_id]

        if is_focused and prev_state.is_focused:
            return prev_state.focus_duration + (timestamp - prev_state.timestamp)
        elif is_focused:
            return timestamp - prev_state.timestamp
        else:
            return 0.0

    def _find_current_zone(self, position_3d: np.ndarray) -> Optional[str]:
        """Find which zone the person is in."""
        for zone_id, zone in self.attention_zones.items():
            bounds = zone['bounds']
            if (bounds[0] <= position_3d[0] <= bounds[3] and
                bounds[1] <= position_3d[1] <= bounds[4] and
                bounds[2] <= position_3d[2] <= bounds[5]):
                return zone_id
        return None

    def _compute_zone_dwell_time(self, person_id: int, zone_id: Optional[str],
                                 timestamp: float) -> float:
        """Compute dwell time in current zone."""
        if person_id not in self.person_states or zone_id is None:
            return 0.0

        prev_state = self.person_states[person_id]

        if prev_state.current_zone == zone_id:
            return prev_state.zone_dwell_time + (timestamp - prev_state.timestamp)
        else:
            return 0.0

    def _update_heatmap(self, point: np.ndarray):
        """Update attention heatmap with gaze point."""
        # Convert world point to heatmap indices
        indices = []
        for i, (p, (min_val, max_val), bins) in enumerate(
            zip(point, self.heatmap_bounds, self.heatmap_resolution)
        ):
            idx = int((p - min_val) / (max_val - min_val) * bins)
            idx = np.clip(idx, 0, bins - 1)
            indices.append(idx)

        # Increment heatmap
        self.attention_heatmap[tuple(indices)] += 1.0

    def get_attention_heatmap(self, normalize: bool = True) -> np.ndarray:
        """
        Get attention heatmap.

        Args:
            normalize: Whether to normalize to 0-1 range

        Returns:
            3D attention heatmap array
        """
        if normalize and self.attention_heatmap.max() > 0:
            return self.attention_heatmap / self.attention_heatmap.max()
        return self.attention_heatmap.copy()

    def get_attention_state(self, person_id: int) -> Optional[AttentionState]:
        """Get attention state for person."""
        return self.person_states.get(person_id)

    def get_zone_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all attention zones."""
        return self.attention_zones.copy()

    def reset_heatmap(self):
        """Reset attention heatmap."""
        self.attention_heatmap = np.zeros(self.heatmap_resolution)
        logger.info("Attention heatmap reset")
