"""
Scene Understanding Module for OMNISENSE platform.

Provides spatial scene analysis including zone detection, occupancy tracking,
flow analysis, and spatial context reasoning.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict, deque

from omnisense.utils.logger import get_logger

logger = get_logger(__name__)


class ZoneType(Enum):
    """Types of spatial zones."""
    ENTRY = "entry"
    EXIT = "exit"
    RESTRICTED = "restricted"
    MONITORED = "monitored"
    SAFE = "safe"
    WORKSPACE = "workspace"
    PATHWAY = "pathway"
    WAITING = "waiting"
    INTERACTION = "interaction"


class OccupancyLevel(Enum):
    """Zone occupancy levels."""
    EMPTY = "empty"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    OVERCROWDED = "overcrowded"


@dataclass
class Zone:
    """Spatial zone definition."""
    zone_id: str
    name: str
    zone_type: ZoneType
    bounds_3d: np.ndarray  # [min_x, min_y, min_z, max_x, max_y, max_z]
    max_capacity: Optional[int] = None
    enabled: bool = True

    # Zone properties
    color: Tuple[int, int, int] = (0, 255, 0)
    height: Optional[float] = None  # Zone height if 2D zone
    priority: int = 0  # Higher priority zones checked first

    def contains(self, position_3d: np.ndarray) -> bool:
        """Check if 3D position is within zone."""
        return (self.bounds_3d[0] <= position_3d[0] <= self.bounds_3d[3] and
                self.bounds_3d[1] <= position_3d[1] <= self.bounds_3d[4] and
                self.bounds_3d[2] <= position_3d[2] <= self.bounds_3d[5])

    def get_center(self) -> np.ndarray:
        """Get zone center point."""
        return np.array([
            (self.bounds_3d[0] + self.bounds_3d[3]) / 2,
            (self.bounds_3d[1] + self.bounds_3d[4]) / 2,
            (self.bounds_3d[2] + self.bounds_3d[5]) / 2
        ])

    def get_volume(self) -> float:
        """Get zone volume in cubic meters."""
        width = self.bounds_3d[3] - self.bounds_3d[0]
        depth = self.bounds_3d[4] - self.bounds_3d[1]
        height = self.bounds_3d[5] - self.bounds_3d[2]
        return width * depth * height


@dataclass
class ZoneEvent:
    """Event occurring in a zone."""
    event_type: str  # "entry", "exit", "dwell", "violation"
    zone_id: str
    person_id: int
    timestamp: float
    position_3d: Optional[np.ndarray] = None


@dataclass
class ZoneOccupancy:
    """Zone occupancy state."""
    zone_id: str
    person_ids: Set[int] = field(default_factory=set)
    occupancy_count: int = 0
    occupancy_level: OccupancyLevel = OccupancyLevel.EMPTY
    density: float = 0.0  # Persons per square meter
    dwell_times: Dict[int, float] = field(default_factory=dict)
    last_update: float = 0.0


@dataclass
class FlowVector:
    """Pedestrian flow vector in a zone."""
    zone_id: str
    direction: np.ndarray  # Average movement direction
    speed: float  # Average speed in m/s
    person_count: int
    timestamp: float


class SceneAnalyzer:
    """
    Analyzes spatial scene understanding.

    Features:
    - Zone-based spatial analysis
    - Occupancy tracking and density estimation
    - Flow analysis and trajectory patterns
    - Entry/exit monitoring
    - Restricted area violations
    - Crowd density estimation
    """

    # Occupancy thresholds (persons per square meter)
    OCCUPANCY_LOW = 0.5
    OCCUPANCY_MEDIUM = 1.0
    OCCUPANCY_HIGH = 2.0

    # Dwell time thresholds
    SHORT_DWELL = 5.0  # seconds
    LONG_DWELL = 60.0  # seconds

    def __init__(self):
        """Initialize scene analyzer."""
        self.zones: Dict[str, Zone] = {}
        self.zone_occupancy: Dict[str, ZoneOccupancy] = {}
        self.zone_events: List[ZoneEvent] = []

        # Person-zone tracking
        self.person_current_zones: Dict[int, Set[str]] = defaultdict(set)
        self.person_zone_history: Dict[int, List[Tuple[str, float, float]]] = defaultdict(list)

        # Flow analysis
        self.flow_history: Dict[str, deque] = {}
        self.flow_window_seconds = 10.0

        # Global occupancy grid (for heatmap)
        self.grid_resolution = (100, 100)  # X, Y bins
        self.grid_bounds = np.array([[-10, 10], [-10, 10]])  # World bounds
        self.occupancy_grid = np.zeros(self.grid_resolution)

        logger.info("Scene analyzer initialized")

    def add_zone(self, zone: Zone):
        """
        Add spatial zone.

        Args:
            zone: Zone to add
        """
        self.zones[zone.zone_id] = zone
        self.zone_occupancy[zone.zone_id] = ZoneOccupancy(
            zone_id=zone.zone_id,
            last_update=time.time()
        )
        self.flow_history[zone.zone_id] = deque(maxlen=int(self.flow_window_seconds * 30))

        logger.info(f"Added zone: {zone.name} ({zone.zone_id}) - Type: {zone.zone_type.value}")

    def remove_zone(self, zone_id: str):
        """Remove zone."""
        if zone_id in self.zones:
            del self.zones[zone_id]
            del self.zone_occupancy[zone_id]
            del self.flow_history[zone_id]
            logger.info(f"Removed zone: {zone_id}")

    def update(self, persons: Dict[int, any], timestamp: Optional[float] = None):
        """
        Update scene analysis with current person states.

        Args:
            persons: Dictionary of Person3D objects keyed by global_id
            timestamp: Current timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        # Clear current occupancy
        for zone_id in self.zone_occupancy:
            self.zone_occupancy[zone_id].person_ids.clear()

        # Update person-zone associations
        for person_id, person in persons.items():
            position = person.position_3d
            velocity = person.velocity_3d

            # Find zones containing this person
            current_zones = set()
            for zone_id, zone in self.zones.items():
                if not zone.enabled:
                    continue

                if zone.contains(position):
                    current_zones.add(zone_id)
                    self.zone_occupancy[zone_id].person_ids.add(person_id)

                    # Update dwell time
                    if person_id not in self.zone_occupancy[zone_id].dwell_times:
                        self.zone_occupancy[zone_id].dwell_times[person_id] = timestamp
                    else:
                        # Calculate dwell time
                        dwell_time = timestamp - self.zone_occupancy[zone_id].dwell_times[person_id]

                    # Update flow
                    self.flow_history[zone_id].append({
                        'person_id': person_id,
                        'velocity': velocity,
                        'timestamp': timestamp
                    })

            # Detect zone transitions (entry/exit)
            previous_zones = self.person_current_zones.get(person_id, set())
            entered_zones = current_zones - previous_zones
            exited_zones = previous_zones - current_zones

            # Generate entry events
            for zone_id in entered_zones:
                event = ZoneEvent(
                    event_type="entry",
                    zone_id=zone_id,
                    person_id=person_id,
                    timestamp=timestamp,
                    position_3d=position.copy()
                )
                self.zone_events.append(event)

                # Check for restricted zone violation
                zone = self.zones[zone_id]
                if zone.zone_type == ZoneType.RESTRICTED:
                    violation_event = ZoneEvent(
                        event_type="violation",
                        zone_id=zone_id,
                        person_id=person_id,
                        timestamp=timestamp,
                        position_3d=position.copy()
                    )
                    self.zone_events.append(violation_event)
                    logger.warning(f"Restricted zone violation: Person {person_id} entered {zone.name}")

            # Generate exit events
            for zone_id in exited_zones:
                event = ZoneEvent(
                    event_type="exit",
                    zone_id=zone_id,
                    person_id=person_id,
                    timestamp=timestamp,
                    position_3d=position.copy()
                )
                self.zone_events.append(event)

                # Record zone history
                if person_id in self.zone_occupancy[zone_id].dwell_times:
                    entry_time = self.zone_occupancy[zone_id].dwell_times[person_id]
                    dwell_time = timestamp - entry_time
                    self.person_zone_history[person_id].append((zone_id, entry_time, dwell_time))
                    del self.zone_occupancy[zone_id].dwell_times[person_id]

            # Update current zones
            self.person_current_zones[person_id] = current_zones

            # Update occupancy grid
            self._update_occupancy_grid(position)

        # Update zone occupancy statistics
        for zone_id, occupancy in self.zone_occupancy.items():
            occupancy.occupancy_count = len(occupancy.person_ids)
            occupancy.occupancy_level = self._compute_occupancy_level(zone_id, occupancy.occupancy_count)
            occupancy.density = self._compute_density(zone_id, occupancy.occupancy_count)
            occupancy.last_update = timestamp

        # Trim old events (keep last hour)
        cutoff_time = timestamp - 3600
        self.zone_events = [e for e in self.zone_events if e.timestamp > cutoff_time]

    def _compute_occupancy_level(self, zone_id: str, count: int) -> OccupancyLevel:
        """Compute occupancy level for zone."""
        if count == 0:
            return OccupancyLevel.EMPTY

        zone = self.zones[zone_id]

        # If max capacity is set, use it
        if zone.max_capacity is not None:
            ratio = count / zone.max_capacity
            if ratio >= 1.0:
                return OccupancyLevel.OVERCROWDED
            elif ratio >= 0.75:
                return OccupancyLevel.HIGH
            elif ratio >= 0.5:
                return OccupancyLevel.MEDIUM
            else:
                return OccupancyLevel.LOW

        # Otherwise use density
        density = self._compute_density(zone_id, count)

        if density >= self.OCCUPANCY_HIGH:
            return OccupancyLevel.OVERCROWDED
        elif density >= self.OCCUPANCY_MEDIUM:
            return OccupancyLevel.HIGH
        elif density >= self.OCCUPANCY_LOW:
            return OccupancyLevel.MEDIUM
        else:
            return OccupancyLevel.LOW

    def _compute_density(self, zone_id: str, count: int) -> float:
        """Compute person density in zone (persons per square meter)."""
        zone = self.zones[zone_id]

        # Compute floor area
        width = zone.bounds_3d[3] - zone.bounds_3d[0]
        depth = zone.bounds_3d[4] - zone.bounds_3d[1]
        area = width * depth

        if area <= 0:
            return 0.0

        return count / area

    def compute_flow_vector(self, zone_id: str) -> Optional[FlowVector]:
        """
        Compute flow vector for zone.

        Args:
            zone_id: Zone ID

        Returns:
            Flow vector or None if insufficient data
        """
        if zone_id not in self.flow_history or len(self.flow_history[zone_id]) < 5:
            return None

        flow_data = list(self.flow_history[zone_id])

        # Extract velocities
        velocities = [data['velocity'] for data in flow_data]
        velocities = np.array(velocities)

        # Compute average direction and speed
        avg_velocity = np.mean(velocities, axis=0)
        avg_speed = np.linalg.norm(avg_velocity)

        if avg_speed < 1e-3:
            direction = np.array([0, 0, 0])
        else:
            direction = avg_velocity / avg_speed

        # Count unique persons
        person_ids = set(data['person_id'] for data in flow_data)

        return FlowVector(
            zone_id=zone_id,
            direction=direction,
            speed=avg_speed,
            person_count=len(person_ids),
            timestamp=flow_data[-1]['timestamp']
        )

    def _update_occupancy_grid(self, position: np.ndarray):
        """Update global occupancy grid."""
        # Convert position to grid indices
        x_idx = int((position[0] - self.grid_bounds[0, 0]) /
                    (self.grid_bounds[0, 1] - self.grid_bounds[0, 0]) *
                    self.grid_resolution[0])
        y_idx = int((position[1] - self.grid_bounds[1, 0]) /
                    (self.grid_bounds[1, 1] - self.grid_bounds[1, 0]) *
                    self.grid_resolution[1])

        # Clip to bounds
        x_idx = np.clip(x_idx, 0, self.grid_resolution[0] - 1)
        y_idx = np.clip(y_idx, 0, self.grid_resolution[1] - 1)

        # Increment grid cell
        self.occupancy_grid[x_idx, y_idx] += 1

    def get_zone_occupancy(self, zone_id: str) -> Optional[ZoneOccupancy]:
        """Get occupancy state for zone."""
        return self.zone_occupancy.get(zone_id)

    def get_all_zone_occupancies(self) -> Dict[str, ZoneOccupancy]:
        """Get occupancy states for all zones."""
        return self.zone_occupancy.copy()

    def get_zone_events(self, zone_id: Optional[str] = None,
                       event_type: Optional[str] = None,
                       since: Optional[float] = None) -> List[ZoneEvent]:
        """
        Get zone events with optional filtering.

        Args:
            zone_id: Filter by zone ID
            event_type: Filter by event type
            since: Filter by timestamp (events after this time)

        Returns:
            List of zone events
        """
        events = self.zone_events

        if zone_id:
            events = [e for e in events if e.zone_id == zone_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events

    def get_person_current_zones(self, person_id: int) -> Set[str]:
        """Get zones currently occupied by person."""
        return self.person_current_zones.get(person_id, set())

    def get_person_zone_history(self, person_id: int) -> List[Tuple[str, float, float]]:
        """
        Get zone visit history for person.

        Returns:
            List of (zone_id, entry_time, dwell_time) tuples
        """
        return self.person_zone_history.get(person_id, [])

    def get_occupancy_grid(self, normalize: bool = True) -> np.ndarray:
        """
        Get occupancy heatmap grid.

        Args:
            normalize: Whether to normalize to 0-1 range

        Returns:
            2D occupancy grid
        """
        if normalize and self.occupancy_grid.max() > 0:
            return self.occupancy_grid / self.occupancy_grid.max()
        return self.occupancy_grid.copy()

    def reset_occupancy_grid(self):
        """Reset occupancy grid."""
        self.occupancy_grid = np.zeros(self.grid_resolution)
        logger.info("Occupancy grid reset")

    def get_zone_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all zones."""
        stats = {}

        for zone_id, zone in self.zones.items():
            occupancy = self.zone_occupancy[zone_id]
            flow = self.compute_flow_vector(zone_id)

            # Compute total visits
            total_visits = len([e for e in self.zone_events
                              if e.zone_id == zone_id and e.event_type == "entry"])

            # Compute average dwell time
            zone_history = [h for p_history in self.person_zone_history.values()
                          for h in p_history if h[0] == zone_id]
            avg_dwell = np.mean([h[2] for h in zone_history]) if zone_history else 0.0

            stats[zone_id] = {
                'name': zone.name,
                'type': zone.zone_type.value,
                'current_occupancy': occupancy.occupancy_count,
                'occupancy_level': occupancy.occupancy_level.value,
                'density': occupancy.density,
                'total_visits': total_visits,
                'average_dwell_time': avg_dwell,
                'flow_speed': flow.speed if flow else 0.0,
                'flow_direction': flow.direction.tolist() if flow else [0, 0, 0]
            }

        return stats

    def detect_crowding(self, threshold: Optional[float] = None) -> List[str]:
        """
        Detect overcrowded zones.

        Args:
            threshold: Custom density threshold (uses OCCUPANCY_HIGH if None)

        Returns:
            List of zone IDs that are overcrowded
        """
        if threshold is None:
            threshold = self.OCCUPANCY_HIGH

        crowded_zones = []

        for zone_id, occupancy in self.zone_occupancy.items():
            if occupancy.density >= threshold or occupancy.occupancy_level == OccupancyLevel.OVERCROWDED:
                crowded_zones.append(zone_id)

        return crowded_zones

    def detect_loitering(self, threshold_seconds: float = 300) -> List[Tuple[int, str, float]]:
        """
        Detect persons loitering in zones.

        Args:
            threshold_seconds: Loitering threshold in seconds

        Returns:
            List of (person_id, zone_id, dwell_time) tuples
        """
        loiterers = []

        for zone_id, occupancy in self.zone_occupancy.items():
            for person_id, entry_time in occupancy.dwell_times.items():
                dwell_time = time.time() - entry_time
                if dwell_time >= threshold_seconds:
                    loiterers.append((person_id, zone_id, dwell_time))

        return loiterers
