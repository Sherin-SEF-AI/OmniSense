"""
Interaction Detection Module for OMNISENSE platform.

Detects and analyzes interactions between tracked persons including
proximity-based interactions, F-formations, and social grouping.
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import time

from omnisense.utils.logger import get_logger

logger = get_logger(__name__)


class InteractionType(Enum):
    """Types of interactions."""
    PROXIMITY = "proximity"
    CONVERSATION = "conversation"
    F_FORMATION = "f_formation"
    FOLLOWING = "following"
    APPROACHING = "approaching"
    AVOIDING = "avoiding"
    PASSING = "passing"
    GROUP = "group"


class GroupFormation(Enum):
    """Group formation types."""
    VIS_A_VIS = "vis_a_vis"  # Face-to-face
    L_SHAPE = "l_shape"  # L-shaped
    SIDE_BY_SIDE = "side_by_side"
    CIRCULAR = "circular"
    SEMICIRCULAR = "semicircular"


@dataclass
class Interaction:
    """Detected interaction between persons."""
    interaction_id: int
    interaction_type: InteractionType
    person_ids: Set[int]
    confidence: float
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0

    # Spatial properties
    distance: Optional[float] = None  # Average distance between persons
    formation: Optional[GroupFormation] = None
    o_space_center: Optional[np.ndarray] = None  # O-space center for F-formations

    # Interaction characteristics
    is_stable: bool = False
    engagement_level: float = 0.0  # 0.0 (low) to 1.0 (high)


@dataclass
class SocialGroup:
    """Social group of interacting persons."""
    group_id: int
    person_ids: Set[int]
    formation: GroupFormation
    center_position: np.ndarray
    radius: float
    created_time: float
    last_update: float
    stability: float = 0.0


class InteractionDetector:
    """
    Detects and analyzes person interactions.

    Features:
    - Proximity-based interaction detection
    - F-formation detection (Kendon's work on conversational groups)
    - Social grouping and tracking
    - Approach/avoidance detection
    - Following behavior detection
    """

    # Distance thresholds (meters)
    INTIMATE_DISTANCE = 0.5
    PERSONAL_DISTANCE = 1.2
    SOCIAL_DISTANCE = 3.6
    INTERACTION_DISTANCE = 4.0

    # Angle thresholds (degrees)
    FACING_ANGLE_THRESHOLD = 45
    F_FORMATION_ANGLE_THRESHOLD = 90

    # Temporal thresholds
    MIN_INTERACTION_DURATION = 2.0  # seconds
    GROUP_MERGE_DISTANCE = 2.0  # meters

    def __init__(self):
        """Initialize interaction detector."""
        self.interactions: Dict[int, Interaction] = {}
        self.interaction_history: List[Interaction] = []
        self.next_interaction_id = 0

        # Social groups
        self.groups: Dict[int, SocialGroup] = {}
        self.next_group_id = 0

        # Person pair history for tracking
        self.pair_proximity_history: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        logger.info("Interaction detector initialized")

    def update(self, persons: Dict[int, any], timestamp: Optional[float] = None):
        """
        Update interaction detection.

        Args:
            persons: Dictionary of Person3D objects
            timestamp: Current timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        # Clear current interactions
        active_interactions = set()

        # Detect pairwise interactions
        person_ids = list(persons.keys())
        for i, person_id_1 in enumerate(person_ids):
            for person_id_2 in person_ids[i+1:]:
                person_1 = persons[person_id_1]
                person_2 = persons[person_id_2]

                # Check proximity
                interaction = self._detect_pairwise_interaction(
                    person_id_1, person_1,
                    person_id_2, person_2,
                    timestamp
                )

                if interaction:
                    active_interactions.add(interaction.interaction_id)

        # Detect group interactions (F-formations)
        group_interactions = self._detect_group_interactions(persons, timestamp)
        for interaction in group_interactions:
            active_interactions.add(interaction.interaction_id)

        # Update group tracking
        self._update_groups(persons, timestamp)

        # Clean up ended interactions
        ended_interactions = []
        for interaction_id, interaction in list(self.interactions.items()):
            if interaction_id not in active_interactions:
                if interaction.end_time is None:
                    interaction.end_time = timestamp
                    interaction.duration = timestamp - interaction.start_time

                    # Only keep interactions that lasted minimum duration
                    if interaction.duration >= self.MIN_INTERACTION_DURATION:
                        self.interaction_history.append(interaction)

                    ended_interactions.append(interaction_id)

        for interaction_id in ended_interactions:
            del self.interactions[interaction_id]

    def _detect_pairwise_interaction(self, person_id_1: int, person_1: any,
                                    person_id_2: int, person_2: any,
                                    timestamp: float) -> Optional[Interaction]:
        """Detect interaction between two persons."""
        # Compute distance
        distance = np.linalg.norm(person_1.position_3d - person_2.position_3d)

        # Store in history
        pair_key = tuple(sorted([person_id_1, person_id_2]))
        self.pair_proximity_history[pair_key].append(distance)
        if len(self.pair_proximity_history[pair_key]) > 30:  # Keep last 30 samples
            self.pair_proximity_history[pair_key].pop(0)

        # Check if within interaction distance
        if distance > self.INTERACTION_DISTANCE:
            return None

        # Determine interaction type based on distance and orientation
        interaction_type = None
        confidence = 0.0

        if distance < self.INTIMATE_DISTANCE:
            interaction_type = InteractionType.PROXIMITY
            confidence = 0.95

        elif distance < self.PERSONAL_DISTANCE:
            # Check if facing each other
            if self._are_facing(person_1, person_2):
                interaction_type = InteractionType.CONVERSATION
                confidence = 0.90
            else:
                interaction_type = InteractionType.PROXIMITY
                confidence = 0.75

        elif distance < self.SOCIAL_DISTANCE:
            # Check movement patterns
            if self._is_following(person_id_1, person_1, person_id_2, person_2):
                interaction_type = InteractionType.FOLLOWING
                confidence = 0.80
            elif self._are_facing(person_1, person_2):
                interaction_type = InteractionType.CONVERSATION
                confidence = 0.70
            else:
                interaction_type = InteractionType.PROXIMITY
                confidence = 0.60

        else:  # Within INTERACTION_DISTANCE but far
            # Check if approaching
            if self._are_approaching(pair_key):
                interaction_type = InteractionType.APPROACHING
                confidence = 0.65
            else:
                return None

        # Find or create interaction
        pair_set = {person_id_1, person_id_2}
        existing_interaction = None

        for interaction in self.interactions.values():
            if interaction.person_ids == pair_set:
                existing_interaction = interaction
                break

        if existing_interaction:
            # Update existing interaction
            existing_interaction.interaction_type = interaction_type
            existing_interaction.confidence = confidence
            existing_interaction.distance = distance
            existing_interaction.duration = timestamp - existing_interaction.start_time
            return existing_interaction
        else:
            # Create new interaction
            interaction = Interaction(
                interaction_id=self.next_interaction_id,
                interaction_type=interaction_type,
                person_ids=pair_set,
                confidence=confidence,
                start_time=timestamp,
                distance=distance
            )
            self.next_interaction_id += 1
            self.interactions[interaction.interaction_id] = interaction
            return interaction

    def _are_facing(self, person_1: any, person_2: any) -> bool:
        """Check if two persons are facing each other."""
        if not person_1.head_pose or not person_2.head_pose:
            return False

        # Get head yaw angles
        yaw_1 = person_1.head_pose.get('yaw', 0)
        yaw_2 = person_2.head_pose.get('yaw', 0)

        # Compute vector from person 1 to person 2
        direction_vector = person_2.position_3d - person_1.position_3d
        direction_angle = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))

        # Check if person 1 is facing person 2
        angle_diff_1 = abs(self._angle_difference(yaw_1, direction_angle))

        # Check if person 2 is facing person 1
        opposite_angle = (direction_angle + 180) % 360
        angle_diff_2 = abs(self._angle_difference(yaw_2, opposite_angle))

        # Both should be facing each other
        return (angle_diff_1 < self.FACING_ANGLE_THRESHOLD and
                angle_diff_2 < self.FACING_ANGLE_THRESHOLD)

    def _is_following(self, person_id_1: int, person_1: any,
                     person_id_2: int, person_2: any) -> bool:
        """Check if person_1 is following person_2."""
        # Check if moving in same direction
        vel_1 = person_1.velocity_3d
        vel_2 = person_2.velocity_3d

        speed_1 = np.linalg.norm(vel_1)
        speed_2 = np.linalg.norm(vel_2)

        # Both must be moving
        if speed_1 < 0.2 or speed_2 < 0.2:
            return False

        # Check if velocities are similar
        vel_angle_diff = np.degrees(np.arccos(
            np.clip(np.dot(vel_1, vel_2) / (speed_1 * speed_2 + 1e-6), -1, 1)
        ))

        # Check if person_1 is behind person_2
        relative_pos = person_1.position_3d - person_2.position_3d
        dot_product = np.dot(relative_pos, vel_2)

        return vel_angle_diff < 30 and dot_product < 0

    def _are_approaching(self, pair_key: Tuple[int, int]) -> bool:
        """Check if two persons are approaching each other."""
        if pair_key not in self.pair_proximity_history:
            return False

        history = self.pair_proximity_history[pair_key]
        if len(history) < 5:
            return False

        # Check if distance is decreasing
        recent_distances = history[-5:]
        decreasing = all(recent_distances[i] > recent_distances[i+1]
                        for i in range(len(recent_distances)-1))

        return decreasing

    def _detect_group_interactions(self, persons: Dict[int, any],
                                   timestamp: float) -> List[Interaction]:
        """Detect F-formation group interactions."""
        interactions = []

        # Find clusters of people
        person_ids = list(persons.keys())
        if len(person_ids) < 3:
            return interactions

        # Try to find F-formations
        for i in range(len(person_ids)):
            for j in range(i+1, len(person_ids)):
                for k in range(j+1, len(person_ids)):
                    person_set = {person_ids[i], person_ids[j], person_ids[k]}
                    persons_subset = {pid: persons[pid] for pid in person_set}

                    formation_info = self._check_f_formation(persons_subset)

                    if formation_info:
                        formation_type, o_space_center, confidence = formation_info

                        # Create interaction
                        interaction = Interaction(
                            interaction_id=self.next_interaction_id,
                            interaction_type=InteractionType.F_FORMATION,
                            person_ids=person_set,
                            confidence=confidence,
                            start_time=timestamp,
                            formation=formation_type,
                            o_space_center=o_space_center
                        )
                        self.next_interaction_id += 1
                        self.interactions[interaction.interaction_id] = interaction
                        interactions.append(interaction)

        return interactions

    def _check_f_formation(self, persons: Dict[int, any]) -> Optional[Tuple[GroupFormation, np.ndarray, float]]:
        """
        Check if persons form an F-formation.

        Returns:
            (formation_type, o_space_center, confidence) or None
        """
        positions = np.array([p.position_3d for p in persons.values()])

        # Compute center point (O-space)
        center = np.mean(positions, axis=0)

        # Check if all persons are facing the center
        facing_center_count = 0
        for person in persons.values():
            if person.head_pose:
                # Vector from person to center
                to_center = center - person.position_3d
                center_angle = np.degrees(np.arctan2(to_center[1], to_center[0]))

                # Person's facing direction
                yaw = person.head_pose.get('yaw', 0)

                angle_diff = abs(self._angle_difference(yaw, center_angle))

                if angle_diff < self.F_FORMATION_ANGLE_THRESHOLD:
                    facing_center_count += 1

        # Determine formation type
        if facing_center_count >= len(persons) * 0.7:  # At least 70% facing center
            # Compute distances to center
            distances = [np.linalg.norm(p.position_3d - center) for p in persons.values()]
            avg_distance = np.mean(distances)

            if avg_distance < self.PERSONAL_DISTANCE:
                # Close formation
                if len(persons) == 2:
                    return GroupFormation.VIS_A_VIS, center, 0.85
                else:
                    return GroupFormation.CIRCULAR, center, 0.85
            else:
                return GroupFormation.SEMICIRCULAR, center, 0.75

        return None

    def _update_groups(self, persons: Dict[int, any], timestamp: float):
        """Update social group tracking."""
        # Extract active groups from current interactions
        active_group_members = set()

        for interaction in self.interactions.values():
            if interaction.interaction_type in [InteractionType.F_FORMATION,
                                               InteractionType.CONVERSATION]:
                # Check if this forms a group
                if len(interaction.person_ids) >= 2:
                    # Find or create group
                    existing_group = None
                    for group in self.groups.values():
                        if len(group.person_ids.intersection(interaction.person_ids)) >= 2:
                            existing_group = group
                            break

                    if existing_group:
                        # Update existing group
                        existing_group.person_ids = interaction.person_ids
                        existing_group.last_update = timestamp
                    else:
                        # Create new group
                        positions = np.array([persons[pid].position_3d
                                            for pid in interaction.person_ids])
                        center = np.mean(positions, axis=0)
                        radius = np.max([np.linalg.norm(positions[i] - center)
                                       for i in range(len(positions))])

                        group = SocialGroup(
                            group_id=self.next_group_id,
                            person_ids=interaction.person_ids,
                            formation=interaction.formation or GroupFormation.CIRCULAR,
                            center_position=center,
                            radius=radius,
                            created_time=timestamp,
                            last_update=timestamp
                        )
                        self.next_group_id += 1
                        self.groups[group.group_id] = group

                    active_group_members.update(interaction.person_ids)

        # Remove stale groups
        stale_groups = []
        for group_id, group in self.groups.items():
            if timestamp - group.last_update > 10.0:  # 10 seconds timeout
                stale_groups.append(group_id)

        for group_id in stale_groups:
            del self.groups[group_id]

    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """Compute smallest difference between two angles in degrees."""
        diff = (angle1 - angle2 + 180) % 360 - 180
        return abs(diff)

    def get_active_interactions(self, interaction_type: Optional[InteractionType] = None) -> List[Interaction]:
        """
        Get active interactions.

        Args:
            interaction_type: Filter by type (returns all if None)

        Returns:
            List of active interactions
        """
        interactions = list(self.interactions.values())

        if interaction_type:
            interactions = [i for i in interactions if i.interaction_type == interaction_type]

        return interactions

    def get_person_interactions(self, person_id: int) -> List[Interaction]:
        """Get all interactions involving a specific person."""
        return [i for i in self.interactions.values() if person_id in i.person_ids]

    def get_social_groups(self) -> List[SocialGroup]:
        """Get all active social groups."""
        return list(self.groups.values())

    def get_interaction_history(self, person_id: Optional[int] = None,
                               interaction_type: Optional[InteractionType] = None,
                               since: Optional[float] = None) -> List[Interaction]:
        """
        Get interaction history with optional filtering.

        Args:
            person_id: Filter by person ID
            interaction_type: Filter by interaction type
            since: Filter by start time

        Returns:
            List of historical interactions
        """
        interactions = self.interaction_history

        if person_id:
            interactions = [i for i in interactions if person_id in i.person_ids]

        if interaction_type:
            interactions = [i for i in interactions if i.interaction_type == interaction_type]

        if since:
            interactions = [i for i in interactions if i.start_time >= since]

        return interactions

    def get_statistics(self) -> Dict[str, any]:
        """Get interaction statistics."""
        stats = {
            'active_interactions': len(self.interactions),
            'total_historical': len(self.interaction_history),
            'active_groups': len(self.groups),
            'by_type': defaultdict(int)
        }

        for interaction in self.interactions.values():
            stats['by_type'][interaction.interaction_type.value] += 1

        return dict(stats)
