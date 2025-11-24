"""
Activity Recognition Module for OMNISENSE platform.

Recognizes human activities and behaviors from pose sequences using
temporal models (LSTM/TCN) and rule-based classification.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time

from omnisense.utils.logger import get_logger

logger = get_logger(__name__)


class ActivityType(Enum):
    """Supported activity types."""
    UNKNOWN = "unknown"
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    RUNNING = "running"
    WAVING = "waving"
    POINTING = "pointing"
    REACHING = "reaching"
    BENDING = "bending"
    FALLING = "falling"
    LYING = "lying"
    TYPING = "typing"
    PHONE_USE = "phone_use"
    DRINKING = "drinking"
    CARRYING = "carrying"


@dataclass
class ActivityDetection:
    """Detected activity."""
    activity_type: ActivityType
    confidence: float
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    person_id: Optional[int] = None


@dataclass
class PoseSequence:
    """Sequence of body poses for activity recognition."""
    person_id: int
    keypoints_sequence: List[np.ndarray]  # List of (33, 3) keypoint arrays
    timestamps: List[float]

    def __len__(self):
        return len(self.keypoints_sequence)

    def get_duration(self) -> float:
        """Get sequence duration in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]


class ActivityRecognizer:
    """
    Recognizes human activities from pose sequences.

    Uses both rule-based and learning-based approaches:
    - Rule-based: Posture analysis, geometric constraints
    - Learning: LSTM/TCN models for temporal patterns (when available)
    """

    # Sequence parameters
    SEQUENCE_LENGTH = 30  # Number of frames
    SEQUENCE_STRIDE = 15  # Frame stride for overlapping windows

    # Activity thresholds
    WALKING_SPEED_THRESHOLD = 0.3  # m/s
    RUNNING_SPEED_THRESHOLD = 2.0  # m/s
    MOTION_THRESHOLD = 0.1  # m/s for motion detection

    # Pose thresholds
    STANDING_HIP_KNEE_ANGLE_MIN = 160  # Degrees
    SITTING_HIP_KNEE_ANGLE_MAX = 110  # Degrees
    BENDING_TORSO_ANGLE_MAX = 120  # Degrees

    def __init__(self, use_ml_model: bool = False):
        """
        Initialize activity recognizer.

        Args:
            use_ml_model: Whether to use ML model (requires training)
        """
        self.use_ml_model = use_ml_model

        # Pose sequences for each person
        self.pose_sequences: Dict[int, deque] = {}

        # Current activities for each person
        self.current_activities: Dict[int, ActivityDetection] = {}

        # Activity history
        self.activity_history: Dict[int, List[ActivityDetection]] = {}

        # ML model (if available)
        self.ml_model = None
        if use_ml_model:
            self._load_ml_model()

        logger.info(f"Activity recognizer initialized (ML: {use_ml_model})")

    def _load_ml_model(self):
        """Load pre-trained ML model for activity recognition."""
        try:
            # TODO: Implement ML model loading
            # This would load a pre-trained LSTM/TCN model
            logger.warning("ML-based activity recognition not yet implemented")
            self.use_ml_model = False
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.use_ml_model = False

    def update(self, person_id: int, keypoints: np.ndarray, velocity_3d: np.ndarray,
               timestamp: Optional[float] = None) -> ActivityDetection:
        """
        Update activity recognition for person.

        Args:
            person_id: Person ID
            keypoints: Body keypoints array (33, 3) or (33, 4) with visibility
            velocity_3d: Person's 3D velocity vector
            timestamp: Timestamp (uses current time if None)

        Returns:
            Current activity detection
        """
        if timestamp is None:
            timestamp = time.time()

        # Initialize sequences if needed
        if person_id not in self.pose_sequences:
            self.pose_sequences[person_id] = deque(maxlen=self.SEQUENCE_LENGTH * 2)
            self.activity_history[person_id] = []

        # Extract keypoints (33, 3)
        if keypoints.shape[1] > 3:
            keypoints = keypoints[:, :3]

        # Add to sequence
        self.pose_sequences[person_id].append({
            'keypoints': keypoints,
            'velocity': velocity_3d,
            'timestamp': timestamp
        })

        # Recognize activity
        activity = self._recognize_activity(person_id, velocity_3d)

        # Update current activity
        self._update_activity_tracking(person_id, activity, timestamp)

        return self.current_activities.get(person_id, ActivityDetection(
            activity_type=ActivityType.UNKNOWN,
            confidence=0.0,
            start_time=timestamp,
            person_id=person_id
        ))

    def _recognize_activity(self, person_id: int, velocity_3d: np.ndarray) -> Tuple[ActivityType, float]:
        """
        Recognize activity from pose sequence.

        Args:
            person_id: Person ID
            velocity_3d: Current velocity

        Returns:
            (activity_type, confidence)
        """
        if len(self.pose_sequences[person_id]) < 5:
            return ActivityType.UNKNOWN, 0.0

        sequence = list(self.pose_sequences[person_id])

        # Use ML model if available
        if self.use_ml_model and self.ml_model is not None:
            return self._recognize_ml(sequence)

        # Use rule-based recognition
        return self._recognize_rule_based(sequence, velocity_3d)

    def _recognize_rule_based(self, sequence: List[Dict], velocity_3d: np.ndarray) -> Tuple[ActivityType, float]:
        """
        Rule-based activity recognition.

        Args:
            sequence: Pose sequence
            velocity_3d: Current velocity

        Returns:
            (activity_type, confidence)
        """
        latest = sequence[-1]
        keypoints = latest['keypoints']

        # Compute speed
        speed = np.linalg.norm(velocity_3d)

        # Check for falling (high priority)
        if self._is_falling(sequence):
            return ActivityType.FALLING, 0.95

        # Check for lying
        if self._is_lying(keypoints):
            return ActivityType.LYING, 0.90

        # Check motion-based activities
        if speed > self.RUNNING_SPEED_THRESHOLD:
            return ActivityType.RUNNING, 0.85
        elif speed > self.WALKING_SPEED_THRESHOLD:
            # Check if actually walking vs other motion
            if self._is_walking(sequence):
                return ActivityType.WALKING, 0.80
            else:
                return ActivityType.STANDING, 0.60  # Moving but not walking gait

        # Check posture-based activities (stationary or slow-moving)
        if self._is_sitting(keypoints):
            # Check for specific sitting activities
            if self._is_typing(sequence):
                return ActivityType.TYPING, 0.75
            elif self._is_phone_use(sequence):
                return ActivityType.PHONE_USE, 0.75
            else:
                return ActivityType.SITTING, 0.85

        if self._is_standing(keypoints):
            # Check for standing gestures
            if self._is_waving(sequence):
                return ActivityType.WAVING, 0.80
            elif self._is_pointing(sequence):
                return ActivityType.POINTING, 0.75
            elif self._is_reaching(sequence):
                return ActivityType.REACHING, 0.75
            elif self._is_bending(keypoints):
                return ActivityType.BENDING, 0.80
            else:
                return ActivityType.STANDING, 0.85

        return ActivityType.UNKNOWN, 0.3

    def _is_standing(self, keypoints: np.ndarray) -> bool:
        """Check if person is standing."""
        # Get key joints
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        left_knee = keypoints[25]
        right_knee = keypoints[26]
        left_ankle = keypoints[27]
        right_ankle = keypoints[28]

        # Check leg angles
        left_angle = self._compute_joint_angle(left_hip, left_knee, left_ankle)
        right_angle = self._compute_joint_angle(right_hip, right_knee, right_ankle)

        # Standing: legs relatively straight
        return (left_angle > self.STANDING_HIP_KNEE_ANGLE_MIN and
                right_angle > self.STANDING_HIP_KNEE_ANGLE_MIN)

    def _is_sitting(self, keypoints: np.ndarray) -> bool:
        """Check if person is sitting."""
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        left_knee = keypoints[25]
        right_knee = keypoints[26]
        left_ankle = keypoints[27]
        right_ankle = keypoints[28]

        left_angle = self._compute_joint_angle(left_hip, left_knee, left_ankle)
        right_angle = self._compute_joint_angle(right_hip, right_knee, right_ankle)

        # Sitting: bent knees
        return (left_angle < self.SITTING_HIP_KNEE_ANGLE_MAX and
                right_angle < self.SITTING_HIP_KNEE_ANGLE_MAX)

    def _is_lying(self, keypoints: np.ndarray) -> bool:
        """Check if person is lying down."""
        # Check if torso is nearly horizontal
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        left_hip = keypoints[23]
        right_hip = keypoints[24]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        # Torso vector
        torso_vector = shoulder_center - hip_center

        # Check vertical component (z)
        vertical_ratio = abs(torso_vector[2]) / (np.linalg.norm(torso_vector) + 1e-6)

        # Lying if torso is nearly horizontal
        return vertical_ratio < 0.3

    def _is_bending(self, keypoints: np.ndarray) -> bool:
        """Check if person is bending."""
        nose = keypoints[0]
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        left_knee = keypoints[25]
        right_knee = keypoints[26]

        hip_center = (left_hip + right_hip) / 2
        knee_center = (left_knee + right_knee) / 2

        # Torso angle
        angle = self._compute_joint_angle(nose, hip_center, knee_center)

        return angle < self.BENDING_TORSO_ANGLE_MAX

    def _is_walking(self, sequence: List[Dict]) -> bool:
        """Check if person exhibits walking gait pattern."""
        if len(sequence) < 10:
            return False

        # Analyze leg motion patterns
        left_ankle_positions = []
        right_ankle_positions = []

        for frame in sequence[-10:]:
            kp = frame['keypoints']
            left_ankle_positions.append(kp[27])
            right_ankle_positions.append(kp[28])

        left_ankle_positions = np.array(left_ankle_positions)
        right_ankle_positions = np.array(right_ankle_positions)

        # Compute vertical motion (step pattern)
        left_z_motion = np.std(left_ankle_positions[:, 2])
        right_z_motion = np.std(right_ankle_positions[:, 2])

        # Walking: alternating leg motion
        return (left_z_motion > 0.05 or right_z_motion > 0.05)

    def _is_falling(self, sequence: List[Dict]) -> bool:
        """Detect falling motion."""
        if len(sequence) < 5:
            return False

        # Check for rapid downward motion
        recent_frames = sequence[-5:]

        # Get torso heights
        heights = []
        for frame in recent_frames:
            kp = frame['keypoints']
            left_shoulder = kp[11]
            right_shoulder = kp[12]
            shoulder_height = (left_shoulder[2] + right_shoulder[2]) / 2
            heights.append(shoulder_height)

        heights = np.array(heights)

        # Check for rapid descent
        height_drop = heights[0] - heights[-1]
        descent_rate = height_drop / (recent_frames[-1]['timestamp'] - recent_frames[0]['timestamp'] + 1e-6)

        return descent_rate > 1.0  # Descending faster than 1 m/s

    def _is_waving(self, sequence: List[Dict]) -> bool:
        """Detect waving gesture."""
        if len(sequence) < 15:
            return False

        # Analyze hand motion
        recent = sequence[-15:]

        for hand_idx in [15, 16]:  # Left and right wrist
            positions = np.array([frame['keypoints'][hand_idx] for frame in recent])

            # Check for oscillating motion
            x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
            y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])

            if x_range > 0.3 or y_range > 0.3:  # Significant motion
                # Check if hand is raised
                shoulder = recent[-1]['keypoints'][11 if hand_idx == 15 else 12]
                if positions[-1, 2] > shoulder[2]:  # Hand above shoulder
                    return True

        return False

    def _is_pointing(self, sequence: List[Dict]) -> bool:
        """Detect pointing gesture."""
        if len(sequence) < 5:
            return False

        latest = sequence[-1]['keypoints']

        # Check if arm is extended
        for side in ['left', 'right']:
            if side == 'left':
                shoulder, elbow, wrist = latest[11], latest[13], latest[15]
            else:
                shoulder, elbow, wrist = latest[12], latest[14], latest[16]

            # Compute arm extension
            shoulder_wrist_dist = np.linalg.norm(wrist - shoulder)
            shoulder_elbow_dist = np.linalg.norm(elbow - shoulder)
            elbow_wrist_dist = np.linalg.norm(wrist - elbow)

            # Arm is extended if wrist is far from shoulder and elbow angle is large
            if shoulder_wrist_dist > 0.5:  # Arm extended
                angle = self._compute_joint_angle(shoulder, elbow, wrist)
                if angle > 150:  # Nearly straight
                    return True

        return False

    def _is_reaching(self, sequence: List[Dict]) -> bool:
        """Detect reaching motion."""
        if len(sequence) < 10:
            return False

        # Check for extending arm motion
        recent = sequence[-10:]

        for hand_idx in [15, 16]:  # Wrists
            positions = np.array([frame['keypoints'][hand_idx] for frame in recent])

            # Check if hand is moving forward and upward
            forward_motion = positions[-1, 0] - positions[0, 0]
            upward_motion = positions[-1, 2] - positions[0, 2]

            if abs(forward_motion) > 0.2 and upward_motion > 0.1:
                return True

        return False

    def _is_typing(self, sequence: List[Dict]) -> bool:
        """Detect typing activity."""
        if len(sequence) < 15:
            return False

        # Check for repetitive hand motion in front of body
        recent = sequence[-15:]

        for hand_idx in [15, 16]:  # Wrists
            positions = np.array([frame['keypoints'][hand_idx] for frame in recent])

            # Check for small, repetitive motions
            x_std = np.std(positions[:, 0])
            y_std = np.std(positions[:, 1])

            # Typing: hands in front, small motions
            if 0.02 < x_std < 0.1 and 0.02 < y_std < 0.1:
                return True

        return False

    def _is_phone_use(self, sequence: List[Dict]) -> bool:
        """Detect phone use."""
        if len(sequence) < 5:
            return False

        latest = sequence[-1]['keypoints']

        # Check if hand is near head
        for hand_idx in [15, 16]:  # Wrists
            hand_pos = latest[hand_idx]
            nose_pos = latest[0]

            distance = np.linalg.norm(hand_pos - nose_pos)

            if distance < 0.25:  # Hand near head
                return True

        return False

    def _compute_joint_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute angle at joint p2 formed by p1-p2-p3.

        Returns:
            Angle in degrees
        """
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    def _recognize_ml(self, sequence: List[Dict]) -> Tuple[ActivityType, float]:
        """
        ML-based activity recognition using LSTM/TCN.

        Args:
            sequence: Pose sequence

        Returns:
            (activity_type, confidence)
        """
        # TODO: Implement ML-based recognition
        # This would:
        # 1. Extract features from pose sequence
        # 2. Run through trained LSTM/TCN model
        # 3. Return predicted activity and confidence

        logger.warning("ML-based recognition not implemented, falling back to rule-based")
        return self._recognize_rule_based(sequence, sequence[-1]['velocity'])

    def _update_activity_tracking(self, person_id: int, activity: Tuple[ActivityType, float],
                                  timestamp: float):
        """Update activity tracking for person."""
        activity_type, confidence = activity

        if person_id not in self.current_activities:
            # Start new activity
            self.current_activities[person_id] = ActivityDetection(
                activity_type=activity_type,
                confidence=confidence,
                start_time=timestamp,
                person_id=person_id
            )
        else:
            current = self.current_activities[person_id]

            if current.activity_type != activity_type:
                # Activity changed - end current and start new
                current.end_time = timestamp
                current.duration = timestamp - current.start_time

                # Add to history
                self.activity_history[person_id].append(current)

                # Start new activity
                self.current_activities[person_id] = ActivityDetection(
                    activity_type=activity_type,
                    confidence=confidence,
                    start_time=timestamp,
                    person_id=person_id
                )
            else:
                # Same activity - update confidence and duration
                current.confidence = confidence
                current.duration = timestamp - current.start_time

    def get_current_activity(self, person_id: int) -> Optional[ActivityDetection]:
        """Get current activity for person."""
        return self.current_activities.get(person_id)

    def get_activity_history(self, person_id: int) -> List[ActivityDetection]:
        """Get activity history for person."""
        return self.activity_history.get(person_id, [])

    def get_all_current_activities(self) -> Dict[int, ActivityDetection]:
        """Get all current activities."""
        return self.current_activities.copy()
