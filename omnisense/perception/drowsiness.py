"""
Drowsiness detection system.

Monitors eyelid closure patterns, blink frequency, and head pose stability
to detect signs of drowsiness and fatigue.
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple
from dataclasses import dataclass
import time

from omnisense.utils.logger import get_logger
from omnisense.perception.face_analysis import FaceDetection

logger = get_logger(__name__)


@dataclass
class DrowsinessState:
    """Drowsiness detection state."""
    is_drowsy: bool
    perclos: float  # Percentage of eyelid closure (0-1)
    blink_rate: float  # Blinks per minute
    avg_eye_aspect_ratio: float
    head_stability_score: float
    alert_level: str  # 'normal', 'warning', 'critical'


class DrowsinessDetector:
    """
    Detects drowsiness from facial analysis.

    Uses PERCLOS (percentage of eyelid closure), blink analysis, and
    head pose stability to assess alertness level.
    """

    # EAR threshold for closed eyes
    EAR_THRESHOLD = 0.2

    # PERCLOS thresholds
    PERCLOS_WARNING = 0.15  # 15% closure
    PERCLOS_CRITICAL = 0.30  # 30% closure

    def __init__(
        self,
        window_seconds: float = 60.0,
        fps: float = 30.0
    ):
        """
        Initialize drowsiness detector.

        Args:
            window_seconds: Time window for PERCLOS calculation
            fps: Expected frame rate
        """
        self.window_seconds = window_seconds
        self.fps = fps
        self.window_frames = int(window_seconds * fps)

        # EAR history for PERCLOS calculation
        self.ear_history = deque(maxlen=self.window_frames)

        # Blink detection
        self.blink_count = 0
        self.last_blink_time = 0
        self.blink_times = deque(maxlen=100)
        self.eyes_closed_frames = 0

        # Head pose history for stability analysis
        self.head_pose_history = deque(maxlen=int(5 * fps))  # 5 seconds

        logger.info(
            f"DrowsinessDetector initialized with {window_seconds}s window"
        )

    def update(
        self,
        face_detection: Optional[FaceDetection]
    ) -> DrowsinessState:
        """
        Update drowsiness detection with new face detection.

        Args:
            face_detection: Face detection result with gaze info

        Returns:
            DrowsinessState
        """
        current_time = time.time()

        if face_detection is None or face_detection.gaze is None:
            # No face detected - cannot assess drowsiness
            return self._get_default_state()

        ear = face_detection.gaze.eye_aspect_ratio

        # Update EAR history
        self.ear_history.append(ear)

        # Detect blinks
        if ear < self.EAR_THRESHOLD:
            self.eyes_closed_frames += 1
        else:
            # Eyes opened
            if self.eyes_closed_frames >= 2:  # Minimum frames for blink
                self.blink_count += 1
                self.blink_times.append(current_time)
                logger.debug(f"Blink detected (total: {self.blink_count})")

            self.eyes_closed_frames = 0

        # Update head pose history
        if face_detection.head_pose is not None:
            self.head_pose_history.append((
                face_detection.head_pose.roll,
                face_detection.head_pose.pitch,
                face_detection.head_pose.yaw
            ))

        # Calculate metrics
        perclos = self._calculate_perclos()
        blink_rate = self._calculate_blink_rate(current_time)
        avg_ear = np.mean(self.ear_history) if self.ear_history else 0.0
        head_stability = self._calculate_head_stability()

        # Determine alert level
        alert_level = self._determine_alert_level(perclos, blink_rate, head_stability)
        is_drowsy = alert_level in ['warning', 'critical']

        return DrowsinessState(
            is_drowsy=is_drowsy,
            perclos=perclos,
            blink_rate=blink_rate,
            avg_eye_aspect_ratio=avg_ear,
            head_stability_score=head_stability,
            alert_level=alert_level
        )

    def _calculate_perclos(self) -> float:
        """
        Calculate PERCLOS (Percentage of Eyelid Closure).

        Returns:
            PERCLOS value (0-1)
        """
        if len(self.ear_history) == 0:
            return 0.0

        closed_frames = sum(1 for ear in self.ear_history if ear < self.EAR_THRESHOLD)
        perclos = closed_frames / len(self.ear_history)

        return perclos

    def _calculate_blink_rate(self, current_time: float) -> float:
        """
        Calculate blink rate in blinks per minute.

        Args:
            current_time: Current timestamp

        Returns:
            Blinks per minute
        """
        if len(self.blink_times) < 2:
            return 0.0

        # Count blinks in last minute
        one_minute_ago = current_time - 60.0
        recent_blinks = sum(1 for t in self.blink_times if t >= one_minute_ago)

        return recent_blinks

    def _calculate_head_stability(self) -> float:
        """
        Calculate head pose stability score.

        Returns:
            Stability score (0-1, higher is more stable)
        """
        if len(self.head_pose_history) < 10:
            return 1.0  # Assume stable if insufficient data

        # Calculate variance in head pose angles
        poses = np.array(list(self.head_pose_history))
        roll_var = np.var(poses[:, 0])
        pitch_var = np.var(poses[:, 1])
        yaw_var = np.var(poses[:, 2])

        # Combined variance (normalized)
        total_var = (roll_var + pitch_var + yaw_var) / 3.0

        # Convert to stability score (lower variance = higher stability)
        stability = 1.0 / (1.0 + total_var * 10.0)

        return stability

    def _determine_alert_level(
        self,
        perclos: float,
        blink_rate: float,
        head_stability: float
    ) -> str:
        """
        Determine overall alert level.

        Args:
            perclos: PERCLOS value
            blink_rate: Blink rate (bpm)
            head_stability: Head stability score

        Returns:
            Alert level: 'normal', 'warning', or 'critical'
        """
        # Critical conditions
        if perclos >= self.PERCLOS_CRITICAL:
            return 'critical'

        if blink_rate < 5 and len(self.blink_times) > 10:  # Very low blink rate
            return 'critical'

        # Warning conditions
        if perclos >= self.PERCLOS_WARNING:
            return 'warning'

        if head_stability < 0.5:  # Unstable head (nodding)
            return 'warning'

        if blink_rate < 10 and len(self.blink_times) > 10:  # Low blink rate
            return 'warning'

        return 'normal'

    def _get_default_state(self) -> DrowsinessState:
        """Get default state when no face is detected."""
        return DrowsinessState(
            is_drowsy=False,
            perclos=0.0,
            blink_rate=0.0,
            avg_eye_aspect_ratio=0.0,
            head_stability_score=0.0,
            alert_level='normal'
        )

    def reset(self):
        """Reset detector state."""
        self.ear_history.clear()
        self.blink_count = 0
        self.blink_times.clear()
        self.eyes_closed_frames = 0
        self.head_pose_history.clear()
        logger.info("DrowsinessDetector reset")
