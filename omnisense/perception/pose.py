"""
Full-body pose estimation using MediaPipe Pose.

Extracts body keypoints and analyzes posture for human state understanding.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import mediapipe as mp

from omnisense.utils.logger import get_logger

logger = get_logger(__name__)


class PostureType(Enum):
    """Detected posture types."""
    STANDING = "standing"
    SITTING = "sitting"
    LEANING = "leaning"
    SLOUCHING = "slouching"
    UNKNOWN = "unknown"


@dataclass
class PoseDetection:
    """Pose estimation result."""
    person_id: int
    keypoints: np.ndarray  # 33 keypoints x 3 (x, y, visibility)
    keypoints_3d: np.ndarray  # 33 keypoints x 3 (x, y, z) world coordinates
    bbox: Tuple[int, int, int, int]
    confidence: float
    posture: PostureType
    spine_angle: float  # Degrees
    shoulder_alignment: float  # 0-1, higher is better


class PoseEstimator:
    """
    Estimates full-body pose using MediaPipe Pose.

    Extracts 33 body keypoints and analyzes posture characteristics.
    """

    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def __init__(self, pose_model=None):
        """
        Initialize pose estimator.

        Args:
            pose_model: Pre-loaded MediaPipe Pose model (optional)
        """
        if pose_model is None:
            mp_pose = mp.solutions.pose
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.pose = pose_model

        self.mp_pose = mp.solutions.pose

        logger.info("PoseEstimator initialized")

    def estimate_pose(
        self,
        frame: np.ndarray,
        person_id: int = 0
    ) -> Optional[PoseDetection]:
        """
        Estimate pose in a frame.

        Args:
            frame: Input image (BGR)
            person_id: ID to assign to detected person

        Returns:
            PoseDetection object or None if no person detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Process frame
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # Extract keypoints
        keypoints = np.array([
            [lm.x * w, lm.y * h, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ])

        # Extract 3D keypoints (world coordinates)
        keypoints_3d = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.pose_world_landmarks.landmark
        ]) if results.pose_world_landmarks else np.zeros((33, 3))

        # Compute bounding box
        visible_points = keypoints[keypoints[:, 2] > 0.5][:, :2]
        if len(visible_points) > 0:
            x_min = int(np.min(visible_points[:, 0]))
            y_min = int(np.min(visible_points[:, 1]))
            x_max = int(np.max(visible_points[:, 0]))
            y_max = int(np.max(visible_points[:, 1]))
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        else:
            bbox = (0, 0, w, h)

        # Analyze posture
        posture = self._classify_posture(keypoints)
        spine_angle = self._calculate_spine_angle(keypoints)
        shoulder_alignment = self._calculate_shoulder_alignment(keypoints)

        # Calculate overall confidence
        confidence = np.mean(keypoints[:, 2])

        return PoseDetection(
            person_id=person_id,
            keypoints=keypoints,
            keypoints_3d=keypoints_3d,
            bbox=bbox,
            confidence=confidence,
            posture=posture,
            spine_angle=spine_angle,
            shoulder_alignment=shoulder_alignment
        )

    def _classify_posture(self, keypoints: np.ndarray) -> PostureType:
        """
        Classify body posture from keypoints.

        Args:
            keypoints: Body keypoints with visibility

        Returns:
            PostureType
        """
        # Check if key points are visible
        shoulders_visible = (
            keypoints[self.LEFT_SHOULDER, 2] > 0.5 and
            keypoints[self.RIGHT_SHOULDER, 2] > 0.5
        )
        hips_visible = (
            keypoints[self.LEFT_HIP, 2] > 0.5 and
            keypoints[self.RIGHT_HIP, 2] > 0.5
        )

        if not (shoulders_visible and hips_visible):
            return PostureType.UNKNOWN

        # Get key points
        left_shoulder = keypoints[self.LEFT_SHOULDER, :2]
        right_shoulder = keypoints[self.RIGHT_SHOULDER, :2]
        left_hip = keypoints[self.LEFT_HIP, :2]
        right_hip = keypoints[self.RIGHT_HIP, :2]

        # Calculate torso center points
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        # Calculate torso angle
        torso_vector = shoulder_center - hip_center
        vertical_angle = np.abs(np.degrees(np.arctan2(torso_vector[0], torso_vector[1])))

        # Check for slouching (bent spine)
        spine_angle = self._calculate_spine_angle(keypoints)

        if spine_angle > 30:
            return PostureType.SLOUCHING

        # Check if standing (knees and ankles visible and extended)
        knees_visible = (
            keypoints[self.LEFT_KNEE, 2] > 0.5 and
            keypoints[self.RIGHT_KNEE, 2] > 0.5
        )
        ankles_visible = (
            keypoints[self.LEFT_ANKLE, 2] > 0.5 and
            keypoints[self.RIGHT_ANKLE, 2] > 0.5
        )

        if knees_visible and ankles_visible:
            # Check if legs are extended (standing)
            left_knee = keypoints[self.LEFT_KNEE, :2]
            left_ankle = keypoints[self.LEFT_ANKLE, :2]

            knee_hip_dist = np.linalg.norm(left_knee - left_hip)
            ankle_knee_dist = np.linalg.norm(left_ankle - left_knee)

            # If legs relatively extended, likely standing
            if knee_hip_dist > 0.2 * keypoints[:, 1].max():
                if vertical_angle > 15:
                    return PostureType.LEANING
                else:
                    return PostureType.STANDING

        # Default to sitting if not standing
        return PostureType.SITTING

    def _calculate_spine_angle(self, keypoints: np.ndarray) -> float:
        """
        Calculate spine curvature angle.

        Args:
            keypoints: Body keypoints

        Returns:
            Spine angle in degrees (0 = straight, higher = more bent)
        """
        if (keypoints[self.LEFT_SHOULDER, 2] < 0.5 or
            keypoints[self.RIGHT_SHOULDER, 2] < 0.5 or
            keypoints[self.LEFT_HIP, 2] < 0.5 or
            keypoints[self.RIGHT_HIP, 2] < 0.5):
            return 0.0

        # Get shoulder and hip centers
        shoulder_center = (
            keypoints[self.LEFT_SHOULDER, :2] +
            keypoints[self.RIGHT_SHOULDER, :2]
        ) / 2
        hip_center = (
            keypoints[self.LEFT_HIP, :2] +
            keypoints[self.RIGHT_HIP, :2]
        ) / 2

        # Calculate angle from vertical
        torso_vector = shoulder_center - hip_center
        vertical_vector = np.array([0, -1])  # Pointing up

        cos_angle = np.dot(torso_vector, vertical_vector) / (
            np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector) + 1e-6
        )
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return angle

    def _calculate_shoulder_alignment(self, keypoints: np.ndarray) -> float:
        """
        Calculate shoulder alignment score.

        Args:
            keypoints: Body keypoints

        Returns:
            Alignment score (0-1, 1 = perfectly aligned)
        """
        if (keypoints[self.LEFT_SHOULDER, 2] < 0.5 or
            keypoints[self.RIGHT_SHOULDER, 2] < 0.5):
            return 0.0

        left_shoulder = keypoints[self.LEFT_SHOULDER, :2]
        right_shoulder = keypoints[self.RIGHT_SHOULDER, :2]

        # Calculate shoulder line angle from horizontal
        shoulder_vector = right_shoulder - left_shoulder
        angle_from_horizontal = np.abs(np.degrees(np.arctan2(
            shoulder_vector[1], shoulder_vector[0]
        )))

        # Perfect alignment is 0 degrees, score decreases with deviation
        alignment_score = 1.0 - min(angle_from_horizontal / 30.0, 1.0)

        return alignment_score

    def draw_pose(
        self,
        frame: np.ndarray,
        pose_detection: PoseDetection,
        draw_skeleton: bool = True,
        draw_bbox: bool = True,
        draw_info: bool = True
    ) -> np.ndarray:
        """
        Draw pose estimation on frame.

        Args:
            frame: Input frame
            pose_detection: Pose detection result
            draw_skeleton: Draw skeleton connections
            draw_bbox: Draw bounding box
            draw_info: Draw posture info

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw bounding box
        if draw_bbox:
            x, y, w, h = pose_detection.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw skeleton
        if draw_skeleton:
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose

            # Convert keypoints back to landmark format
            h, w_img = frame.shape[:2]
            landmarks = mp_pose.PoseLandmark

            # Create landmark list
            class LandmarkWrapper:
                def __init__(self, x, y, visibility):
                    self.x = x / w_img
                    self.y = y / h_img
                    self.visibility = visibility

            class LandmarkListWrapper:
                def __init__(self, keypoints):
                    self.landmark = [
                        LandmarkWrapper(kp[0], kp[1], kp[2])
                        for kp in keypoints
                    ]

            landmark_list = LandmarkListWrapper(pose_detection.keypoints)

            mp_drawing.draw_landmarks(
                annotated,
                landmark_list,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
            )

        # Draw info
        if draw_info:
            x, y, _, _ = pose_detection.bbox
            info_text = [
                f"Posture: {pose_detection.posture.value}",
                f"Spine: {pose_detection.spine_angle:.1f}°",
                f"Shoulder: {pose_detection.shoulder_alignment:.2f}"
            ]

            for i, text in enumerate(info_text):
                cv2.putText(
                    annotated,
                    text,
                    (x, y - 10 - i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

        return annotated
