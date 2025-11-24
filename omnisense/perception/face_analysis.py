"""
Face analysis module using MediaPipe Face Mesh.

Implements face detection, facial landmark localization, head pose estimation,
and eye gaze estimation for human state analysis.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import mediapipe as mp

from omnisense.utils.logger import get_logger
from omnisense.utils.geometry import rotation_matrix_to_euler
from omnisense.utils.exceptions import PoseEstimationError, GazeEstimationError

logger = get_logger(__name__)


@dataclass
class FaceDetection:
    """Face detection result with landmarks and pose."""
    face_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: np.ndarray  # 468 3D landmarks
    landmarks_2d: np.ndarray  # 2D image coordinates
    confidence: float
    head_pose: Optional['HeadPose'] = None
    gaze: Optional['GazeEstimate'] = None


@dataclass
class HeadPose:
    """Head pose estimation result."""
    roll: float  # Rotation around z-axis (radians)
    pitch: float  # Rotation around x-axis (radians)
    yaw: float  # Rotation around y-axis (radians)
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray


@dataclass
class GazeEstimate:
    """Gaze estimation result."""
    gaze_vector: np.ndarray  # 3D normalized gaze direction
    gaze_origin: np.ndarray  # 3D gaze ray origin
    left_eye_center: np.ndarray
    right_eye_center: np.ndarray
    eye_aspect_ratio: float  # For drowsiness detection


class FaceAnalyzer:
    """
    Analyzes faces using MediaPipe Face Mesh.

    Detects faces, extracts landmarks, estimates head pose and gaze direction.
    """

    # Face mesh landmark indices for specific features
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    NOSE_TIP_INDEX = 1
    CHIN_INDEX = 152
    LEFT_EYE_CORNER_INDEX = 33
    RIGHT_EYE_CORNER_INDEX = 263
    LEFT_MOUTH_CORNER_INDEX = 61
    RIGHT_MOUTH_CORNER_INDEX = 291

    # 3D model points for head pose estimation (canonical face model)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype=np.float64)

    def __init__(self, face_mesh_model=None):
        """
        Initialize face analyzer.

        Args:
            face_mesh_model: Pre-loaded MediaPipe Face Mesh model (optional)
        """
        if face_mesh_model is None:
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = face_mesh_model

        logger.info("FaceAnalyzer initialized")

    def analyze_frame(
        self,
        frame: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None
    ) -> List[FaceDetection]:
        """
        Analyze a frame for faces.

        Args:
            frame: Input image (BGR)
            camera_matrix: Camera intrinsic matrix for pose estimation
            dist_coeffs: Camera distortion coefficients

        Returns:
            List of FaceDetection objects
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Process frame
        results = self.face_mesh.process(rgb_frame)

        detections = []

        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                # Extract landmarks
                landmarks_3d = np.array([
                    [lm.x * w, lm.y * h, lm.z * w]
                    for lm in face_landmarks.landmark
                ])

                landmarks_2d = landmarks_3d[:, :2]

                # Compute bounding box
                x_min = int(np.min(landmarks_2d[:, 0]))
                y_min = int(np.min(landmarks_2d[:, 1]))
                x_max = int(np.max(landmarks_2d[:, 0]))
                y_max = int(np.max(landmarks_2d[:, 1]))

                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                # Create detection object
                detection = FaceDetection(
                    face_id=face_id,
                    bbox=bbox,
                    landmarks=landmarks_3d,
                    landmarks_2d=landmarks_2d,
                    confidence=1.0  # MediaPipe doesn't provide confidence per face
                )

                # Estimate head pose if camera matrix provided
                if camera_matrix is not None:
                    try:
                        detection.head_pose = self._estimate_head_pose(
                            landmarks_2d,
                            camera_matrix,
                            dist_coeffs
                        )
                    except Exception as e:
                        logger.warning(f"Head pose estimation failed: {e}")

                # Estimate gaze
                try:
                    detection.gaze = self._estimate_gaze(landmarks_3d, landmarks_2d)
                except Exception as e:
                    logger.warning(f"Gaze estimation failed: {e}")

                detections.append(detection)

        return detections

    def _estimate_head_pose(
        self,
        landmarks_2d: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray]
    ) -> HeadPose:
        """
        Estimate head pose using PnP algorithm.

        Args:
            landmarks_2d: 2D facial landmarks
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients

        Returns:
            HeadPose object
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))

        # Select key landmarks for pose estimation
        image_points = np.array([
            landmarks_2d[self.NOSE_TIP_INDEX],
            landmarks_2d[self.CHIN_INDEX],
            landmarks_2d[self.LEFT_EYE_CORNER_INDEX],
            landmarks_2d[self.RIGHT_EYE_CORNER_INDEX],
            landmarks_2d[self.LEFT_MOUTH_CORNER_INDEX],
            landmarks_2d[self.RIGHT_MOUTH_CORNER_INDEX]
        ], dtype=np.float64)

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            raise PoseEstimationError("PnP solving failed")

        # Convert rotation vector to matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Convert to Euler angles
        roll, pitch, yaw = rotation_matrix_to_euler(rotation_matrix)

        return HeadPose(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector
        )

    def _estimate_gaze(
        self,
        landmarks_3d: np.ndarray,
        landmarks_2d: np.ndarray
    ) -> GazeEstimate:
        """
        Estimate gaze direction from eye landmarks.

        Args:
            landmarks_3d: 3D facial landmarks
            landmarks_2d: 2D facial landmarks

        Returns:
            GazeEstimate object
        """
        # Get eye landmarks
        left_eye_3d = landmarks_3d[self.LEFT_EYE_INDICES]
        right_eye_3d = landmarks_3d[self.RIGHT_EYE_INDICES]

        # Compute eye centers
        left_eye_center = np.mean(left_eye_3d, axis=0)
        right_eye_center = np.mean(right_eye_3d, axis=0)

        # Simplified gaze estimation: use vector from eye center to nose tip
        nose_tip = landmarks_3d[self.NOSE_TIP_INDEX]

        # Average gaze direction (simplified - in production use learned model)
        gaze_origin = (left_eye_center + right_eye_center) / 2
        gaze_vector = nose_tip - gaze_origin
        gaze_vector = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-6)

        # Compute Eye Aspect Ratio (EAR) for drowsiness detection
        ear = self._compute_eye_aspect_ratio(landmarks_2d)

        return GazeEstimate(
            gaze_vector=gaze_vector,
            gaze_origin=gaze_origin,
            left_eye_center=left_eye_center,
            right_eye_center=right_eye_center,
            eye_aspect_ratio=ear
        )

    def _compute_eye_aspect_ratio(self, landmarks_2d: np.ndarray) -> float:
        """
        Compute Eye Aspect Ratio (EAR) for drowsiness detection.

        Args:
            landmarks_2d: 2D facial landmarks

        Returns:
            Eye aspect ratio (higher = more open)
        """
        def eye_aspect_ratio(eye_points):
            # Compute vertical distances
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            # Compute horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            # EAR
            ear = (v1 + v2) / (2.0 * h + 1e-6)
            return ear

        # Get left and right eye points
        left_eye = landmarks_2d[self.LEFT_EYE_INDICES]
        right_eye = landmarks_2d[self.RIGHT_EYE_INDICES]

        # Average EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        return (left_ear + right_ear) / 2.0

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[FaceDetection],
        draw_landmarks: bool = True,
        draw_bbox: bool = True,
        draw_pose: bool = True,
        draw_gaze: bool = True
    ) -> np.ndarray:
        """
        Draw face detections on frame for visualization.

        Args:
            frame: Input frame
            detections: List of face detections
            draw_landmarks: Draw facial landmarks
            draw_bbox: Draw bounding boxes
            draw_pose: Draw head pose axes
            draw_gaze: Draw gaze direction

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for detection in detections:
            # Draw bounding box
            if draw_bbox:
                x, y, w, h = detection.bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"Face {detection.face_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # Draw landmarks
            if draw_landmarks:
                for point in detection.landmarks_2d:
                    cv2.circle(annotated, (int(point[0]), int(point[1])), 1, (0, 255, 255), -1)

            # Draw head pose
            if draw_pose and detection.head_pose is not None:
                # Draw pose axes (simplified)
                nose_tip = detection.landmarks_2d[self.NOSE_TIP_INDEX].astype(int)
                cv2.putText(
                    annotated,
                    f"Yaw: {np.degrees(detection.head_pose.yaw):.1f}",
                    (nose_tip[0], nose_tip[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1
                )

            # Draw gaze
            if draw_gaze and detection.gaze is not None:
                origin = detection.gaze.gaze_origin[:2].astype(int)
                gaze_end = (origin + detection.gaze.gaze_vector[:2] * 100).astype(int)
                cv2.arrowedLine(annotated, tuple(origin), tuple(gaze_end), (255, 0, 255), 2, tipLength=0.3)

        return annotated
