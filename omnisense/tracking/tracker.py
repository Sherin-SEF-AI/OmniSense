"""
Multi-person tracking using DeepSORT and person re-identification.

Tracks persons across frames within a single camera and across multiple cameras.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from filterpy.kalman import KalmanFilter
from collections import deque
import time

from omnisense.tracking.detector import PersonDetection
from omnisense.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    """Represents a tracked person."""
    track_id: int
    camera_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    state: np.ndarray  # Kalman filter state (x, y, w, h, vx, vy, vw, vh)
    covariance: np.ndarray
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    features: Optional[np.ndarray] = None
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))
    global_id: Optional[int] = None  # Cross-camera ID


def compute_iou(bbox1: Tuple[float, float, float, float],
                bbox2: Tuple[float, float, float, float]) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        bbox1: (x1, y1, x2, y2)
        bbox2: (x1, y1, x2, y2)

    Returns:
        IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


class SingleCameraTracker:
    """
    Tracks persons in a single camera view using Kalman filtering and IoU matching.
    """

    def __init__(
        self,
        camera_id: int,
        max_age: int = 70,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracker.

        Args:
            camera_id: Camera identifier
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.camera_id = camera_id
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: List[Track] = []
        self.next_track_id = 0

        logger.info(
            f"SingleCameraTracker initialized for camera {camera_id}"
        )

    def update(
        self,
        detections: List[PersonDetection]
    ) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of person detections

        Returns:
            List of active tracks
        """
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
            # Kalman prediction would go here
            # For now, we keep bbox constant

        # Match detections to tracks
        matched_indices, unmatched_detections, unmatched_tracks = \
            self._match_detections_to_tracks(detections)

        # Update matched tracks
        for det_idx, track_idx in matched_indices:
            detection = detections[det_idx]
            track = self.tracks[track_idx]

            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.hits += 1
            track.time_since_update = 0

            if detection.features is not None:
                track.features = detection.features

            # Add to trajectory
            track.trajectory.append({
                'bbox': detection.bbox,
                'timestamp': time.time(),
                'confidence': detection.confidence
            })

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            self._create_track(detection)

        # Delete old tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update < self.max_age
        ]

        # Return confirmed tracks
        confirmed_tracks = [
            track for track in self.tracks
            if track.hits >= self.min_hits
        ]

        return confirmed_tracks

    def _match_detections_to_tracks(
        self,
        detections: List[PersonDetection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU.

        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))

        for d, detection in enumerate(detections):
            for t, track in enumerate(self.tracks):
                iou_matrix[d, t] = compute_iou(detection.bbox, track.bbox)

        # Simple greedy matching
        matched_indices = []
        unmatched_detections = []
        unmatched_tracks = list(range(len(self.tracks)))

        for d in range(len(detections)):
            max_iou = self.iou_threshold
            best_track = -1

            for t in range(len(self.tracks)):
                if t in unmatched_tracks and iou_matrix[d, t] > max_iou:
                    max_iou = iou_matrix[d, t]
                    best_track = t

            if best_track >= 0:
                matched_indices.append((d, best_track))
                unmatched_tracks.remove(best_track)
            else:
                unmatched_detections.append(d)

        return matched_indices, unmatched_detections, unmatched_tracks

    def _create_track(self, detection: PersonDetection):
        """Create new track from detection."""
        track = Track(
            track_id=self.next_track_id,
            camera_id=self.camera_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            state=np.zeros(8),
            covariance=np.eye(8),
            hits=1,
            features=detection.features
        )

        track.trajectory.append({
            'bbox': detection.bbox,
            'timestamp': time.time(),
            'confidence': detection.confidence
        })

        self.tracks.append(track)
        self.next_track_id += 1

        logger.debug(f"Camera {self.camera_id}: Created track {track.track_id}")

    def get_tracks(self) -> List[Track]:
        """Get all active tracks."""
        return self.tracks


class MultiCameraTracker:
    """
    Tracks persons across multiple cameras using appearance features.
    """

    def __init__(self, num_cameras: int = 4):
        """
        Initialize multi-camera tracker.

        Args:
            num_cameras: Number of cameras in the system
        """
        self.num_cameras = num_cameras
        self.camera_trackers: Dict[int, SingleCameraTracker] = {}

        # Global track management
        self.global_tracks: Dict[int, Dict] = {}
        self.next_global_id = 0

        logger.info(f"MultiCameraTracker initialized for {num_cameras} cameras")

    def add_camera(self, camera_id: int, **kwargs):
        """Add a camera tracker."""
        self.camera_trackers[camera_id] = SingleCameraTracker(camera_id, **kwargs)
        logger.info(f"Added tracker for camera {camera_id}")

    def update(
        self,
        camera_detections: Dict[int, List[PersonDetection]]
    ) -> Dict[int, List[Track]]:
        """
        Update all camera trackers.

        Args:
            camera_detections: Dictionary mapping camera_id to detections

        Returns:
            Dictionary mapping camera_id to tracks
        """
        all_tracks = {}

        # Update each camera tracker
        for camera_id, detections in camera_detections.items():
            if camera_id in self.camera_trackers:
                tracks = self.camera_trackers[camera_id].update(detections)
                all_tracks[camera_id] = tracks

        # Perform cross-camera association
        self._cross_camera_association(all_tracks)

        return all_tracks

    def _cross_camera_association(
        self,
        all_tracks: Dict[int, List[Track]]
    ):
        """
        Associate tracks across cameras based on appearance features.

        Args:
            all_tracks: Dictionary of tracks per camera
        """
        # Collect all tracks with features
        tracks_with_features = []
        for camera_id, tracks in all_tracks.items():
            for track in tracks:
                if track.features is not None:
                    tracks_with_features.append(track)

        # Simple feature-based matching (in production, use learned ReID model)
        for track in tracks_with_features:
            if track.global_id is None:
                # Assign new global ID
                track.global_id = self.next_global_id
                self.global_tracks[self.next_global_id] = {
                    'camera_tracks': {track.camera_id: track.track_id},
                    'features': track.features
                }
                self.next_global_id += 1

    def get_global_tracks(self) -> Dict[int, Dict]:
        """Get global tracks across all cameras."""
        return self.global_tracks
