"""
Person detection using YOLO models.

Detects persons in frames using YOLOv8/v9 for multi-person tracking.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from omnisense.utils.logger import get_logger
from omnisense.utils.exceptions import PersonDetectionError

logger = get_logger(__name__)


@dataclass
class PersonDetection:
    """Person detection result."""
    detection_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # Person class
    features: Optional[np.ndarray] = None  # Appearance features for ReID


class PersonDetector:
    """
    Detects persons in images using YOLO.

    Provides person bounding boxes with confidence scores for tracking.
    """

    def __init__(
        self,
        model,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ):
        """
        Initialize person detector.

        Args:
            model: Loaded YOLO model
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        logger.info(
            f"PersonDetector initialized with confidence={confidence_threshold}, "
            f"nms={nms_threshold}"
        )

    def detect(
        self,
        frame: np.ndarray,
        camera_id: int = 0
    ) -> List[PersonDetection]:
        """
        Detect persons in a frame.

        Args:
            frame: Input frame (BGR)
            camera_id: Camera ID for logging

        Returns:
            List of PersonDetection objects
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                classes=[0],  # Person class only
                verbose=False
            )

            detections = []

            # Process results
            for result in results:
                boxes = result.boxes

                for i, box in enumerate(boxes):
                    # Get box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    # Create detection
                    detection = PersonDetection(
                        detection_id=i,
                        bbox=tuple(xyxy),
                        confidence=conf,
                        class_id=cls
                    )

                    detections.append(detection)

            logger.debug(
                f"Camera {camera_id}: Detected {len(detections)} persons"
            )

            return detections

        except Exception as e:
            logger.error(f"Person detection failed: {e}")
            raise PersonDetectionError(f"Detection failed: {e}")

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[PersonDetection],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detections on frame.

        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Show confidence scores

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence
            if show_confidence:
                label = f"Person {detection.confidence:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        return annotated
