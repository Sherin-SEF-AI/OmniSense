"""
Camera Manager for OMNISENSE platform.

Handles interface with multiple USB cameras using OpenCV with V4L2 backend,
including camera initialization, frame capture, and basic control.
"""

import cv2
import numpy as np
import threading
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from queue import Queue, Full
import time

from omnisense.utils.logger import get_logger
from omnisense.utils.exceptions import (
    CameraError, CameraConnectionError, CameraConfigurationError
)
from omnisense.utils.timing import FPSCounter
from omnisense.config.manager import CameraConfig

logger = get_logger(__name__)


@dataclass
class CameraFrame:
    """Container for a captured camera frame with metadata."""
    camera_id: int
    frame: np.ndarray
    timestamp: float
    frame_number: int
    resolution: Tuple[int, int]


class Camera:
    """
    Manages a single camera device.

    Handles video capture, frame buffering, and camera-specific operations.
    """

    def __init__(self, config: CameraConfig):
        """
        Initialize camera.

        Args:
            config: Camera configuration

        Raises:
            CameraConnectionError: If camera cannot be opened
        """
        self.config = config
        self.camera_id = config.camera_id
        self.device_path = config.device_path
        self.name = config.name

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_count = 0

        self.fps_counter = FPSCounter()

        # Frame buffer
        self.frame_buffer: Queue = Queue(maxsize=30)

        # Capture thread
        self.capture_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        logger.info(f"Initializing camera {self.camera_id}: {self.name}")

    def open(self):
        """
        Open camera device and configure it.

        Raises:
            CameraConnectionError: If camera cannot be opened
            CameraConfigurationError: If camera cannot be configured
        """
        try:
            # Try to open using device path
            if self.config.backend == "v4l2":
                self.cap = cv2.VideoCapture(
                    self.device_path,
                    cv2.CAP_V4L2
                )
            else:
                self.cap = cv2.VideoCapture(self.device_path)

            if not self.cap.isOpened():
                raise CameraConnectionError(
                    f"Failed to open camera {self.camera_id} at {self.device_path}"
                )

            # Configure camera
            self._configure_camera()

            # Test read a frame to ensure camera is actually working
            logger.info(f"Testing camera {self.camera_id}...")
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(
                    f"Camera {self.camera_id} opened but cannot read frames. "
                    f"This camera may not work properly."
                )
            else:
                logger.info(
                    f"Camera {self.camera_id} test successful - "
                    f"captured {frame.shape[1]}x{frame.shape[0]} frame"
                )

            logger.info(f"Camera {self.camera_id} opened successfully")

        except Exception as e:
            raise CameraConnectionError(
                f"Error opening camera {self.camera_id}: {e}"
            )

    def _configure_camera(self):
        """Configure camera parameters."""
        try:
            # Set resolution
            width, height = self.config.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            if (actual_width, actual_height) != self.config.resolution:
                logger.warning(
                    f"Camera {self.camera_id}: Requested {self.config.resolution}, "
                    f"got ({actual_width}, {actual_height})"
                )

            if abs(actual_fps - self.config.fps) > 1:
                logger.warning(
                    f"Camera {self.camera_id}: Requested {self.config.fps} FPS, "
                    f"got {actual_fps} FPS"
                )

            logger.info(
                f"Camera {self.camera_id} configured: {actual_width}x{actual_height} @ {actual_fps} FPS"
            )

        except Exception as e:
            raise CameraConfigurationError(
                f"Error configuring camera {self.camera_id}: {e}"
            )

    def start_capture(self):
        """Start continuous frame capture in background thread."""
        if self.is_running:
            logger.warning(f"Camera {self.camera_id} already capturing")
            return

        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"Camera{self.camera_id}Capture",
            daemon=True
        )
        self.capture_thread.start()
        logger.info(f"Started capture thread for camera {self.camera_id}")

    def _capture_loop(self):
        """Main capture loop running in background thread."""
        logger.info(f"Capture loop started for camera {self.camera_id}")

        consecutive_failures = 0
        max_consecutive_failures = 100  # Auto-disable after 100 failures
        last_error_log_time = 0
        error_log_interval = 5.0  # Only log errors every 5 seconds

        while self.is_running:
            try:
                ret, frame = self.cap.read()

                if not ret:
                    consecutive_failures += 1

                    # Rate-limited error logging
                    current_time = time.time()
                    if current_time - last_error_log_time >= error_log_interval:
                        logger.error(
                            f"Failed to read frame from camera {self.camera_id} "
                            f"({consecutive_failures} consecutive failures)"
                        )
                        last_error_log_time = current_time

                    # Auto-disable camera after too many failures
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            f"Camera {self.camera_id} has failed {consecutive_failures} times. "
                            f"Stopping capture thread."
                        )
                        self.is_running = False
                        break

                    time.sleep(0.01)
                    continue

                # Reset failure counter on success
                if consecutive_failures > 0:
                    logger.info(
                        f"Camera {self.camera_id} recovered after {consecutive_failures} failures"
                    )
                    consecutive_failures = 0

                timestamp = time.time()
                self.frame_count += 1

                # Create frame object
                camera_frame = CameraFrame(
                    camera_id=self.camera_id,
                    frame=frame.copy(),
                    timestamp=timestamp,
                    frame_number=self.frame_count,
                    resolution=frame.shape[:2][::-1]
                )

                # Add to buffer, discard oldest if full
                try:
                    self.frame_buffer.put(camera_frame, block=False)
                except Full:
                    # Buffer full, remove oldest and add new
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put(camera_frame, block=False)
                    except:
                        pass

                # Update FPS
                self.fps_counter.update()

            except Exception as e:
                logger.error(f"Error in capture loop for camera {self.camera_id}: {e}")
                time.sleep(0.1)

        logger.info(f"Capture loop ended for camera {self.camera_id}")

    def get_frame(self, timeout: float = 1.0) -> Optional[CameraFrame]:
        """
        Get latest frame from buffer.

        Args:
            timeout: Timeout in seconds

        Returns:
            CameraFrame or None if timeout
        """
        try:
            return self.frame_buffer.get(timeout=timeout)
        except:
            return None

    def stop_capture(self):
        """Stop frame capture."""
        if not self.is_running:
            return

        logger.info(f"Stopping capture for camera {self.camera_id}")
        self.is_running = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

    def close(self):
        """Close camera and release resources."""
        self.stop_capture()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        logger.info(f"Camera {self.camera_id} closed")

    def get_fps(self) -> float:
        """Get current capture FPS."""
        return self.fps_counter.get_fps()

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()


class CameraManager:
    """
    Manages multiple cameras for the OMNISENSE system.

    Handles initialization, capture coordination, and camera lifecycle.
    """

    def __init__(self):
        """Initialize camera manager."""
        self.cameras: Dict[int, Camera] = {}
        self.is_running = False
        logger.info("CameraManager initialized")

    def initialize_cameras(self, camera_configs: Dict[int, CameraConfig]):
        """
        Initialize all cameras from configuration.

        Args:
            camera_configs: Dictionary of camera configurations keyed by camera ID

        Raises:
            CameraError: If any camera fails to initialize
        """
        logger.info(f"Initializing {len(camera_configs)} cameras")

        for camera_id, config in camera_configs.items():
            if not config.enabled:
                logger.info(f"Skipping disabled camera {camera_id}")
                continue

            try:
                camera = Camera(config)
                camera.open()
                self.cameras[camera_id] = camera
                logger.info(f"Camera {camera_id} initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize camera {camera_id}: {e}")
                # Clean up already initialized cameras
                self.close_all()
                raise CameraError(f"Camera initialization failed: {e}")

        logger.info(f"Successfully initialized {len(self.cameras)} cameras")

    def start_all(self):
        """Start capture on all cameras."""
        logger.info("Starting capture on all cameras")

        for camera_id, camera in self.cameras.items():
            try:
                camera.start_capture()
            except Exception as e:
                logger.error(f"Failed to start camera {camera_id}: {e}")

        self.is_running = True
        logger.info("All cameras started")

    def stop_all(self):
        """Stop capture on all cameras."""
        logger.info("Stopping all cameras")

        for camera in self.cameras.values():
            try:
                camera.stop_capture()
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")

        self.is_running = False
        logger.info("All cameras stopped")

    def close_all(self):
        """Close and release all cameras."""
        logger.info("Closing all cameras")

        self.stop_all()

        for camera in self.cameras.values():
            try:
                camera.close()
            except Exception as e:
                logger.error(f"Error closing camera: {e}")

        self.cameras.clear()
        logger.info("All cameras closed")

    def get_camera(self, camera_id: int) -> Optional[Camera]:
        """Get camera by ID."""
        return self.cameras.get(camera_id)

    def get_all_cameras(self) -> Dict[int, Camera]:
        """Get all cameras."""
        return self.cameras.copy()

    def get_camera_count(self) -> int:
        """Get number of active cameras."""
        return len(self.cameras)

    def get_status(self) -> Dict[int, Dict]:
        """
        Get status of all cameras.

        Returns:
            Dictionary with status info for each camera
        """
        status = {}
        for camera_id, camera in self.cameras.items():
            status[camera_id] = {
                'name': camera.name,
                'opened': camera.is_opened(),
                'capturing': camera.is_running,
                'fps': camera.get_fps(),
                'frame_count': camera.frame_count,
            }
        return status
