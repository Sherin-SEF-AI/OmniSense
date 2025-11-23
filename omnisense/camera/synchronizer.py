"""
Frame synchronization system for multi-camera capture.

Synchronizes frames from multiple cameras based on timestamps, grouping
frames captured within a specified time window.
"""

import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from queue import Queue
import time

from omnisense.camera.manager import CameraManager, CameraFrame
from omnisense.utils.logger import get_logger
from omnisense.utils.exceptions import FrameSynchronizationError
from omnisense.utils.timing import FPSCounter

logger = get_logger(__name__)


@dataclass
class SynchronizedFrameSet:
    """Set of synchronized frames from all cameras."""
    frames: Dict[int, CameraFrame]  # camera_id -> frame
    sync_timestamp: float  # Average timestamp of all frames
    frame_count: int  # Number of frames in set


class FrameSynchronizer:
    """
    Synchronizes frames from multiple cameras based on timestamps.

    Buffers frames from each camera and groups them into synchronized sets
    where all frames were captured within a specified time window.
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        sync_window_ms: float = 50.0,
        buffer_size: int = 60
    ):
        """
        Initialize frame synchronizer.

        Args:
            camera_manager: Camera manager instance
            sync_window_ms: Time window in milliseconds for frame synchronization
            buffer_size: Maximum frames to buffer per camera
        """
        self.camera_manager = camera_manager
        self.sync_window = sync_window_ms / 1000.0  # Convert to seconds
        self.buffer_size = buffer_size

        # Frame buffers for each camera
        self.frame_buffers: Dict[int, List[CameraFrame]] = {}

        # Synchronized frame output queue
        self.sync_queue: Queue = Queue(maxsize=30)

        # Synchronization thread
        self.is_running = False
        self.sync_thread: Optional[threading.Thread] = None

        # Statistics
        self.fps_counter = FPSCounter()
        self.total_synced = 0
        self.total_dropped = 0

        logger.info(
            f"FrameSynchronizer initialized with {sync_window_ms}ms sync window"
        )

    def start(self):
        """Start frame synchronization."""
        if self.is_running:
            logger.warning("FrameSynchronizer already running")
            return

        # Initialize buffers for all cameras
        for camera_id in self.camera_manager.get_all_cameras().keys():
            self.frame_buffers[camera_id] = []

        self.is_running = True
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            name="FrameSynchronizer",
            daemon=True
        )
        self.sync_thread.start()
        logger.info("FrameSynchronizer started")

    def _sync_loop(self):
        """Main synchronization loop."""
        logger.info("Frame synchronization loop started")

        while self.is_running:
            try:
                # Collect new frames from all cameras
                self._collect_frames()

                # Attempt to create synchronized frame set
                sync_set = self._create_sync_set()

                if sync_set is not None:
                    # Add to output queue
                    try:
                        self.sync_queue.put(sync_set, timeout=0.1)
                        self.total_synced += 1
                        self.fps_counter.update()
                    except:
                        logger.warning("Sync queue full, dropping frame set")
                        self.total_dropped += 1

                # Small sleep to prevent busy waiting
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                time.sleep(0.01)

        logger.info("Frame synchronization loop ended")

    def _collect_frames(self):
        """Collect new frames from all cameras."""
        for camera_id, camera in self.camera_manager.get_all_cameras().items():
            # Try to get frame with short timeout
            frame = camera.get_frame(timeout=0.01)

            if frame is not None:
                # Add to buffer
                buffer = self.frame_buffers[camera_id]
                buffer.append(frame)

                # Trim buffer if too large
                if len(buffer) > self.buffer_size:
                    # Remove oldest frames
                    removed = buffer[:len(buffer) - self.buffer_size]
                    self.total_dropped += len(removed)
                    self.frame_buffers[camera_id] = buffer[-self.buffer_size:]

    def _create_sync_set(self) -> Optional[SynchronizedFrameSet]:
        """
        Attempt to create a synchronized frame set.

        Returns:
            SynchronizedFrameSet if frames are available and synchronized, else None
        """
        # Check if all cameras have at least one frame
        if not all(len(buf) > 0 for buf in self.frame_buffers.values()):
            return None

        # Get oldest frame from each camera
        oldest_frames = {
            cam_id: buf[0]
            for cam_id, buf in self.frame_buffers.items()
        }

        # Find timestamp range
        timestamps = [f.timestamp for f in oldest_frames.values()]
        min_ts = min(timestamps)
        max_ts = max(timestamps)

        # Check if frames are within sync window
        if max_ts - min_ts <= self.sync_window:
            # Frames are synchronized!
            # Remove used frames from buffers
            for cam_id in self.frame_buffers.keys():
                self.frame_buffers[cam_id].pop(0)

            # Calculate average timestamp
            avg_timestamp = sum(timestamps) / len(timestamps)

            return SynchronizedFrameSet(
                frames=oldest_frames,
                sync_timestamp=avg_timestamp,
                frame_count=len(oldest_frames)
            )

        else:
            # Frames not synchronized - discard oldest frame
            # Find camera with oldest frame
            oldest_cam_id = min(
                oldest_frames.keys(),
                key=lambda cid: oldest_frames[cid].timestamp
            )

            # Remove oldest frame from that camera
            dropped_frame = self.frame_buffers[oldest_cam_id].pop(0)
            self.total_dropped += 1

            logger.debug(
                f"Dropped frame from camera {oldest_cam_id} "
                f"(timestamp {dropped_frame.timestamp})"
            )

            return None

    def get_synchronized_frames(
        self,
        timeout: float = 1.0
    ) -> Optional[SynchronizedFrameSet]:
        """
        Get next synchronized frame set.

        Args:
            timeout: Timeout in seconds

        Returns:
            SynchronizedFrameSet or None if timeout
        """
        try:
            return self.sync_queue.get(timeout=timeout)
        except:
            return None

    def stop(self):
        """Stop frame synchronization."""
        if not self.is_running:
            return

        logger.info("Stopping FrameSynchronizer")
        self.is_running = False

        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=2.0)

        logger.info("FrameSynchronizer stopped")

    def get_fps(self) -> float:
        """Get synchronized frame rate."""
        return self.fps_counter.get_fps()

    def get_stats(self) -> Dict:
        """
        Get synchronization statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'fps': self.get_fps(),
            'total_synced': self.total_synced,
            'total_dropped': self.total_dropped,
            'drop_rate': (
                self.total_dropped / (self.total_synced + self.total_dropped)
                if (self.total_synced + self.total_dropped) > 0
                else 0.0
            ),
            'buffer_sizes': {
                cam_id: len(buf)
                for cam_id, buf in self.frame_buffers.items()
            }
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.total_synced = 0
        self.total_dropped = 0
        self.fps_counter.reset()
