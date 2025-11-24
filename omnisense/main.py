"""
OMNISENSE Main Application Entry Point.

Initializes and orchestrates all system components including cameras, perception,
tracking, fusion, GUI, database, and API services.
"""

import sys
import signal
from pathlib import Path
from typing import Optional
import argparse

from PyQt6.QtWidgets import QApplication

from omnisense.config.manager import ConfigManager
from omnisense.camera.manager import CameraManager
from omnisense.camera.synchronizer import FrameSynchronizer
from omnisense.models.manager import ModelManager
from omnisense.tracking.detector import PersonDetector
from omnisense.tracking.tracker import MultiCameraTracker
from omnisense.perception.face_analysis import FaceAnalyzer
from omnisense.perception.pose import PoseEstimator
from omnisense.perception.drowsiness import DrowsinessDetector
from omnisense.fusion.world_state import SensorFusion
from omnisense.database.schema import DatabaseManager
from omnisense.utils.logger import setup_logger, get_logger
from omnisense.utils.exceptions import OmniSenseError

logger = get_logger(__name__)


class OmniSenseApplication:
    """
    Main OMNISENSE application class.

    Manages initialization, execution, and shutdown of all system components.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize OMNISENSE application.

        Args:
            config_path: Path to configuration file
        """
        logger.info("=" * 60)
        logger.info("OMNISENSE Multi-Camera Spatial Intelligence Platform")
        logger.info("=" * 60)

        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load(config_path)

        logger.info(f"Loaded configuration profile: {self.config.profile_name}")

        # Initialize components
        self.camera_manager: Optional[CameraManager] = None
        self.frame_synchronizer: Optional[FrameSynchronizer] = None
        self.model_manager: Optional[ModelManager] = None
        self.person_detector: Optional[PersonDetector] = None
        self.multi_camera_tracker: Optional[MultiCameraTracker] = None
        self.face_analyzer: Optional[FaceAnalyzer] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        self.drowsiness_detector: Optional[DrowsinessDetector] = None
        self.sensor_fusion: Optional[SensorFusion] = None
        self.database: Optional[DatabaseManager] = None
        self.qt_app: Optional[QApplication] = None
        self.main_window = None

        self.is_running = False

    def initialize(self):
        """Initialize all system components."""
        logger.info("Initializing system components...")

        try:
            # Initialize model manager
            logger.info("Initializing model manager...")
            self.model_manager = ModelManager(self.config.models_dir)

            # Initialize camera manager
            if self.config.cameras:
                logger.info("Initializing camera manager...")
                self.camera_manager = CameraManager()
                self.camera_manager.initialize_cameras(self.config.cameras)

                # Initialize frame synchronizer
                logger.info("Initializing frame synchronizer...")
                self.frame_synchronizer = FrameSynchronizer(
                    self.camera_manager,
                    sync_window_ms=50.0
                )

            # Initialize tracking components
            if self.config.tracking.enabled:
                logger.info("Loading detection model...")
                yolo_model = self.model_manager.load_yolo_model(
                    self.config.tracking.yolo_model
                )

                logger.info("Initializing person detector...")
                self.person_detector = PersonDetector(
                    yolo_model,
                    confidence_threshold=self.config.tracking.detection_confidence,
                    nms_threshold=self.config.tracking.nms_threshold
                )

                logger.info("Initializing multi-camera tracker...")
                self.multi_camera_tracker = MultiCameraTracker(
                    num_cameras=len(self.config.cameras)
                )

                # Add camera trackers
                for camera_id in self.config.cameras.keys():
                    self.multi_camera_tracker.add_camera(
                        camera_id,
                        max_age=self.config.tracking.max_age,
                        min_hits=self.config.tracking.min_hits,
                        iou_threshold=self.config.tracking.iou_threshold
                    )

            # Initialize perception modules
            if self.config.perception.enabled:
                if self.config.perception.face_detection_enabled:
                    logger.info("Initializing face analyzer...")
                    face_mesh = self.model_manager.load_mediapipe_face_mesh()
                    self.face_analyzer = FaceAnalyzer(face_mesh)

                if self.config.perception.pose_estimation_enabled:
                    logger.info("Initializing pose estimator...")
                    pose_model = self.model_manager.load_mediapipe_pose()
                    self.pose_estimator = PoseEstimator(pose_model)

                if self.config.perception.drowsiness_detection_enabled:
                    logger.info("Initializing drowsiness detector...")
                    self.drowsiness_detector = DrowsinessDetector(
                        window_seconds=self.config.perception.perclos_window_seconds
                    )

            # Initialize sensor fusion
            if self.config.fusion.enabled:
                logger.info("Initializing sensor fusion...")
                self.sensor_fusion = SensorFusion()

            # Initialize database
            if self.config.database.enabled:
                try:
                    logger.info("Initializing database...")

                    # Get password from environment variable if not in config
                    import os
                    db_password = self.config.database.password
                    if not db_password:
                        db_password = os.getenv('OMNISENSE_DB_PASSWORD', '')

                    if not db_password:
                        logger.warning(
                            "No database password configured. "
                            "Set OMNISENSE_DB_PASSWORD environment variable or disable database in config."
                        )
                        logger.info("Continuing without database...")
                        self.config.database.enabled = False
                    else:
                        db_url = (
                            f"postgresql://{self.config.database.user}:"
                            f"{db_password}@"
                            f"{self.config.database.host}:{self.config.database.port}/"
                            f"{self.config.database.database}"
                        )

                        self.database = DatabaseManager(db_url)
                        self.database.create_tables()

                        try:
                            self.database.create_hypertables()
                        except Exception as e:
                            logger.warning(f"TimescaleDB not available: {e}")

                        logger.info("Database initialized successfully")

                except Exception as e:
                    logger.error(f"Database initialization failed: {e}")
                    logger.warning("Continuing without database...")
                    self.config.database.enabled = False
                    self.database = None

            # Initialize GUI
            if self.config.gui.enabled:
                logger.info("Initializing GUI...")
                from omnisense.gui.main_window import OmniSenseMainWindow

                self.qt_app = QApplication.instance()
                if self.qt_app is None:
                    self.qt_app = QApplication(sys.argv)

                self.main_window = OmniSenseMainWindow(self)

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            raise OmniSenseError(f"Failed to initialize OMNISENSE: {e}")

    def start(self):
        """Start the OMNISENSE system."""
        logger.info("Starting OMNISENSE system...")

        try:
            # Start cameras
            if self.camera_manager:
                self.camera_manager.start_all()

            # Start frame synchronizer
            if self.frame_synchronizer:
                self.frame_synchronizer.start()

            self.is_running = True

            # Show GUI and enter event loop
            if self.main_window:
                self.main_window.show()
                self.qt_app.exec()
            else:
                # Run in headless mode
                self._run_headless()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
        finally:
            self.shutdown()

    def _run_headless(self):
        """Run in headless mode without GUI."""
        import time

        logger.info("Running in headless mode. Press Ctrl+C to stop.")

        try:
            while self.is_running:
                # Get synchronized frames
                if self.frame_synchronizer:
                    sync_set = self.frame_synchronizer.get_synchronized_frames(timeout=1.0)

                    if sync_set:
                        # Process frames
                        self._process_frame_set(sync_set)

                time.sleep(0.001)

        except KeyboardInterrupt:
            logger.info("Stopping...")

    def _process_frame_set(self, sync_set):
        """Process a synchronized frame set."""
        # Detect persons in each camera
        detections_per_camera = {}

        if self.person_detector:
            for camera_id, frame_obj in sync_set.frames.items():
                detections = self.person_detector.detect(frame_obj.frame, camera_id)
                detections_per_camera[camera_id] = detections

        # Update tracker
        if self.multi_camera_tracker and detections_per_camera:
            tracks_per_camera = self.multi_camera_tracker.update(detections_per_camera)

            # Update sensor fusion
            if self.sensor_fusion:
                world_state = self.sensor_fusion.update(tracks_per_camera)

                # Log stats
                logger.debug(
                    f"Tracked {world_state.get_person_count()} persons | "
                    f"Sync FPS: {self.frame_synchronizer.get_fps():.1f}"
                )

    def shutdown(self):
        """Shutdown the OMNISENSE system."""
        logger.info("Shutting down OMNISENSE system...")

        self.is_running = False

        # Stop frame synchronizer
        if self.frame_synchronizer:
            self.frame_synchronizer.stop()

        # Stop cameras
        if self.camera_manager:
            self.camera_manager.close_all()

        # Clear model cache
        if self.model_manager:
            self.model_manager.clear_cache()

        logger.info("OMNISENSE shutdown complete")

    def get_status(self) -> dict:
        """
        Get system status.

        Returns:
            Dictionary with system status information
        """
        status = {
            'running': self.is_running,
            'cameras': {},
            'tracking': {},
            'fusion': {},
        }

        if self.camera_manager:
            status['cameras'] = self.camera_manager.get_status()

        if self.frame_synchronizer:
            status['sync_fps'] = self.frame_synchronizer.get_fps()
            status['sync_stats'] = self.frame_synchronizer.get_stats()

        if self.sensor_fusion:
            status['person_count'] = self.sensor_fusion.get_person_count()

        return status


def main():
    """Main entry point for OMNISENSE application."""
    parser = argparse.ArgumentParser(
        description='OMNISENSE Multi-Camera Spatial Intelligence Platform'
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=Path('configs/default/omnisense_config.yaml'),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without GUI'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    import logging
    log_level = getattr(logging, args.log_level)
    setup_logger("omnisense", level=log_level)

    # Create and run application
    try:
        app = OmniSenseApplication(config_path=args.config)

        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal")
            app.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Initialize and start
        app.initialize()
        app.start()

    except Exception as e:
        logger.error(f"OMNISENSE failed to start: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
