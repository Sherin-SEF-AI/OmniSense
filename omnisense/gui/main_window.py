"""
PyQt6 Main Window for OMNISENSE platform.

Provides multi-pane interface with camera views, 3D visualization,
analytics dashboard, and system controls.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QTextEdit, QTabWidget,
    QSlider, QComboBox, QStatusBar, QMenuBar, QMenu, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QAction
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLLinePlotItem

from omnisense.utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingWorker(QThread):
    """Worker thread for processing frames."""

    frame_processed = pyqtSignal(dict)  # Emits processed frame data

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.running = False

    def run(self):
        """Main processing loop."""
        self.running = True

        while self.running:
            try:
                if self.app.frame_synchronizer:
                    sync_set = self.app.frame_synchronizer.get_synchronized_frames(timeout=0.1)

                    if sync_set:
                        # Process frames
                        processed_data = self._process_frames(sync_set)
                        self.frame_processed.emit(processed_data)

            except Exception as e:
                logger.error(f"Processing error: {e}")

            self.msleep(10)  # Small delay

    def _process_frames(self, sync_set):
        """Process synchronized frame set."""
        processed = {
            'frames': {},
            'detections': {},
            'tracks': {},
            'world_state': None
        }

        # Store frames
        for camera_id, frame_obj in sync_set.frames.items():
            processed['frames'][camera_id] = frame_obj.frame.copy()

        # Detect persons
        if self.app.person_detector:
            for camera_id, frame_obj in sync_set.frames.items():
                detections = self.app.person_detector.detect(frame_obj.frame, camera_id)
                processed['detections'][camera_id] = detections

        # Update tracker
        if self.app.multi_camera_tracker and processed['detections']:
            tracks = self.app.multi_camera_tracker.update(processed['detections'])
            processed['tracks'] = tracks

        # Update fusion
        if self.app.sensor_fusion and processed['tracks']:
            world_state = self.app.sensor_fusion.update(processed['tracks'])
            processed['world_state'] = world_state

        return processed

    def stop(self):
        """Stop processing."""
        self.running = False


class OmniSenseMainWindow(QMainWindow):
    """
    Main application window for OMNISENSE platform.

    Provides multi-pane interface with camera views, 3D visualization,
    analytics, and controls.
    """

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWindowTitle("OMNISENSE - Multi-Camera Spatial Intelligence Platform")

        # Processing worker
        self.worker = None

        # Initialize UI
        self._init_ui()

        # Start update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_ui)
        self.update_timer.start(100)  # 10 Hz UI update

        logger.info("Main window initialized")

    def _init_ui(self):
        """Initialize user interface."""
        # Set window size
        self.resize(
            self.app.config.gui.window_width,
            self.app.config.gui.window_height
        )

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Camera views
        left_panel = self._create_camera_panel()
        splitter.addWidget(left_panel)

        # Right panel: Analytics and controls
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        # Set initial sizes
        splitter.setSizes([1200, 720])

        main_layout.addWidget(splitter)

        # Create menu bar
        self._create_menu_bar()

        # Create status bar
        self._create_status_bar()

    def _create_camera_panel(self) -> QWidget:
        """Create camera view panel with 2x2 grid."""
        panel = QWidget()
        layout = QGridLayout(panel)

        # Camera view labels
        self.camera_labels = {}

        rows, cols = self.app.config.gui.camera_grid_layout

        for i, camera_id in enumerate(sorted(self.app.config.cameras.keys())):
            row = i // cols
            col = i % cols

            # Create group box for camera
            group = QGroupBox(f"Camera {camera_id}")
            group_layout = QVBoxLayout()

            # Camera view label
            label = QLabel()
            label.setMinimumSize(640, 480)
            label.setScaledContents(True)
            label.setStyleSheet("border: 1px solid gray;")

            group_layout.addWidget(label)

            # Camera info label
            info_label = QLabel("Initializing...")
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            group_layout.addWidget(info_label)

            group.setLayout(group_layout)
            layout.addWidget(group, row, col)

            self.camera_labels[camera_id] = {
                'view': label,
                'info': info_label
            }

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create right panel with tabs for analytics and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create tab widget
        tabs = QTabWidget()

        # Analytics tab
        analytics_tab = self._create_analytics_tab()
        tabs.addTab(analytics_tab, "Analytics")

        # 3D Visualization tab
        viz_tab = self._create_3d_viz_tab()
        tabs.addTab(viz_tab, "3D View")

        # Controls tab
        controls_tab = self._create_controls_tab()
        tabs.addTab(controls_tab, "Controls")

        # Log tab
        log_tab = self._create_log_tab()
        tabs.addTab(log_tab, "Logs")

        layout.addWidget(tabs)

        return panel

    def _create_analytics_tab(self) -> QWidget:
        """Create analytics dashboard tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Person count
        person_group = QGroupBox("Person Tracking")
        person_layout = QVBoxLayout()

        self.person_count_label = QLabel("Persons Detected: 0")
        self.person_count_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        person_layout.addWidget(self.person_count_label)

        person_group.setLayout(person_layout)
        layout.addWidget(person_group)

        # Performance metrics
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout()

        self.fps_label = QLabel("FPS: 0.0")
        self.sync_fps_label = QLabel("Sync FPS: 0.0")
        self.processing_time_label = QLabel("Processing: 0 ms")

        perf_layout.addWidget(self.fps_label)
        perf_layout.addWidget(self.sync_fps_label)
        perf_layout.addWidget(self.processing_time_label)

        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        # Alerts
        alert_group = QGroupBox("Alerts")
        alert_layout = QVBoxLayout()

        self.alert_text = QTextEdit()
        self.alert_text.setReadOnly(True)
        self.alert_text.setMaximumHeight(200)
        alert_layout.addWidget(self.alert_text)

        alert_group.setLayout(alert_layout)
        layout.addWidget(alert_group)

        layout.addStretch()

        return widget

    def _create_3d_viz_tab(self) -> QWidget:
        """Create 3D visualization tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 3D view using PyQtGraph
        self.gl_view = GLViewWidget()
        self.gl_view.setCameraPosition(distance=10)
        layout.addWidget(self.gl_view)

        # Person scatter plot
        self.person_scatter = GLScatterPlotItem()
        self.gl_view.addItem(self.person_scatter)

        return widget

    def _create_controls_tab(self) -> QWidget:
        """Create controls tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Start/Stop controls
        control_group = QGroupBox("System Control")
        control_layout = QVBoxLayout()

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self._start_processing)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self._stop_processing)
        self.stop_button.setEnabled(False)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Detection settings
        det_group = QGroupBox("Detection Settings")
        det_layout = QVBoxLayout()

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("Confidence: 0.50")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"Confidence: {v/100:.2f}")
        )

        det_layout.addWidget(self.conf_label)
        det_layout.addWidget(self.conf_slider)

        det_group.setLayout(det_layout)
        layout.addWidget(det_group)

        layout.addStretch()

        return widget

    def _create_log_tab(self) -> QWidget:
        """Create log display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        return widget

    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        calib_action = QAction("Camera Calibration", self)
        tools_menu.addAction(calib_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        """Create status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_bar.showMessage("Ready")

    def _start_processing(self):
        """Start frame processing."""
        if self.worker is None:
            self.worker = ProcessingWorker(self.app)
            self.worker.frame_processed.connect(self._on_frame_processed)
            self.worker.start()

            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_bar.showMessage("Processing started")

            logger.info("Processing started from GUI")

    def _stop_processing(self):
        """Stop frame processing."""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_bar.showMessage("Processing stopped")

            logger.info("Processing stopped from GUI")

    def _on_frame_processed(self, data: dict):
        """Handle processed frame data."""
        # Update camera views
        for camera_id, frame in data['frames'].items():
            if camera_id in self.camera_labels:
                # Draw detections on frame
                if camera_id in data.get('detections', {}):
                    detections = data['detections'][camera_id]
                    if self.app.person_detector:
                        frame = self.app.person_detector.draw_detections(frame, detections)

                # Convert frame to QPixmap
                pixmap = self._frame_to_pixmap(frame)
                self.camera_labels[camera_id]['view'].setPixmap(pixmap)

        # Update 3D visualization
        if data.get('world_state'):
            self._update_3d_viz(data['world_state'])

    def _frame_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """Convert OpenCV frame to QPixmap."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(
            rgb_frame.data,
            w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        return QPixmap.fromImage(qt_image)

    def _update_3d_viz(self, world_state):
        """Update 3D visualization with world state."""
        persons = world_state.get_all_persons()

        if persons:
            positions = np.array([p.position_3d for p in persons])

            self.person_scatter.setData(
                pos=positions,
                color=(1.0, 0.5, 0.0, 1.0),
                size=10
            )

    def _update_ui(self):
        """Periodic UI updates."""
        # Update status
        status = self.app.get_status()

        # Update camera info
        if 'cameras' in status:
            for camera_id, cam_status in status['cameras'].items():
                if camera_id in self.camera_labels:
                    info = f"FPS: {cam_status['fps']:.1f} | Frames: {cam_status['frame_count']}"
                    self.camera_labels[camera_id]['info'].setText(info)

        # Update analytics
        if 'person_count' in status:
            self.person_count_label.setText(f"Persons Detected: {status['person_count']}")

        if 'sync_fps' in status:
            self.sync_fps_label.setText(f"Sync FPS: {status['sync_fps']:.1f}")

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Main window closing")

        # Stop processing
        if self.worker:
            self.worker.stop()
            self.worker.wait()

        event.accept()
