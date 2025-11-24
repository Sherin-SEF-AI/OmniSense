"""
Configuration management system for OMNISENSE platform.

Handles loading, validation, and runtime access to configuration parameters
from YAML files with support for hierarchical configuration and schema validation.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import json
from dataclasses import dataclass, field, asdict
from omnisense.utils.logger import get_logger
from omnisense.utils.exceptions import ConfigurationError, ConfigValidationError, ConfigFileError

logger = get_logger(__name__)


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    camera_id: int
    device_path: str
    name: str
    resolution: tuple = (640, 480)
    fps: int = 30
    backend: str = "v4l2"
    calibration_file: Optional[str] = None
    enabled: bool = True


@dataclass
class SLAMConfig:
    """Configuration for SLAM system."""
    enabled: bool = True
    stereo_camera_pair: tuple = (1, 2)  # Camera indices for stereo SLAM
    orb_features: int = 2000
    scale_factor: float = 1.2
    n_levels: int = 8
    loop_closure_enabled: bool = True
    relocalization_enabled: bool = True
    map_save_path: str = "data/maps"


@dataclass
class TrackingConfig:
    """Configuration for person tracking."""
    enabled: bool = True
    yolo_model: str = "yolov8n.pt"
    detection_confidence: float = 0.5
    nms_threshold: float = 0.4
    tracker_type: str = "deep_sort"
    max_age: int = 70
    min_hits: int = 3
    iou_threshold: float = 0.3
    reid_model: str = "osnet_x1_0"
    reid_confidence: float = 0.6


@dataclass
class PerceptionConfig:
    """Configuration for human state analysis."""
    enabled: bool = True
    face_detection_enabled: bool = True
    pose_estimation_enabled: bool = True
    gaze_estimation_enabled: bool = True
    drowsiness_detection_enabled: bool = True
    mediapipe_model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    gaze_model: str = "eth_xgaze"
    perclos_window_seconds: float = 60.0
    drowsiness_threshold: float = 0.3


@dataclass
class FusionConfig:
    """Configuration for sensor fusion."""
    enabled: bool = True
    kalman_filter_enabled: bool = True
    process_noise: float = 0.01
    measurement_noise: float = 0.1
    triangulation_min_views: int = 2
    triangulation_max_reprojection_error: float = 5.0
    temporal_smoothing_alpha: float = 0.3


@dataclass
class IntelligenceConfig:
    """Configuration for intelligence layer."""
    activity_recognition_enabled: bool = True
    anomaly_detection_enabled: bool = True
    interaction_detection_enabled: bool = True
    risk_assessment_enabled: bool = True
    attention_heatmap_enabled: bool = True
    zone_detection_enabled: bool = True


@dataclass
class DatabaseConfig:
    """Configuration for database."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 5432
    database: str = "omnisense"
    user: str = "omnisense"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    retention_days: int = 30


@dataclass
class APIConfig:
    """Configuration for API services."""
    rest_api_enabled: bool = True
    rest_api_host: str = "0.0.0.0"
    rest_api_port: int = 8000
    websocket_enabled: bool = True
    websocket_port: int = 8001
    ros2_enabled: bool = False
    mqtt_enabled: bool = False
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    jwt_secret: str = "change_this_secret"
    jwt_expiration_hours: int = 24


@dataclass
class GUIConfig:
    """Configuration for GUI."""
    enabled: bool = True
    window_width: int = 1920
    window_height: int = 1080
    fullscreen: bool = False
    theme: str = "dark"
    show_fps: bool = True
    show_diagnostics: bool = True
    camera_grid_layout: tuple = (2, 2)
    visualization_3d_enabled: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""
    gpu_enabled: bool = True
    gpu_device_id: int = 0
    num_threads: int = 4
    batch_size: int = 1
    inference_precision: str = "fp16"  # fp32, fp16, int8
    frame_skip_enabled: bool = False
    adaptive_processing: bool = True
    max_queue_size: int = 10


@dataclass
class OmniSenseConfig:
    """Master configuration for OMNISENSE platform."""
    # System metadata
    config_version: str = "1.0.0"
    profile_name: str = "default"

    # Module configurations
    cameras: Dict[int, CameraConfig] = field(default_factory=dict)
    slam: SLAMConfig = field(default_factory=SLAMConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Paths
    models_dir: str = "models"
    data_dir: str = "data"
    calibration_dir: str = "configs/calibration"
    logs_dir: str = "logs"


class ConfigManager:
    """
    Manages configuration loading, validation, and runtime access.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to main configuration file
        """
        self.config_path = config_path
        self.config: Optional[OmniSenseConfig] = None
        self._config_dict: Dict[str, Any] = {}

    def load(self, config_path: Optional[Path] = None) -> OmniSenseConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file (overrides init path)

        Returns:
            Loaded configuration object

        Raises:
            ConfigFileError: If file cannot be read
            ConfigValidationError: If configuration is invalid
        """
        if config_path is None:
            config_path = self.config_path

        if config_path is None:
            # Use default configuration
            logger.info("No config file specified, using defaults")
            self.config = OmniSenseConfig()
            return self.config

        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigFileError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                self._config_dict = yaml.safe_load(f)

            logger.info(f"Loaded configuration from {config_path}")

            # Parse into dataclass
            self.config = self._parse_config(self._config_dict)

            # Validate configuration
            self._validate_config()

            return self.config

        except yaml.YAMLError as e:
            raise ConfigFileError(f"Error parsing YAML file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")

    def _parse_config(self, config_dict: Dict[str, Any]) -> OmniSenseConfig:
        """Parse dictionary into OmniSenseConfig object."""
        # Parse camera configurations
        cameras = {}
        if 'cameras' in config_dict:
            for cam_id, cam_cfg in config_dict['cameras'].items():
                # Ensure camera_id matches the key
                cam_cfg_copy = cam_cfg.copy()
                cam_cfg_copy['camera_id'] = int(cam_id)
                cameras[int(cam_id)] = CameraConfig(**cam_cfg_copy)

        # Parse other modules
        slam = SLAMConfig(**config_dict.get('slam', {}))
        tracking = TrackingConfig(**config_dict.get('tracking', {}))
        perception = PerceptionConfig(**config_dict.get('perception', {}))
        fusion = FusionConfig(**config_dict.get('fusion', {}))
        intelligence = IntelligenceConfig(**config_dict.get('intelligence', {}))
        database = DatabaseConfig(**config_dict.get('database', {}))
        api = APIConfig(**config_dict.get('api', {}))
        gui = GUIConfig(**config_dict.get('gui', {}))
        performance = PerformanceConfig(**config_dict.get('performance', {}))

        return OmniSenseConfig(
            config_version=config_dict.get('config_version', '1.0.0'),
            profile_name=config_dict.get('profile_name', 'default'),
            cameras=cameras,
            slam=slam,
            tracking=tracking,
            perception=perception,
            fusion=fusion,
            intelligence=intelligence,
            database=database,
            api=api,
            gui=gui,
            performance=performance,
            models_dir=config_dict.get('models_dir', 'models'),
            data_dir=config_dict.get('data_dir', 'data'),
            calibration_dir=config_dict.get('calibration_dir', 'configs/calibration'),
            logs_dir=config_dict.get('logs_dir', 'logs'),
        )

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config is None:
            raise ConfigValidationError("No configuration loaded")

        # Validate cameras
        if len(self.config.cameras) == 0:
            logger.warning("No cameras configured")

        # Validate SLAM stereo pair
        if self.config.slam.enabled:
            cam1, cam2 = self.config.slam.stereo_camera_pair
            if cam1 not in self.config.cameras or cam2 not in self.config.cameras:
                raise ConfigValidationError(
                    f"SLAM stereo pair ({cam1}, {cam2}) references non-existent cameras"
                )

        # Validate GPU settings
        if self.config.performance.gpu_enabled:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("GPU enabled but CUDA not available, falling back to CPU")
                    self.config.performance.gpu_enabled = False
            except ImportError:
                logger.warning("PyTorch not available, GPU will not be used")
                self.config.performance.gpu_enabled = False

        logger.info("Configuration validation passed")

    def save(self, config_path: Optional[Path] = None):
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save configuration (overrides init path)
        """
        if config_path is None:
            config_path = self.config_path

        if config_path is None:
            raise ConfigurationError("No config path specified for saving")

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary
        config_dict = asdict(self.config)

        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Saved configuration to {config_path}")

        except Exception as e:
            raise ConfigFileError(f"Error saving configuration: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dotted key path.

        Args:
            key: Dotted key path (e.g., 'tracking.detection_confidence')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = asdict(self.config) if self.config else {}

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set a configuration value by dotted key path.

        Args:
            key: Dotted key path
            value: New value
        """
        if self.config is None:
            raise ConfigurationError("No configuration loaded")

        keys = key.split('.')
        obj = self.config

        for k in keys[:-1]:
            obj = getattr(obj, k)

        setattr(obj, keys[-1], value)

    def reload(self):
        """Reload configuration from file."""
        if self.config_path is None:
            raise ConfigurationError("Cannot reload: no config path set")

        self.load(self.config_path)
