"""
Custom exception classes for OMNISENSE platform.

Provides specific exception types for different failure modes to enable
targeted error handling and recovery strategies.
"""


class OmniSenseError(Exception):
    """Base exception for all OMNISENSE errors."""
    pass


# Camera-related exceptions
class CameraError(OmniSenseError):
    """Base exception for camera-related errors."""
    pass


class CameraConnectionError(CameraError):
    """Camera failed to connect or disconnected unexpectedly."""
    pass


class CameraConfigurationError(CameraError):
    """Camera configuration is invalid or unsupported."""
    pass


class CameraCalibrationError(CameraError):
    """Camera calibration failed or calibration data is invalid."""
    pass


class FrameSynchronizationError(CameraError):
    """Frame synchronization failed across cameras."""
    pass


# Model inference exceptions
class ModelError(OmniSenseError):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Model failed to load or is corrupted."""
    pass


class ModelInferenceError(ModelError):
    """Model inference failed."""
    pass


class ModelNotFoundError(ModelError):
    """Required model file not found."""
    pass


# SLAM exceptions
class SLAMError(OmniSenseError):
    """Base exception for SLAM-related errors."""
    pass


class SLAMInitializationError(SLAMError):
    """SLAM system failed to initialize."""
    pass


class SLAMTrackingLostError(SLAMError):
    """SLAM tracking was lost and recovery failed."""
    pass


class SLAMMapError(SLAMError):
    """Error in map representation or update."""
    pass


# Tracking exceptions
class TrackingError(OmniSenseError):
    """Base exception for tracking-related errors."""
    pass


class PersonDetectionError(TrackingError):
    """Person detection failed."""
    pass


class ReIDError(TrackingError):
    """Person re-identification failed."""
    pass


class TrackAssociationError(TrackingError):
    """Track association across cameras failed."""
    pass


# Fusion exceptions
class FusionError(OmniSenseError):
    """Base exception for sensor fusion errors."""
    pass


class TriangulationError(FusionError):
    """Multi-view triangulation failed."""
    pass


class KalmanFilterError(FusionError):
    """Kalman filter update failed."""
    pass


# Database exceptions
class DatabaseError(OmniSenseError):
    """Base exception for database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Database connection failed."""
    pass


class DatabaseQueryError(DatabaseError):
    """Database query execution failed."""
    pass


# Configuration exceptions
class ConfigurationError(OmniSenseError):
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigurationError):
    """Configuration validation failed."""
    pass


class ConfigFileError(ConfigurationError):
    """Configuration file not found or corrupted."""
    pass


# API exceptions
class APIError(OmniSenseError):
    """Base exception for API-related errors."""
    pass


class AuthenticationError(APIError):
    """Authentication failed."""
    pass


class AuthorizationError(APIError):
    """Authorization failed - insufficient permissions."""
    pass


# Resource exceptions
class ResourceError(OmniSenseError):
    """Base exception for resource-related errors."""
    pass


class GPUMemoryError(ResourceError):
    """Insufficient GPU memory."""
    pass


class DiskSpaceError(ResourceError):
    """Insufficient disk space."""
    pass


class SystemResourceError(ResourceError):
    """System resource limit reached (CPU, RAM, etc.)."""
    pass


# Processing exceptions
class ProcessingError(OmniSenseError):
    """Base exception for processing pipeline errors."""
    pass


class DepthEstimationError(ProcessingError):
    """Depth estimation failed."""
    pass


class PoseEstimationError(ProcessingError):
    """Pose estimation failed."""
    pass


class GazeEstimationError(ProcessingError):
    """Gaze estimation failed."""
    pass
