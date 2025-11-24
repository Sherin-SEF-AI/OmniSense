"""
Model management system for OMNISENSE platform.

Handles downloading, loading, and managing all deep learning models used
in the perception pipeline including YOLO, MediaPipe, ReID, and gaze models.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import onnxruntime as ort
from urllib.request import urlretrieve
from tqdm import tqdm

from omnisense.utils.logger import get_logger
from omnisense.utils.exceptions import ModelError, ModelLoadError, ModelNotFoundError

logger = get_logger(__name__)


class DownloadProgressBar:
    """Progress bar for model downloads."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


class ModelManager:
    """
    Manages all models for the OMNISENSE platform.

    Handles model downloading, loading, and inference configuration.
    """

    # Model URLs
    MODEL_URLS = {
        'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'yolov8n-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt',
    }

    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager.

        Args:
            models_dir: Root directory for storing models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.yolo_dir = self.models_dir / "yolo"
        self.mediapipe_dir = self.models_dir / "mediapipe"
        self.reid_dir = self.models_dir / "reid"
        self.gaze_dir = self.models_dir / "gaze"
        self.slam_dir = self.models_dir / "slam"

        for dir_path in [self.yolo_dir, self.mediapipe_dir, self.reid_dir, self.gaze_dir, self.slam_dir]:
            dir_path.mkdir(exist_ok=True)

        # Cache for loaded models
        self.loaded_models: Dict[str, Any] = {}

        # Inference providers
        self.onnx_providers = self._get_onnx_providers()

        logger.info(f"ModelManager initialized. Models directory: {self.models_dir}")
        logger.info(f"ONNX Runtime providers: {self.onnx_providers}")

    def _get_onnx_providers(self) -> list:
        """Get available ONNX Runtime execution providers."""
        providers = []

        # Check for CUDA
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
            logger.info("CUDA execution provider available")

        # Fallback to CPU
        providers.append('CPUExecutionProvider')

        return providers

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """
        Download a model if not already present.

        Args:
            model_name: Name of the model to download
            force: Force redownload even if file exists

        Returns:
            Path to downloaded model file

        Raises:
            ModelError: If download fails
        """
        if model_name not in self.MODEL_URLS:
            raise ModelNotFoundError(f"Unknown model: {model_name}")

        url = self.MODEL_URLS[model_name]
        filename = url.split('/')[-1]

        # Determine output directory
        if 'yolo' in model_name:
            output_dir = self.yolo_dir
        else:
            output_dir = self.models_dir

        output_path = output_dir / filename

        if output_path.exists() and not force:
            logger.info(f"Model {model_name} already exists at {output_path}")
            return output_path

        logger.info(f"Downloading {model_name} from {url}")

        try:
            urlretrieve(url, output_path, DownloadProgressBar())
            logger.info(f"Downloaded {model_name} to {output_path}")
            return output_path

        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise ModelError(f"Failed to download model {model_name}: {e}")

    def load_yolo_model(
        self,
        model_name: str = 'yolov8n',
        device: Optional[str] = None
    ):
        """
        Load YOLO model for object detection.

        Args:
            model_name: YOLO model variant (yolov8n, yolov8s, etc.)
            device: Device to load model on ('cuda' or 'cpu')

        Returns:
            Loaded YOLO model

        Raises:
            ModelLoadError: If model fails to load
        """
        cache_key = f"yolo_{model_name}"

        if cache_key in self.loaded_models:
            logger.debug(f"Returning cached YOLO model: {model_name}")
            return self.loaded_models[cache_key]

        try:
            from ultralytics import YOLO

            # Ensure model is downloaded
            model_path = self.yolo_dir / f"{model_name}.pt"
            if not model_path.exists():
                model_path = self.download_model(model_name)

            # Load model
            logger.info(f"Loading YOLO model: {model_name}")
            model = YOLO(str(model_path))

            # Set device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            model.to(device)

            self.loaded_models[cache_key] = model
            logger.info(f"YOLO model {model_name} loaded on {device}")

            return model

        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model {model_name}: {e}")

    def load_mediapipe_face_mesh(self):
        """
        Load MediaPipe Face Mesh model.

        Returns:
            MediaPipe FaceMesh solution
        """
        cache_key = "mediapipe_face_mesh"

        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        try:
            import mediapipe as mp

            logger.info("Loading MediaPipe Face Mesh")
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.loaded_models[cache_key] = face_mesh
            logger.info("MediaPipe Face Mesh loaded")

            return face_mesh

        except Exception as e:
            raise ModelLoadError(f"Failed to load MediaPipe Face Mesh: {e}")

    def load_mediapipe_pose(self):
        """
        Load MediaPipe Pose model.

        Returns:
            MediaPipe Pose solution
        """
        cache_key = "mediapipe_pose"

        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        try:
            import mediapipe as mp

            logger.info("Loading MediaPipe Pose")
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.loaded_models[cache_key] = pose
            logger.info("MediaPipe Pose loaded")

            return pose

        except Exception as e:
            raise ModelLoadError(f"Failed to load MediaPipe Pose: {e}")

    def load_reid_model(self, model_name: str = 'osnet_x1_0'):
        """
        Load person re-identification model.

        Args:
            model_name: ReID model name

        Returns:
            Loaded ReID model
        """
        cache_key = f"reid_{model_name}"

        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        try:
            import torchreid

            logger.info(f"Loading ReID model: {model_name}")

            # Load pretrained model
            model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                loss='softmax',
                pretrained=True
            )

            # Set to eval mode
            model.eval()

            # Move to GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)

            self.loaded_models[cache_key] = model
            logger.info(f"ReID model {model_name} loaded on {device}")

            return model

        except Exception as e:
            raise ModelLoadError(f"Failed to load ReID model {model_name}: {e}")

    def get_model(self, cache_key: str) -> Optional[Any]:
        """Get a loaded model from cache."""
        return self.loaded_models.get(cache_key)

    def clear_cache(self):
        """Clear all loaded models from memory."""
        logger.info("Clearing model cache")
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.

        Returns:
            Dictionary with model information
        """
        info = {
            'models_dir': str(self.models_dir),
            'loaded_models': list(self.loaded_models.keys()),
            'num_loaded': len(self.loaded_models),
            'onnx_providers': self.onnx_providers,
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)
            info['cuda_memory_reserved'] = torch.cuda.memory_reserved(0)

        return info
