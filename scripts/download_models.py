#!/usr/bin/env python3
"""
Model download script for OMNISENSE platform.

Downloads all required pre-trained models for the perception pipeline.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omnisense.models.manager import ModelManager
from omnisense.utils.logger import setup_logger, get_logger

# Setup logging
setup_logger("omnisense")
logger = get_logger(__name__)


def main():
    """Download all required models."""
    logger.info("=" * 60)
    logger.info("OMNISENSE Model Download Utility")
    logger.info("=" * 60)

    # Initialize model manager
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_manager = ModelManager(str(models_dir))

    # List of models to download
    models_to_download = [
        'yolov8n',
        'yolov8s',
        'yolov8n-pose',
    ]

    logger.info(f"Downloading {len(models_to_download)} models...")

    for model_name in models_to_download:
        try:
            logger.info(f"Downloading {model_name}...")
            model_manager.download_model(model_name)
            logger.info(f"✓ {model_name} downloaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to download {model_name}: {e}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Model download complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Note: MediaPipe models will be downloaded automatically on first use")
    logger.info("Note: ReID models will be downloaded from torchreid on first use")
    logger.info("")


if __name__ == '__main__':
    main()
