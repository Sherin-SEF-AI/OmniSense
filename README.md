# OMNISENSE Multi-Camera Spatial Intelligence Platform

**Production-ready, real-time multi-camera perception system providing unified spatial intelligence through simultaneous visual SLAM, human state analysis, multi-person tracking, and sensor fusion.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## Overview

OMNISENSE is a comprehensive spatial intelligence platform that processes input from four USB cameras simultaneously, implementing advanced computer vision and AI to build a unified understanding of the monitored environment. The system tracks people in 3D space, analyzes their states (attention, posture, drowsiness), and provides actionable intelligence through multiple interfaces.

### Key Features

- **Multi-Camera Perception**: Synchronized capture from 4 USB cameras with V4L2 backend
- **Person Detection & Tracking**: YOLOv8/v9 detection with DeepSORT tracking
- **Cross-Camera Re-identification**: Person ReID across multiple views
- **3D Spatial Fusion**: Multi-view triangulation and Kalman filtering
- **Human State Analysis**:
  - Face detection with 468 3D landmarks (MediaPipe)
  - Head pose estimation (roll, pitch, yaw)
  - Eye gaze tracking and attention analysis
  - Drowsiness detection via PERCLOS
  - Full-body pose estimation (33 keypoints)
  - Posture classification
- **Visual SLAM**: ORB-SLAM3 integration for environment mapping
- **Real-time Processing**: Multi-threaded pipeline with GPU acceleration
- **PyQt6 GUI**: Multi-pane interface with camera views, 3D visualization, and analytics
- **Database Integration**: PostgreSQL + TimescaleDB for time-series data
- **API Services**: REST API, WebSocket streaming, ROS2 bridge, MQTT support

---

## System Requirements

### Hardware

- **CPU**: Multi-core processor (Intel i7/i9 or AMD Ryzen 7/9 recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 6GB+ VRAM (CUDA 11.8+)
- **Cameras**: 4x USB cameras (UVC compatible)
- **Storage**: 256GB+ SSD for data storage

### Software

- **OS**: Ubuntu 22.04 LTS or newer
- **Python**: 3.9, 3.10, or 3.11
- **CUDA**: 11.8+ (for GPU acceleration)
- **PostgreSQL**: 14+ with TimescaleDB extension (optional)
- **Docker**: 20.10+ (for containerized deployment)

---

## Installation

### Quick Start (Ubuntu 22.04)

```bash
# Clone repository
git clone https://github.com/Sherin-SEF-AI/OmniSense.git
cd OmniSense

# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh

# Activate virtual environment
source venv/bin/activate

# Download models
python scripts/download_models.py

# Start OMNISENSE
python -m omnisense.main
```

### Manual Installation

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev git cmake build-essential \
    libopencv-dev libv4l-dev v4l-utils libpq-dev qt6-base-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install OMNISENSE
pip install -e .

# Download models
python scripts/download_models.py
```

### Docker Deployment

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f omnisense

# Stop services
docker-compose down
```

---

## Configuration

The platform is configured via YAML files in `configs/default/omnisense_config.yaml`.

### Camera Configuration

```yaml
cameras:
  0:
    camera_id: 0
    device_path: "/dev/video0"
    name: "ACER HD Camera"
    resolution: [640, 480]
    fps: 30
    enabled: true
```

### Detection & Tracking

```yaml
tracking:
  enabled: true
  yolo_model: "yolov8n.pt"
  detection_confidence: 0.5
  tracker_type: "deep_sort"
  reid_model: "osnet_x1_0"
```

### Performance Settings

```yaml
performance:
  gpu_enabled: true
  gpu_device_id: 0
  num_threads: 4
  batch_size: 1
  inference_precision: "fp16"
```

---

## Camera Calibration

Camera calibration is required for accurate 3D reconstruction and multi-view geometry.

### Intrinsic Calibration

```bash
# Capture calibration images (use chessboard pattern)
# Then run calibration tool
omnisense-calibrate intrinsic \
    --images data/calibration/camera_0/ \
    --output configs/calibration/camera_0.yaml \
    --chessboard 9,6 \
    --square-size 0.025
```

### Stereo Calibration

```bash
# Calibrate stereo pair (e.g., cameras 1 and 2)
omnisense-calibrate stereo \
    --images-left data/calibration/camera_1/ \
    --images-right data/calibration/camera_2/ \
    --output configs/calibration/stereo_1_2.yaml
```

---

## Usage

### GUI Mode (Default)

```bash
python -m omnisense.main --config configs/default/omnisense_config.yaml
```

### Headless Mode

```bash
python -m omnisense.main --headless --config configs/default/omnisense_config.yaml
```

### Command Line Options

```
--config, -c       Path to configuration file
--headless         Run without GUI
--log-level        Logging level (DEBUG, INFO, WARNING, ERROR)
```

---

## Architecture

### System Components

```
omnisense/
├── camera/           # Camera management and synchronization
│   ├── manager.py    # Camera interface with V4L2
│   ├── synchronizer.py  # Frame synchronization
│   └── calibration.py   # Camera calibration
├── models/           # Model management
│   └── manager.py    # Model loading and inference
├── perception/       # Human state analysis
│   ├── face_analysis.py  # Face detection and gaze
│   ├── pose.py       # Body pose estimation
│   └── drowsiness.py # Drowsiness detection
├── tracking/         # Person detection and tracking
│   ├── detector.py   # YOLO-based detection
│   └── tracker.py    # DeepSORT + multi-camera tracking
├── fusion/           # Sensor fusion
│   └── world_state.py  # 3D state estimation
├── slam/             # Visual SLAM (ORB-SLAM3)
├── intelligence/     # High-level reasoning
├── database/         # Database interface
│   └── schema.py     # PostgreSQL schema
├── api/              # External interfaces
├── gui/              # PyQt6 interface
│   └── main_window.py  # Main GUI window
├── config/           # Configuration management
│   └── manager.py    # Config loading and validation
└── utils/            # Utilities
    ├── logger.py     # Logging system
    ├── geometry.py   # 3D geometry functions
    ├── timing.py     # Performance monitoring
    └── exceptions.py # Custom exceptions
```

### Processing Pipeline

```
Camera Capture (4x) → Frame Synchronization → Undistortion
                              ↓
         ┌────────────────────┴─────────────────────┐
         ↓                    ↓                     ↓
    Person Detection    Face Analysis        Pose Estimation
         ↓                    ↓                     ↓
    Multi-Camera         Gaze & Head Pose    Posture Analysis
    Tracking                  ↓                     ↓
         ↓                Drowsiness           Activity
         └────────────────→ Detection ←────────Recognition
                              ↓
                      Sensor Fusion (3D)
                              ↓
                      World State Update
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
            GUI          Database           API
```

---

## API Reference

### REST API

Default endpoint: `http://localhost:8000`

#### Get World State

```bash
GET /api/v1/world-state
```

Response:
```json
{
  "timestamp": 1234567890.123,
  "persons": [
    {
      "global_id": 1,
      "position_3d": [1.2, 0.5, 2.3],
      "velocity_3d": [0.1, 0.0, 0.2],
      "confidence": 0.95,
      "cameras_visible": [0, 1, 2]
    }
  ]
}
```

#### Get Person by ID

```bash
GET /api/v1/persons/{global_id}
```

#### Get System Status

```bash
GET /api/v1/status
```

### WebSocket Streaming

Connect to `ws://localhost:8001/ws` for real-time updates.

---

## Database Schema

### Key Tables

- **person_tracks**: Person tracking data (time-series)
- **face_states**: Face analysis results
- **body_poses**: Body pose keypoints
- **drowsiness_events**: Drowsiness detections
- **activities**: Detected activities
- **interactions**: Person interactions
- **alerts**: System alerts
- **system_metrics**: Performance metrics

### Querying Data

```python
from omnisense.database.schema import DatabaseManager

db = DatabaseManager("postgresql://omnisense:password@localhost/omnisense")
session = db.get_session()

# Query recent person tracks
from omnisense.database.schema import PersonTrack
tracks = session.query(PersonTrack).order_by(PersonTrack.timestamp.desc()).limit(100).all()
```

---

## Performance

### Typical Performance Metrics

- **Detection**: 30-60 FPS (YOLOv8n on RTX 3080)
- **Face Analysis**: 25-40 FPS (MediaPipe)
- **Pose Estimation**: 20-35 FPS (MediaPipe)
- **Synchronized Frames**: 20-30 FPS (4 cameras)
- **End-to-End Latency**: 50-100ms

### Optimization Tips

1. **Use FP16 precision** for 2x inference speedup
2. **Batch processing** when latency permits
3. **Reduce resolution** for distant cameras
4. **Disable unused modules** in configuration
5. **Use GPU for all inference** operations

---

## Troubleshooting

### Camera Issues

```bash
# List cameras
v4l2-ctl --list-devices

# Check camera capabilities
v4l2-ctl --device=/dev/video0 --all

# Test camera
ffplay /dev/video0
```

### GPU Issues

```bash
# Check CUDA
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Database Issues

```bash
# Check PostgreSQL
sudo systemctl status postgresql

# Connect to database
psql -h localhost -U omnisense -d omnisense
```

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=omnisense tests/

# Run specific module tests
pytest tests/unit/test_camera.py
```

### Code Formatting

```bash
# Format code
black omnisense/

# Check style
flake8 omnisense/

# Type checking
mypy omnisense/
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run code formatters
6. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

If you use OMNISENSE in your research, please cite:

```bibtex
@software{omnisense2024,
  title={OMNISENSE: Multi-Camera Spatial Intelligence Platform},
  author={OmniSense Development Team},
  year={2024},
  url={https://github.com/Sherin-SEF-AI/OmniSense}
}
```

---

## Acknowledgments

- **YOLOv8/v9**: Ultralytics team
- **MediaPipe**: Google Research
- **ORB-SLAM3**: Universidad de Zaragoza
- **PyQt6**: Riverbank Computing
- **PostgreSQL**: PostgreSQL Global Development Group
- **TimescaleDB**: Timescale Inc.

---

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/Sherin-SEF-AI/OmniSense/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sherin-SEF-AI/OmniSense/discussions)

---

**Built with ❤️ for spatial intelligence applications**
