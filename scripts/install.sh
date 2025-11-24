#!/bin/bash
# OMNISENSE Installation Script for Ubuntu 22.04+

set -e  # Exit on error

echo "========================================"
echo "OMNISENSE Platform Installation"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu
if [ ! -f /etc/os-release ]; then
    echo -e "${RED}Error: Cannot determine OS version${NC}"
    exit 1
fi

source /etc/os-release

if [[ "$ID" != "ubuntu" ]]; then
    echo -e "${YELLOW}Warning: This script is designed for Ubuntu. Proceed with caution.${NC}"
fi

echo -e "${GREEN}Detected: $PRETTY_NAME${NC}"

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.9+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}Python version OK: $PYTHON_VERSION${NC}"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    libv4l-dev \
    v4l-utils \
    libpq-dev \
    postgresql-client \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    qt6-base-dev

echo -e "${GREEN}System dependencies installed${NC}"

# Create virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

echo -e "${GREEN}Virtual environment activated${NC}"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

echo -e "${GREEN}Python packages installed${NC}"

# Install OMNISENSE package
echo "Installing OMNISENSE package..."
pip install -e .

echo -e "${GREEN}OMNISENSE package installed${NC}"

# Create necessary directories
echo "Creating data directories..."
mkdir -p models/{yolo,mediapipe,reid,gaze,slam}
mkdir -p data/{calibration,recordings,maps}
mkdir -p logs
mkdir -p configs/calibration

echo -e "${GREEN}Directories created${NC}"

# Download models
echo "Downloading required models..."
python3 scripts/download_models.py

# Check for CUDA
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}CUDA detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo -e "${YELLOW}Warning: CUDA not detected. GPU acceleration will not be available.${NC}"
fi

# Check cameras
echo "Detecting cameras..."
if ls /dev/video* 1> /dev/null 2>&1; then
    echo -e "${GREEN}Detected cameras:${NC}"
    for device in /dev/video*; do
        if [ -c "$device" ]; then
            v4l2-ctl --device=$device --info 2>/dev/null | grep "Card type" || echo "$device"
        fi
    done
else
    echo -e "${YELLOW}Warning: No camera devices detected at /dev/video*${NC}"
fi

# Setup PostgreSQL (optional)
echo ""
read -p "Install and configure PostgreSQL locally? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo apt-get install -y postgresql postgresql-contrib

    # Install TimescaleDB
    echo "Installing TimescaleDB..."
    sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
    wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y timescaledb-2-postgresql-14

    # Setup database
    sudo -u postgres psql -c "CREATE DATABASE omnisense;"
    sudo -u postgres psql -c "CREATE USER omnisense WITH PASSWORD 'omnisense';"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE omnisense TO omnisense;"
    sudo -u postgres psql -d omnisense -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

    echo -e "${GREEN}PostgreSQL and TimescaleDB configured${NC}"
fi

# Create systemd service (optional)
echo ""
read -p "Create systemd service for auto-start? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    OMNISENSE_DIR=$(pwd)
    SERVICE_FILE="/etc/systemd/system/omnisense.service"

    sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=OMNISENSE Multi-Camera Spatial Intelligence Platform
After=network.target postgresql.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$OMNISENSE_DIR
Environment="PATH=$OMNISENSE_DIR/venv/bin:\$PATH"
ExecStart=$OMNISENSE_DIR/venv/bin/python -m omnisense.main --config configs/default/omnisense_config.yaml
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable omnisense.service

    echo -e "${GREEN}Systemd service created and enabled${NC}"
    echo "Use 'sudo systemctl start omnisense' to start the service"
fi

echo ""
echo "========================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run camera calibration: python -m omnisense.camera.calibration"
echo "3. Edit configuration: configs/default/omnisense_config.yaml"
echo "4. Start OMNISENSE: python -m omnisense.main"
echo ""
echo "For Docker deployment:"
echo "  docker-compose up -d"
echo ""
echo "Documentation: docs/README.md"
echo "========================================"
