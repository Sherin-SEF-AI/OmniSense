# OMNISENSE Multi-Camera Spatial Intelligence Platform
# Production Docker Image

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
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
    libglu1-mesa \
    libxi6 \
    libxmu6 \
    libgconf-2-4 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Qt dependencies for PyQt6
RUN apt-get update && apt-get install -y \
    qt6-base-dev \
    libqt6gui6 \
    libqt6widgets6 \
    libqt6opengl6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Install OMNISENSE package
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p /app/models \
    /app/data/calibration \
    /app/data/recordings \
    /app/data/maps \
    /app/logs \
    /app/configs/calibration

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import omnisense; print('OK')" || exit 1

# Entry point
CMD ["python3", "-m", "omnisense.main", "--config", "configs/default/omnisense_config.yaml"]
