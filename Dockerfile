# Multi-stage Dockerfile for Helicopter - Reverse Pakati Visual Knowledge Extraction

# Base stage with common dependencies
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Expose ports
EXPOSE 8000 8888

# Set development environment
ENV HELICOPTER_ENV=development
ENV WANDB_MODE=offline

# Command for development
CMD ["python", "-m", "helicopter.api.server", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY helicopter/ ./helicopter/
COPY setup.py pyproject.toml README.md ./
COPY configs/ ./configs/

# Install package
RUN pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash helicopter
USER helicopter

# Expose port
EXPOSE 8000

# Set production environment
ENV HELICOPTER_ENV=production
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command for production
CMD ["python", "-m", "helicopter.api.server", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Training stage - optimized for model training
FROM base as training

# Install additional training dependencies
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    optuna \
    ray[tune] \
    accelerate

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Set training environment
ENV HELICOPTER_ENV=training
ENV WANDB_PROJECT=helicopter

# Command for training
CMD ["python", "-m", "helicopter.training.cli"]

# Inference stage - minimal for inference only
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as inference

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy only inference requirements
COPY requirements-inference.txt ./
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy minimal package files
COPY helicopter/core/ ./helicopter/core/
COPY helicopter/models/ ./helicopter/models/
COPY helicopter/utils/ ./helicopter/utils/
COPY helicopter/__init__.py ./helicopter/

# Copy pre-trained models (if included)
COPY models/ ./models/

# Create non-root user
RUN useradd --create-home --shell /bin/bash helicopter
USER helicopter

# Set inference environment
ENV HELICOPTER_ENV=inference
ENV PYTHONPATH=/app

# Command for inference
CMD ["python", "-c", "from helicopter import HelicopterPipeline; print('Helicopter ready for inference')"] 