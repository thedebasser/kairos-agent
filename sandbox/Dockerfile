# =============================================================================
# Simulation Sandbox — Pygame + Pymunk (Pipeline 1: Oddly Satisfying Physics)
# =============================================================================
# This image is used by execute_simulation() to run untrusted agent-generated
# simulation code in an isolated container.
#
# Security constraints applied at runtime (not baked into the image):
#   --network=none    No internet access
#   --memory=4g       4GB RAM limit
#   --cpus=2          2 CPU cores max
#   --read-only       Read-only root filesystem (except /workspace)
#   --stop-timeout    300s max execution time
# =============================================================================

FROM python:3.12-slim

# Install system dependencies for Pygame (headless rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — pinned versions
RUN pip install --no-cache-dir \
    pygame==2.6.1 \
    pymunk==6.8.1 \
    numpy==1.26.4 \
    Pillow==10.4.0

# Create non-root user for sandbox execution
RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /workspace

# Create output directory
RUN mkdir -p /workspace/output

# Set SDL to use dummy video driver (headless rendering)
ENV SDL_VIDEODRIVER=dummy
ENV SDL_AUDIODRIVER=dummy

# No ENTRYPOINT — command provided at runtime by execute_simulation()
