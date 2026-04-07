# =============================================================================
# Simulation Sandbox — Blender 3D Physics (Pipeline: Oddly Satisfying Physics)
# =============================================================================
# This image is used by the Blender executor to run agent-generated
# simulation configs in an isolated container.
#
# Security constraints applied at runtime (not baked into the image):
#   --network=none    No internet access
#   --memory=4g       4GB RAM limit
#   --cpus=2          2 CPU cores max
#   --read-only       Read-only root filesystem (except /workspace)
#   --stop-timeout    300s max execution time
# =============================================================================

FROM python:3.12-slim

# Install system dependencies for headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — pinned versions
RUN pip install --no-cache-dir \
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
