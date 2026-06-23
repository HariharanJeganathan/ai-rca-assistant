# ============================================================
# Dockerfile — Builds the FastAPI API container
# ============================================================
# A Dockerfile is a recipe for creating a Docker image.
# Each line is a layer that gets cached.
#
# Build: docker build -t rca-api .
# Run:   docker run -p 8000:8000 rca-api
# ============================================================

# Start from official Python 3.11 slim image
# "slim" = smaller size, no unnecessary tools
FROM python:3.11-slim

# Set working directory inside container
# All subsequent commands run from here
WORKDIR /app

# ── Layer 1: System dependencies ──────────────────────────
# These rarely change, so Docker caches this layer
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*   # Clean up to reduce image size

# ── Layer 2: Python dependencies ──────────────────────────
# Copy ONLY requirements.txt first (not the whole code)
# This way, if code changes but requirements don't,
# Docker skips this slow layer and uses the cache!
COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

# ── Layer 3: Application code ─────────────────────────────
# Copy everything else
COPY . /app/

# ── Environment ───────────────────────────────────────────
# Tell Python not to write .pyc files (cleaner container)
ENV PYTHONDONTWRITEBYTECODE=1
# Don't buffer stdout/stderr (so logs appear immediately)
ENV PYTHONUNBUFFERED=1
# Add /app to Python path so imports work
ENV PYTHONPATH=/app

# ── Port ──────────────────────────────────────────────────
# Document which port the app listens on
EXPOSE 8000

# ── Health Check ──────────────────────────────────────────
# Docker uses this to know if the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start Command ─────────────────────────────────────────
# This runs when the container starts
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
