# Multi-stage Dockerfile for All-in-One Fullstack Deploy (Railway / Single Container)
# Stage 1: Build the React frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
# Build frontend with relative API path (same origin)
ENV VITE_API_URL=""
RUN npm run build

# Stage 2: Python Backend + FFmpeg + MediaPipe + Built Frontend
FROM python:3.11-slim

WORKDIR /app

# Install FFmpeg and OpenCV / MediaPipe system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libgles2 \
        libegl1 \
        libglib2.0-0 \
        libgomp1 \
        fonts-liberation \
        fonts-dejavu \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in an isolated venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY backend/ /app/

# Copy built frontend assets so FastAPI can serve the dashboard
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

ENV PYTHONUNBUFFERED=1
ENV OUTPUT_ROOT=/app/output
ENV KOKORO_CACHE=/app/models/kokoro
RUN mkdir -p /app/output /app/autoclipper/models /app/models/kokoro /app/data

EXPOSE 8000

# Support dynamic PORT assignment for Railway or fallback to 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
