# ── Stage 1: Build React frontend ──────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
# Vite builds to ../frontend_build (one level up from frontend/)
RUN npm run build

# ── Stage 2: Python backend + bundled frontend ──────────────
FROM python:3.11-slim

WORKDIR /app

# PyMuPDF ships its own compiled binary -- no system libmupdf needed
# Only install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cached layer)
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Backend source
COPY backend/ ./backend/

# Copy built frontend (Vite outputs to frontend_build/ at repo root)
COPY --from=frontend-builder /build/frontend_build ./frontend_build

# Uploads directory
RUN mkdir -p /app/backend/uploads

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
