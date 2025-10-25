# GPU-ready Python backend
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \    python3 python3-pip python3-venv python3-dev build-essential \    sox libsox-dev libsndfile1 ffmpeg fluidsynth \ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# Source
COPY server /app/server
RUN mkdir -p /app/data/uploads /app/data/outputs /app/data/checkpoints /app/assets/soundfonts

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
