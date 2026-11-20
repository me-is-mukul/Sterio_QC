# Pipeline + dashboard image. The optional vision-language model (Ollama) runs in its own
# container (see docker-compose.yml, profile "vlm").
FROM python:3.12-slim-bookworm

# FFmpeg/FFprobe from Debian (CPU encoding with libx264; NVENC is not included).
# libgl1 / libglib2.0-0 are needed by OpenCV (used by MediaPipe).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY models ./models
COPY scripts ./scripts
COPY src ./src

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Default: the read-only dashboard. The pipeline runs via `docker compose run pipeline ...`.
CMD ["python", "scripts/serve_ui.py", "--host", "0.0.0.0", "--port", "8000"]
