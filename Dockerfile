# Pipeline + dashboard image. Ollama runs in its own container (see docker-compose.yml).
FROM python:3.12-slim-bookworm

# FFmpeg/FFprobe from Debian (CPU encoding with libx264; NVENC is not included).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY scripts ./scripts
COPY src ./src

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Default: the read-only dashboard. The pipeline runs via `docker compose run pipeline ...`.
CMD ["python", "scripts/serve_ui.py", "--host", "0.0.0.0", "--port", "8000"]
