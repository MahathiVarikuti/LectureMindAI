FROM python:3.11-slim

# ffmpeg/ffprobe: audio extraction, frame sampling, duration checks
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV UPLOAD_DIR=/app/data/uploads \
    CHROMA_DIR=/app/data/chroma \
    SQLITE_PATH=/app/data/chat_history.db \
    HF_HOME=/app/data/hf-cache
RUN mkdir -p /app/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
