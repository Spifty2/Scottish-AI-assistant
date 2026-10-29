FROM python:3.12-slim
WORKDIR /app

ENV OLLAMA_API_URL=http://host.docker.internal:11434/api/chat
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio
RUN pip install --no-cache-dir -r requirements.txt
RUN pip list | grep -i -E "torch|nvidia" || true
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

COPY . .
EXPOSE 8080
CMD ["python", "Voiceassistant.py"]
