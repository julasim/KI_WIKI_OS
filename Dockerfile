FROM python:3.12-slim

# ffmpeg für Whisper, git+rsync für Backup-Tool
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies zuerst (besseres Layer-Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bot-Code
COPY ki_wiki_bot.py .

# Whisper-Modell-Cache wird persistiert über Volume
ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "ki_wiki_bot.py"]
