FROM python:3.12-slim

# ffmpeg fuer Whisper. Bot v2 hat KEINEN Photo-Path mehr (kein OCR/Tesseract),
# kein Backup-Vault (kein git/rsync). Nur das Noetigste.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies zuerst (besseres Layer-Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bot-Code
COPY ki_wiki_bot.py .
# Phase X3 Thin-Client: MCP-HTTP-Session + Tool-Wrapper
COPY mcp_client.py .
COPY mcp_thin_tools.py .
# Smoke-Tests + Maintenance-Scripts (Diagnostik im Container)
COPY scripts/ ./scripts/

# Whisper-Modell-Cache wird persistiert über Volume
ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "ki_wiki_bot.py"]
