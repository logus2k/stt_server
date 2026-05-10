# stt_server v2 — Parakeet TDT streaming.
#
# Reuses the v1 base image so torch + CUDA + python-socketio + uvicorn are
# already in place. We only add NeMo's ASR collection (~1-2 GB layer) and
# COPY the v2 server source. The v1 image (stt_server:1.0) is untouched —
# so rolling back is `docker compose stop stt_server_v2 && docker compose
# up -d stt_server` (one line, no rebuild).
#
# Build:
#   docker build -f stt_server.v2.Dockerfile -t stt_server:2.0 .
#
# Build can take 10-20 min on first run (NeMo install pulls a lot of deps).

FROM stt_server-server:1.0

USER root

# Install NeMo's ASR collection. We deliberately install on top of the
# existing torch/cuda already in the base image to avoid version
# conflicts; NeMo's setup respects existing torch.
RUN pip install --no-cache-dir \
      "nemo_toolkit[asr]>=2.0.0"

COPY stt_server_v2.py /stt_server/stt_server_v2.py

EXPOSE 2700

WORKDIR /stt_server

CMD ["python", "-u", "stt_server_v2.py"]
