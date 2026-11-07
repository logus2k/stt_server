FROM python:3.10.18-slim

USER root

RUN pip install --upgrade pip
# every package at the version of the image that ran until 2026-11-06 (base-requirements.lock: pip freeze of stt_server:1.0) -
# the base was lost in an image clean-up (Cortex register #175) and these installs were unpinned: a rebuild could drift
COPY base-requirements.lock /tmp/base-requirements.lock
RUN pip install -c /tmp/base-requirements.lock transformers accelerate python-socketio silero-vad uvicorn

WORKDIR /stt_server
