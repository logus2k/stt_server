FROM python:3.10.18-slim

USER root

RUN pip install --upgrade pip
RUN pip install transformers accelerate python-socketio silero-vad uvicorn

WORKDIR /stt_server
