FROM stt_server-server:1.0

USER root

COPY stt_server.py /stt_server

EXPOSE 2700

WORKDIR /stt_server

CMD ["python", "stt_server.py"]
