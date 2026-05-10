#!/usr/bin/env bash
# Switch STT to Whisper (v1).
# Stops any running stt_server_v2 (Parakeet) then brings v1 up.
# Both services bind host port 2700 and answer to the `stt_server`
# Docker hostname, so noted/agent_server work without re-config.
set -euo pipefail
cd "$(dirname "$0")"

docker compose --profile v2 stop stt_server_v2 >/dev/null 2>&1 || true
docker compose --profile default up -d stt_server

echo "✅ STT backend: Whisper (v1) — http://stt_server:2700"
docker ps --filter name=stt_server --format '   {{.Names}}: {{.Status}}'
