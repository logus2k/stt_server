#!/usr/bin/env bash
# Switch STT to Parakeet (v2).
# Stops any running stt_server (Whisper) then brings v2 up.
# Both services bind host port 2700 and answer to the `stt_server`
# Docker hostname, so noted/agent_server work without re-config.
set -euo pipefail
cd "$(dirname "$0")"

docker compose --profile default stop stt_server >/dev/null 2>&1 || true
docker compose --profile v2 up -d stt_server_v2

echo "✅ STT backend: Parakeet (v2) — http://stt_server:2700"
docker ps --filter name=stt_server --format '   {{.Names}}: {{.Status}}'
