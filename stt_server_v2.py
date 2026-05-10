"""
stt_server v2 — NVIDIA Parakeet TDT streaming + Silero VAD upstream gate.

Hybrid design:
  - Silero VAD runs continuously on the input stream. It decides WHEN
    Parakeet should be invoked (utterance boundaries) — same gating
    discipline as v1, which we tested empirically on 2026-05-10.
  - Parakeet runs streaming inference INSIDE each detected utterance,
    emitting per-chunk partials (~400ms cadence) for the tele-type UX
    that Whisper couldn't deliver.

Why VAD upstream after the original v2 plan said "no VAD": empirical
test on 2026-05-10 showed Parakeet TDT 0.6B v3 hallucinates short
phrases ("Okay.", "Yeah.", "Mm-hmm.", "Who.") on background noise even
above the energy floor. The "Parakeet is robust against noise" claim
in the migration doc was overstated. Documented in the alternatives
table; this is the documented fallback.

Wire-compatible with v1:
  - Listens on the SAME port (2700)
  - Same Socket.IO inbound events: audio_data, subscribe_transcripts,
    cleanup_client, client_disconnected
  - Same outbound `transcription` event for FINAL transcripts (agent_server
    forwards this unchanged → noted's UserTranscript path keeps working)

NEW outbound event for streaming:
  - `transcription_partial {text, client_id, ts, frame_id}` fires every
    ~400ms during an utterance. agent_server needs a small extension to
    forward this to noted clients (separate piece of work).

Pragmatic latency expectations (will measure on real traffic):
  - First partial: ~500-800ms after speech start
  - Subsequent partials: every ~400ms
  - Final: ~500ms after Silero detects speech-end (default min_silence)

If perceived latency is too high, the alternatives are documented in
~/env/assets/noted/documents/stt/stt_migration.md (cache-aware streaming
on Parakeet-CTC, NVIDIA Riva, etc).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import socketio
import uvicorn

# NeMo is loaded lazily (inside main) so module import is cheap during
# Dockerfile linting / test discovery. Heavy GPU initialisation only runs
# when the server actually starts.

# ── Configuration (env-overridable) ──────────────────────────────────

MODEL_PATH = os.environ.get(
    "STT_V2_MODEL_PATH",
    "/stt_server/data/models/parakeet/parakeet-tdt-0.6b-v3.nemo",
)
SETTINGS_PATH = os.environ.get(
    "STT_V2_SETTINGS_PATH",
    "/stt_server/data/configuration/stt.server.settings.json",
)
PORT = int(os.environ.get("STT_V2_PORT", "2700"))
SAMPLE_RATE = 16000  # Parakeet is trained on 16kHz; do not change.

# Streaming knobs. Tuned for ~500ms first-partial latency at 16kHz.
# All values in seconds unless suffixed _samples.
PARTIAL_INTERVAL_SEC = float(os.environ.get("STT_V2_PARTIAL_INTERVAL", "0.4"))
MAX_BUFFER_SEC = float(os.environ.get("STT_V2_MAX_BUFFER", "30.0"))

# Silero VAD knobs. The VAD decides WHEN Parakeet runs (gates utterances).
# Tuned 2026-05-10 from real-traffic observations:
#   - SPEECH_PAD_MS bumped 200→400ms because the leading "I" was being
#     dropped: VAD detects speech-start after the short "I" already
#     played, and our pre-roll wasn't long enough to recover it.
#   - MIN_SPEECH_MS bumped 250→350ms so brief vocalizations ("uh",
#     "huh") don't open a transcription window. Parakeet has no
#     no-speech probability check like Whisper, so VAD has to gate.
VAD_THRESHOLD = float(os.environ.get("STT_V2_VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_MS = int(os.environ.get("STT_V2_VAD_MIN_SPEECH_MS", "350"))
VAD_MIN_SILENCE_MS = int(os.environ.get("STT_V2_VAD_MIN_SILENCE_MS", "500"))
VAD_SPEECH_PAD_MS = int(os.environ.get("STT_V2_VAD_SPEECH_PAD_MS", "400"))
# Silero VAD requires fixed sub-chunks at 16kHz: 512 samples (32ms).
VAD_CHUNK_SAMPLES = 512

# Derived sample counts.
PARTIAL_INTERVAL_SAMPLES = int(SAMPLE_RATE * PARTIAL_INTERVAL_SEC)
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * MAX_BUFFER_SEC)

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Per-client streaming state ───────────────────────────────────────


@dataclass
class ClientStream:
    """Per-client audio buffer + transcription state.

    Hybrid VAD + Parakeet flow:
      1. Incoming PCM16 chunks accumulate in `samples` (full session audio).
      2. Silero VADIterator processes 512-sample sub-chunks; tracks
         in-speech / out-of-speech transitions.
      3. While in speech: every PARTIAL_INTERVAL_SAMPLES of new audio,
         run Parakeet on samples[speech_start_idx:] and emit a partial.
      4. On VAD speech-end: emit final + reset.

    Why we run Parakeet only inside VAD windows: Parakeet TDT v3
    hallucinates short phrases on background noise. VAD as upstream
    gate prevents that without losing the streaming partials inside
    each utterance."""
    client_id: str
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # VAD state.
    vad_iterator: object = None  # silero_vad.VADIterator, set by server
    vad_leftover: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    in_speech: bool = False
    speech_start_idx: int = 0       # buffer index where current utterance started
    speech_end_pending: bool = False  # set by VAD on speech-end; consumed by finalize
    # Transcription state.
    samples_at_last_partial: int = 0
    last_partial_text: str = ""
    frame_counter: int = 0
    # Monotonic utterance id; incremented on reset() (after every final).
    # _emit_partial captures this at start; if it changes by the time
    # transcription completes (because a final fired in between), the
    # partial is dropped — prevents post-final ghost partials.
    utterance_id: int = 0

    def add_samples(self, new_samples: np.ndarray) -> tuple[bool, bool]:
        """Append samples + run VAD on full 512-sample sub-chunks.
        Returns (speech_started_now, speech_ended_now) so the caller can
        react to transitions on this batch."""
        self.samples = np.concatenate([self.samples, new_samples])

        speech_started = False
        speech_ended = False

        if self.vad_iterator is None:
            return speech_started, speech_ended

        # Stitch with leftover from previous call so we always feed Silero
        # contiguous 512-sample windows.
        combined = np.concatenate([self.vad_leftover, new_samples])
        n_full = len(combined) // VAD_CHUNK_SAMPLES
        for i in range(n_full):
            sub = combined[i * VAD_CHUNK_SAMPLES:(i + 1) * VAD_CHUNK_SAMPLES]
            try:
                event = self.vad_iterator(sub, return_seconds=False)
            except Exception:
                event = None
            if not event:
                continue
            if "start" in event and not self.in_speech:
                self.in_speech = True
                # Anchor the transcription window at the VAD-reported start
                # (clamped to current buffer; Silero counts samples since
                # the iterator was reset).
                self.speech_start_idx = min(event["start"], len(self.samples))
                speech_started = True
            if "end" in event and self.in_speech:
                self.in_speech = False
                self.speech_end_pending = True
                speech_ended = True

        self.vad_leftover = combined[n_full * VAD_CHUNK_SAMPLES:]
        return speech_started, speech_ended

    def should_emit_partial(self) -> bool:
        """True while currently in a speech window AND enough new audio
        accumulated to warrant another transcription pass."""
        if not self.in_speech:
            return False
        new_in_window = len(self.samples) - max(self.samples_at_last_partial, self.speech_start_idx)
        return new_in_window >= PARTIAL_INTERVAL_SAMPLES

    def should_finalize(self) -> bool:
        """True when VAD declared speech-end AND we have a partial to
        commit. Also force-finalize if the utterance window has grown past
        MAX_BUFFER_SEC."""
        if self.speech_end_pending and self.last_partial_text:
            return True
        if (
            self.in_speech
            and (len(self.samples) - self.speech_start_idx) >= MAX_BUFFER_SAMPLES
            and self.last_partial_text
        ):
            return True
        return False

    def utterance_audio(self) -> np.ndarray:
        """Audio of the current utterance (since the last VAD speech-start).
        This is what Parakeet transcribes — never the whole session."""
        return self.samples[self.speech_start_idx:]

    def reset(self) -> None:
        """Reset transcription state but keep the VAD iterator so it
        continues processing seamlessly. Trim accumulated samples to
        keep memory bounded across many utterances."""
        # Trim the buffer; keep a tail equal to vad_leftover so VAD
        # alignment is preserved (vad_leftover indexes into `samples`).
        self.samples = self.vad_leftover.copy() if len(self.vad_leftover) else np.zeros(0, dtype=np.float32)
        self.samples_at_last_partial = 0
        self.last_partial_text = ""
        self.speech_start_idx = 0
        self.speech_end_pending = False
        self.utterance_id += 1  # invalidates any in-flight partials for this stream
        # in_speech stays False (we just finalized)


# ── Server ───────────────────────────────────────────────────────────


class STTServerV2:
    def __init__(self, model_path: str = MODEL_PATH, port: int = PORT):
        self.model_path = model_path
        self.port = port
        self.asr_model = None  # loaded in initialize()
        self.vad_model = None  # silero_vad model, loaded in initialize()
        self.client_streams: dict[str, ClientStream] = {}
        # Single lock around model.transcribe() because NeMo's transcribe
        # is not safe to call concurrently with itself on the same model.
        self._model_lock = asyncio.Lock()

        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins="*",
            max_http_buffer_size=10 * 1024 * 1024,
        )
        self.app = socketio.ASGIApp(self.sio)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"🔗 connect sid={sid[:8]}")
            await self.sio.emit("connection_status", {"status": "connected"}, room=sid)

        @self.sio.event
        async def disconnect(sid):
            logger.info(f"❌ disconnect sid={sid[:8]}")

        @self.sio.event
        async def client_disconnected(sid, data):
            client_id = (data or {}).get("clientId")
            if client_id and client_id in self.client_streams:
                logger.info(f"🧹 client_disconnected client_id={client_id}")
                del self.client_streams[client_id]

        @self.sio.event
        async def cleanup_client(sid, data):
            client_id = (data or {}).get("clientId")
            cleaned = []
            if client_id and client_id in self.client_streams:
                del self.client_streams[client_id]
                cleaned.append(client_id)
                logger.info(f"🧹 cleanup_client client_id={client_id}")
            await self.sio.emit(
                "cleanup_confirmed",
                {"clientId": client_id, "cleanedSessions": cleaned, "timestamp": time.time()},
                room=sid,
            )

        @self.sio.event
        async def subscribe_transcripts(sid, data):
            """Same protocol as v1: agent_server (or any listener) joins
            the room named by clientId to receive transcription events
            for that client."""
            client_id = (data or {}).get("clientId")
            if not client_id:
                return await self.sio.emit("error", {"msg": "missing clientId"}, room=sid)
            await self.sio.enter_room(sid, client_id)
            logger.info(f"👂 subscribe_transcripts sid={sid[:8]} room={client_id}")
            await self.sio.emit("subscribed", {"clientId": client_id}, room=sid)

        @self.sio.event
        async def audio_data(sid, data):
            """Per-chunk audio frame from the client.

            Format mirrors v1: dict with audioData (PCM16 bytes) + clientId,
            OR raw bytes (legacy). Audio is 16kHz mono PCM16 little-endian
            (already resampled by the noted-side AudioResampler)."""
            try:
                if isinstance(data, dict) and "audioData" in data and "clientId" in data:
                    audio_bytes = data["audioData"]
                    client_id = data["clientId"]
                else:
                    audio_bytes = data
                    client_id = f"client-{sid[:8]}"

                if not isinstance(audio_bytes, (bytes, bytearray)):
                    return  # unsupported shape; stay silent

                # Ensure the sender is in the per-client room (idempotent).
                await self.sio.enter_room(sid, client_id)

                # Get-or-create stream state for this client. New stream
                # gets its own VADIterator (Silero is stateless per call;
                # the iterator wraps it with per-stream state).
                if client_id not in self.client_streams:
                    logger.info(f"🆕 stream CREATED client_id={client_id}")
                    stream = ClientStream(client_id=client_id)
                    stream.vad_iterator = self._new_vad_iterator()
                    self.client_streams[client_id] = stream
                stream = self.client_streams[client_id]

                # PCM16 → float32 in [-1, 1].
                pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                speech_started, speech_ended = stream.add_samples(pcm)
                if speech_started:
                    logger.info(f"🎤 speech START client={client_id[:8]}")
                if speech_ended:
                    logger.info(f"🤫 speech END   client={client_id[:8]}")

                # Run partial inference if enough new audio has arrived
                # WITHIN the current speech window.
                if stream.should_emit_partial():
                    await self._emit_partial(stream, sid)

                # Finalize on VAD speech-end or buffer-cap overflow.
                if stream.should_finalize():
                    await self._emit_final(stream, sid)
            except Exception as e:
                logger.exception(f"audio_data error: {e}")

    async def _emit_partial(self, stream: ClientStream, sender_sid: str) -> None:
        """Run transcription on the CURRENT UTTERANCE (audio since the
        last VAD speech-start) and emit a partial event if the text
        changed since the last emit. Drops the partial if a final fired
        on the stream during transcription (utterance_id moved on)."""
        captured_uid = stream.utterance_id
        text = await self._transcribe(stream.utterance_audio())
        if stream.utterance_id != captured_uid:
            # A final committed while we were transcribing — this partial
            # belongs to the previous utterance; suppress it so the
            # client doesn't see a ghost partial after the final.
            return
        stream.samples_at_last_partial = len(stream.samples)
        stream.frame_counter += 1
        if text and text != stream.last_partial_text:
            stream.last_partial_text = text
            payload = {
                "text": text,
                "client_id": stream.client_id,
                "ts": time.time(),
                "frame_id": stream.frame_counter,
            }
            # Broadcast to subscribers (agent_server) AND echo to sender.
            await self.sio.emit("transcription_partial", payload, room=stream.client_id)
            await self.sio.emit("transcription_partial", payload, room=sender_sid)
            logger.info(
                f"📝 partial client={stream.client_id[:8]} "
                f"frame={stream.frame_counter} text={text!r}"
            )

    async def _emit_final(self, stream: ClientStream, sender_sid: str) -> None:
        """Commit the most recent partial as final, emit the v1-shape
        `transcription` event, and reset the buffer."""
        text = stream.last_partial_text
        if not text:
            stream.reset()
            return
        # Duration of the UTTERANCE (samples since the last speech-start),
        # not the session buffer length.
        duration = max(0.0, (len(stream.samples) - stream.speech_start_idx) / SAMPLE_RATE)
        payload = {
            "text": text,
            "duration": duration,
            "client_id": stream.client_id,
            "ts": time.time(),
        }
        # Same shape & event name as v1 → agent_server forwards unchanged.
        await self.sio.emit("transcription", payload, room=stream.client_id)
        await self.sio.emit("transcription", payload, room=sender_sid)
        logger.info(
            f"✅ FINAL client={stream.client_id[:8]} "
            f"dur={duration:.1f}s text={text!r}"
        )
        stream.reset()

    async def _transcribe(self, samples: np.ndarray) -> str:
        """Single transcription pass. NeMo's `transcribe()` is the
        simplest API; for MVP we re-run on the full buffer each call.
        If compute becomes a bottleneck we'll switch to NeMo's
        cache-aware streaming primitives (StreamingBatchedAudioBuffer +
        decoding_computer with prev_batched_state)."""
        if len(samples) == 0:
            return ""
        # Run inference off the event loop so we don't block other
        # incoming audio_data events. NeMo transcribe is GPU-bound so
        # the asyncio.to_thread offloads only the brief CPU coordination
        # bits; the GPU work itself doesn't yield.
        async with self._model_lock:
            try:
                result = await asyncio.to_thread(self._transcribe_sync, samples)
            except Exception as e:
                logger.exception(f"transcribe failed: {e}")
                return ""
        return (result or "").strip()

    def _transcribe_sync(self, samples: np.ndarray) -> str:
        # NeMo accepts a list of numpy arrays directly. Output is a list
        # of Hypothesis objects (NeMo 2.x) OR a list of strings (1.x).
        # Handle both shapes so we don't break across NeMo upgrades.
        out = self.asr_model.transcribe([samples], batch_size=1, verbose=False)
        if not out:
            return ""
        first = out[0]
        if isinstance(first, str):
            return first
        # NeMo 2.x: Hypothesis with .text attr
        if hasattr(first, "text"):
            return getattr(first, "text", "") or ""
        return str(first)

    async def initialize(self) -> None:
        """Load Parakeet model + Silero VAD. Called once before serving."""
        logger.info(f"🔄 Loading Parakeet model from {self.model_path}")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Place parakeet-tdt-0.6b-v3.nemo in data/models/parakeet/."
            )
        # Load on a worker thread so import + GPU init don't block startup
        # event loop coordination (uvicorn lifespan).
        self.asr_model = await asyncio.to_thread(self._load_model)
        logger.info("✅ Parakeet model loaded")

        logger.info("🔄 Loading Silero VAD")
        self.vad_model = await asyncio.to_thread(self._load_vad)
        logger.info(
            f"✅ Silero VAD loaded "
            f"(threshold={VAD_THRESHOLD}, min_speech_ms={VAD_MIN_SPEECH_MS}, "
            f"min_silence_ms={VAD_MIN_SILENCE_MS})"
        )

    def _load_model(self):
        import torch
        import nemo.collections.asr as nemo_asr
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"   Device: {device}")
        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=self.model_path,
            map_location=device,
        )
        model.eval()
        if device == "cuda":
            model = model.to(device)
        return model

    def _load_vad(self):
        import silero_vad
        return silero_vad.load_silero_vad()

    def _new_vad_iterator(self):
        """Per-stream VAD iterator. Silero's VADIterator wraps the shared
        model with per-call state (current speech boundaries). Cheap to
        construct; one per client_id."""
        import silero_vad
        return silero_vad.VADIterator(
            model=self.vad_model,
            threshold=VAD_THRESHOLD,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )

    async def serve(self) -> None:
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        logger.info(f"🚀 stt_server v2 listening on :{self.port}")
        await server.serve()


# ── Entry point ──────────────────────────────────────────────────────


async def main() -> None:
    # Optional settings file (mirrors v1's structure; presently we read
    # only env-vars but the file is here for parity with v1's deploy.)
    if Path(SETTINGS_PATH).exists():
        try:
            with open(SETTINGS_PATH, "r") as f:
                _ = json.load(f)
            logger.info(f"📄 settings file present at {SETTINGS_PATH}")
        except Exception as e:
            logger.warning(f"could not read {SETTINGS_PATH}: {e}")

    server = STTServerV2()
    await server.initialize()
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
