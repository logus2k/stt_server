# STT Server — Architecture Overview

> Companion text for `architecture.drawio`.

---

## 1. High-Level Summary

The STT Server is a single-process Python application that accepts real-time audio streams over **Socket.IO**, detects speech boundaries with **Silero VAD**, and transcribes complete utterances with the **Whisper Large v3 Turbo** model. An asynchronous event loop (Uvicorn / ASGI) keeps the server non-blocking while a dedicated thread pool handles the synchronous ML inference.

The entire runtime is encapsulated in the `STTServer` class defined in `stt_server.py`.

---

## 2. External Actors

Two types of clients connect to the server over Socket.IO (WebSocket transport):

| Actor | Role | Key Events |
|:---|:---|:---|
| **Audio Client** (browser, mobile app, or any Socket.IO client) | Streams raw PCM16 audio at 16 kHz and receives transcription results. | Sends `audio_data`; receives `transcription`. |
| **Monitoring Client** (LLM Assistant, Agent Server, or dashboard) | Subscribes to transcripts for a given `clientId` without sending audio. Can also request session cleanup. | Sends `subscribe_transcripts`, `cleanup_client`; receives `transcription` (via room broadcast). |

---

## 3. Network / Transport Layer

Incoming connections flow through three stacked components:

```
Uvicorn (ASGI Server)
  └─ ASGIApp
       └─ Socket.IO AsyncServer (python-socketio)
```

- **Uvicorn** binds to `host:port` (default `0.0.0.0:2700`) and serves the ASGI application.
- **ASGIApp** is a thin wrapper provided by `python-socketio` that adapts the Socket.IO server to the ASGI interface.
- **Socket.IO AsyncServer** manages WebSocket connections and dispatches incoming messages to the registered **Event Handlers**.

### Event Handlers

All client-server interaction is routed through six Socket.IO events:

| Event | Direction | Purpose |
|:---|:---|:---|
| `connect` | Client → Server | Initializes a session and creates a per-client `_ClientTranscriber`. |
| `disconnect` | Client → Server | Tears down the session and removes the transcriber. |
| `audio_data` | Client → Server | Delivers a chunk of PCM16 audio (with an optional `clientId`). |
| `subscribe_transcripts` | Client → Server | Joins the caller into the Socket.IO room for a given `clientId`, so it receives all transcription broadcasts for that client. |
| `cleanup_client` | Client → Server | Requests deletion of the transcriber state for a specific `clientId`. Returns a `cleanup_confirmed` acknowledgement. |
| `client_disconnected` | Client → Server | Notifies the server that an external client has left, triggering session cleanup by `clientId`. |

### Socket.IO Rooms

Each `clientId` maps to a Socket.IO room. When a client sends `audio_data` with a `clientId`, the sender is automatically joined to that room. Monitoring clients join the same room via `subscribe_transcripts`. Transcription results are broadcast to the room so both the audio sender and any subscribers receive them.

---

## 4. Client Session Management

### `client_transcribers` Dictionary

The server maintains a dictionary `client_transcribers: Dict[client_id, _ClientTranscriber]`. Event handlers create, look up, and delete entries in this dictionary as clients connect, send audio, and disconnect.

### `_ClientTranscriber` (Inner Class)

Each connected client gets its own `_ClientTranscriber` instance, which holds:

| Attribute / Method | Description |
|:---|:---|
| `audio_buffer` | A rolling list of float32 samples (max 10 seconds). |
| `last_processed_time` | Timestamp of the last segment that was transcribed, used to avoid re-processing. |
| `add_audio(samples)` | Appends new samples to the buffer and trims excess beyond the 10-second window. |
| `get_ready_segment()` | Rate-limited method that runs Silero VAD on the buffer, checks for a completed speech segment (enough trailing silence), and returns it as a NumPy array — or `None` if nothing is ready yet. |

The rate-limiting is controlled by `processing_interval` (default 0.3 s), which prevents the VAD from scanning on every incoming audio chunk.

---

## 5. ML Inference Layer

### Silero VAD

Called synchronously inside `_ClientTranscriber.get_ready_segment()`. It runs `get_speech_timestamps()` on the current audio buffer to locate speech boundaries. A segment is considered ready when:

1. The segment duration >= `min_speech_duration` (default 0.4 s).
2. The trailing silence after the segment >= `silence_duration` (default 0.8 s).

These thresholds, together with `vad_threshold` (sensitivity), form the primary tuning surface for balancing latency, false positives, and missed speech. See `CONFIG.md` for tuning profiles.

### ThreadPoolExecutor

The Whisper model runs synchronously (PyTorch inference), but the server's event loop is asynchronous. A `ThreadPoolExecutor` (sized by `max_workers`) bridges this gap. The `audio_data` event handler calls `_transcribe_async()`, which delegates to `asyncio.run_in_executor()`, ensuring that inference does not block the event loop.

### Whisper Large v3 Turbo

The transcription model, loaded at startup via `initialize()`:

- **Model**: `AutoModelForSpeechSeq2Seq` from Hugging Face Transformers.
- **Processor**: `AutoProcessor` for feature extraction and decoding.
- **Optimizations**: The model is compiled with `torch.compile(mode="reduce-overhead", fullgraph=True)` and runs in `torch.inference_mode()`. It uses `float16` on CUDA and `float32` on CPU.
- **Generation**: Greedy decoding (`num_beams=1`, `do_sample=False`) with a max of 200 new tokens per segment.

The synchronous entry point is `_transcribe_sync(audio_segment, client_id)`, which preprocesses the audio, runs generation, and decodes the output tokens into text.

---

## 6. Output Paths

When a segment is transcribed, the result follows two parallel paths:

1. **Socket.IO emission** — The transcription payload (`text`, `duration`, `client_id`, `ts`) is broadcast to the `clientId` room (reaching both the audio sender and any subscribed monitoring clients) and also emitted directly to the sender's `sid` for legacy client compatibility.
2. **Callback hook** — If an `on_transcription` callable was provided (via constructor or `from_config`), it is invoked with `(text, client_id, duration)`. This is the integration point for downstream systems such as an LLM chatbot.

---

## 7. Configuration

All server parameters are loaded from a JSON file via the `STTServer.from_config()` class method. The default path used in `main()` is:

```
data/configuration/stt.server.settings.json
```

The configuration feeds every tunable parameter (model path, network bind, VAD thresholds, thread pool size, etc.) directly into the `STTServer` constructor. If the file is missing, the server falls back to hard-coded defaults. See `CONFIG.md` for the full parameter reference and ready-to-use tuning profiles.

---

## 8. Docker Deployment

The project uses a two-stage Docker build:

| Image | Dockerfile | Contents |
|:---|:---|:---|
| `stt_server-server:1.0` (base) | `stt_server.server.Dockerfile` | Python 3.10-slim with all pip dependencies (`transformers`, `accelerate`, `python-socketio`, `silero-vad`, `uvicorn`). |
| `stt_server:1.0` (app) | `stt_server.Dockerfile` | Extends the base image. Copies `stt_server.py`, sets the working directory to `/stt_server`, and exposes port 2700. |

This separation means the heavy dependency layer is cached independently from the application code, so code-only changes result in fast image rebuilds.

At runtime, model weights and the configuration file are expected to be available inside the container (typically via a volume mount to `/stt_server/data/`).

---

## 9. Data Flow Summary

```
Audio Client                                          Monitoring Client
     │                                                       │
     │  audio_data (PCM16 @ 16kHz)                           │  subscribe_transcripts
     ▼                                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Socket.IO AsyncServer  →  Event Handlers                      │
│       │                         │                              │
│       │ create/lookup           │ join room                    │
│       ▼                         ▼                              │
│  client_transcribers    Socket.IO Rooms                        │
│       │                    (per clientId)                       │
│       ▼                         ▲                              │
│  _ClientTranscriber             │                              │
│       │                         │ emit transcription           │
│       │ get_ready_segment()     │                              │
│       ▼                         │                              │
│  Silero VAD                     │                              │
│       │                         │                              │
│       │ speech segment          │                              │
│       ▼                         │                              │
│  ThreadPoolExecutor ──► Whisper ─┘                             │
│                                                                │
│                    on_transcription(text, id, dur) ──► Callback │
└─────────────────────────────────────────────────────────────────┘
```
