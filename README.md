# Real-Time Speech-to-Text (STT) Server

A high-performance server for real-time, low-latency Speech-to-Text transcription. This project uses the powerful [Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) model combined with [Silero Voice Activity Detection (VAD)](https://github.com/snakers4/silero-vad) to efficiently process audio streams delivered via [Socket.IO](https://socket.io/).

The server leverages an asynchronous architecture [(Uvicorn/ASGI)](https://uvicorn.dev/) and a multi-threading approach to handle the synchronous nature of the machine learning model, ensuring non-blocking performance while transcribing.

## Features

* **Real-Time Streaming:** Transcribes audio data streamed continuously from clients via Socket.IO.
* **Voice Activity Detection (VAD):** Uses Silero VAD to intelligently segment audio, detecting complete phrases based on silence duration before sending them to Whisper for transcription.
* **Asynchronous Performance:** Built on `uvicorn` and `python-socketio` for high concurrency and efficient handling of multiple simultaneous clients.
* **Whisper Integration:** Utilizes the Hugging Face `transformers` library to run the Whisper model for state-of-the-art accuracy.
* **Configurable:** Supports loading server and model parameters from a JSON configuration file.
* **Callback System:** Provides a Python callback hook for immediate integration with backend logic (e.g., an LLM or chatbot).

---

## Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **STT Model** | Whisper | State-of-the-art speech recognition model. |
| **VAD** | Silero VAD | Used for accurate voice activity detection and segmentation. |
| **Server Framework** | Uvicorn / ASGI | High-performance asynchronous server to host the application. |
| **Real-Time I/O** | python-socketio | Handles WebSocket communication for streaming audio data. |
| **ML Libraries** | PyTorch, Transformers, Accelerate | Core libraries for model loading and inference. |

---

## Installation and Setup

Follow these steps to set up the server environment and run the application.

### 1. Clone the repository

```bash
git clone [https://github.com/logus2k/stt_server.git](https://github.com/logus2k/stt_server.git)
cd stt_server
```

### 2\. Set up the Python Environment

It is highly recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3\. Install Dependencies

Install all necessary packages, including `transformers`, `uvicorn`, and `python-socketio`.

```bash
pip install --upgrade pip 
pip install transformers accelerate python-socketio uvicorn silero-vad
```

### 4\. Model Setup

The server is configured to load a model (default: `whisper-large-v3-turbo`). Ensure your model weights are accessible at the path specified in your configuration file or the default server settings.

-----

## Usage

### 1\. Configuration (Optional)

You can customize the server by creating a configuration file named `stt.server.settings.json` in a path accessible by the server (e.g., `data/configuration/stt.server.settings.json` if using the example `main()` block).

Example `stt.server.settings.json` (using default values):

```json
{
    "model_path": "models/whisper-large-v3-turbo",
    "port": 2700,
    "host": "0.0.0.0",
    "silence_duration": 0.8,
    "min_speech_duration": 0.4,
    "vad_threshold": 0.4
}
```

### 2\. Run the Server

Execute the main server file to start the application. The server will automatically load the models (if not already loaded) and begin listening.

```bash
python stt_server.py
```

The server will typically start on `http://0.0.0.0:2700`.

-----

## Client-Server Protocol (Socket.IO Events)

Clients communicate with the server using standard Socket.IO events:

| Event Name | Direction | Payload | Description |
| :--- | :--- | :--- | :--- |
| `audio_data` | Client -\> Server | `bytes` or `{'audioData': bytes, 'clientId': str, 'language'?: str or None, 'pause'?: float}` | Stream raw PCM16 (16kHz) audio data. Passing `language` (a Whisper code such as `pt`, or null to detect) opts the clientId into the **clip decoder** (see below); without it the stream is handled as before. |
| `transcription` | Server -\> Client | `{'text': str, 'duration': float, 'client_id': str, 'start': float, 'end': float, 'rejected'?: str or None, 'words'?: [[text, start, end], ...], ...}` | A confirmed speech segment has been transcribed. `start`/`end`: its place in the sender's stream (seconds from its first audio); segments can finish out of order. Opt-in senders also get `words`: each word with its time in the stream (for giving each word its speaker). Legacy senders get each one twice (the clientId's room and their own socket); opt-in senders once, including rejected segments (empty text, `rejected` set). |
| `audio_end` | Client -\> Server (with ack) | `{'clientId': str}` → `{'segments': int}` | The audio has ended: segments being transcribed finish, the speech still in the buffer is transcribed without waiting for a pause, and the ack comes once all are sent. |
| `subscribe_transcripts` | Client -\> Server | `{'clientId': str}` | Allows a monitoring client (e.g., an LLM Assistant) to receive all transcripts for a specific user ID. |
| `cleanup_client` | Client -\> Server | `{'clientId': str}` | Request the server to clear the audio buffer for a specific client session. |
| `transcribe` | Client -\> Server (with ack) | `{'audio': bytes, 'language': str or None}` → `{'text': str, 'rejected': str or None}` | Transcribe one finished clip (PCM16 16 kHz mono, e.g. a speaker turn; up to 32 MB). The language is fixed (Whisper code such as `pt`), or None to detect it. Clips over 28 s are cut at VAD pauses. Each piece is decoded greedily, then again at rising temperatures while the text repeats itself (zlib ratio > 2.4) or holds more than 6 words/s. `rejected` is `silence`, `repetition` or an error. Optional `'words': true`: the answer also has `segments`, each `{start, end, text, words: [[text, start, end], ...]}` in seconds from the clip's start, and a piece that keeps looping is dropped alone instead of rejecting the whole clip. Optional `'prompt'`: text Whisper takes as preceding context, at most 200 tokens, e.g. a vocabulary. Measured on meeting clips, a 12-name glossary fixed a few spellings but invented one and did not reduce errors overall, so Cortex does not send one. Used by Cortex (diarization/refine.py). The live `audio_data` path is unaffected. |

-----

### Live clip decoder (opt-in: `language` in `audio_data`)

Used by Cortex's live transcript (Settings > Transcription > Live transcript: Whisper). For such a clientId:

- **Quiet speech is raised before the VAD.** Each packet is raised so the 90th-percentile packet level of
  the last ~30 s reaches -20 dBFS. The gain is at most 40 dB, never lowers the audio, and is capped by
  the loudest peak of the last ~1 s. A soft limiter follows.
- **Segments end after a longer pause** (`pause`, default 0.8 s; Cortex sends 1.5 s) and hold at least
  1 s of speech. That removes the 0.3–0.7 s noise segments Whisper turned into "Obrigado." or "Nossa.".
- **Each segment is decoded like `transcribe`:** the language is fixed, with temperature fallback on
  repetition and impossible speech rates.
- **Word times.** Each word gets a time from Whisper's cross-attention alignment heads, decoded on the
  uncompiled model (the timings need the attention weights). That's ~2.9 s for a 28 s segment on the
  GPU, instead of ~0.9 s. Word ends are accurate to about 0.3 s (measured against a reading's pauses);
  a word's start is where the previous word ended, so a pause is counted in the next word.
- **Forced cuts at a quiet point.** A segment forced out at the 30 s buffer cap (nobody paused) is cut
  at the quietest 0.3 s of its last 4 s, not wherever the cap fell. Cut mid-word, Whisper lost the words
  around the cut and the next segment's start (5 s of a second reader).
- **A segment that starts exactly where the last one ended** (speech running on) is transcribed. The
  legacy path counts it as already done, and loses it when the buffer reaches its 30 s cap.

Measured 2026-09-30, recordings streamed in real time:

| recording | errors, before (legacy live path) | errors, opt-in |
| --- | --- | --- |
| a sonnet read into a phone (99 words) | 2% | 2% |
| the same, another reader, −49 dBFS | 100%, all dropped as silence | 2% |
| a TV across a room, −60 dBFS | nothing transcribed | 76 s transcribed, coherent |

The speech that ran straight on after a 29 s segment (a phone recording's last 16 s) was lost before the
`<` fix. It now comes back at `audio_end`, 0.7 s after it is sent.

Known limits:
- Whisper sometimes returns only part of a long segment of raised background noise (up to 29 s when
  nobody pauses): a TV segment came back as dots.
- In continuous multi-speaker talk a segment spans several speakers.

### Connections and cleanup

A connection's per-clientId buffers are freed when it disconnects, unless another connection still sends
under the same clientId. Before 2026-09-30 they were keyed by clientId but deleted by connection id, so
they were never freed, and a sender that reconnected under the same clientId had the old connection's
leftover audio transcribed into its new session.

## License

This project is licensed under the **Apache License 2.0**.

---