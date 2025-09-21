# STT Server Configuration Guide

This server performs real-time speech-to-text using **Whisper (HF Transformers)** and **Silero VAD** with **Socket.IO** transport.

## Parameters (what they do)

| Key                   | Type         |                           Default | What it controls                                                                                               | Trade-offs / Notes                                                                                                   |
| --------------------- | ------------ | --------------------------------: | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `model_path`          | string       | `"models/whisper-large-v3-turbo"` | Local path to a Whisper model.                                                                                 | Larger models → better accuracy, higher latency/VRAM.                                                                |
| `port`                | int          |                            `2700` | TCP port for the Socket.IO server.                                                                             | —                                                                                                                    |
| `host`                | string       |                       `"0.0.0.0"` | Bind interface.                                                                                                | Keep `0.0.0.0` if running in a container.                                                                            |
| `silence_duration`    | float (s)    |                             `0.8` | How much trailing silence must be observed before a detected speech segment is **closed** and sent to Whisper. | Higher = fewer mid-sentence chops but higher latency.                                                                |
| `min_speech_duration` | float (s)    |                             `0.4` | Minimum speech length to consider a valid segment.                                                             | Higher = fewer noise blips, but short utterances (e.g., “yes”) may be ignored.                                       |
| `vad_threshold`       | float \[0–1] |                             `0.4` | VAD sensitivity.                                                                                               | **Higher** (0.55–0.7) = stricter (fewer false positives). **Lower** (0.3–0.45) = more sensitive (easier to trigger). |
| `processing_interval` | float (s)    |                             `0.3` | How often the VAD scans the buffer for segments.                                                               | Lower = lower latency, higher CPU. Typical 0.2–0.5s.                                                                 |
| `max_workers`         | int          |                               `2` | Thread-pool size for Whisper inference.                                                                        | On GPU, 1–2 is usually best. Too many threads can degrade throughput.                                                |
| `sample_rate`         | int (Hz)     |                           `16000` | Expected audio sampling rate.                                                                                  | Keep at 16 kHz; resample at the client if needed.                                                                    |
| `enable_logging`      | bool         |                            `true` | Prints connection and transcript logs.                                                                         | Set `false` to reduce noise in prod.                                                                                 |

> Whisper language: by default, multilingual Whisper **auto-detects language** and then transcribes. The console warning you may see is expected. Forcing a language would require a small code change (not part of this server’s current config).

---

## Tuning patterns

Think of **false positives** vs **missed speech** and **latency** as a triangle: improving one usually hurts another. The three most impactful knobs are:

* **`vad_threshold`** – sensitivity (raise to reduce phantom triggers).
* **`silence_duration`** – how long to wait before closing a segment (raise to avoid cutting sentences, but adds latency).
* **`min_speech_duration`** – ignore very short bursts (raise to suppress blips, but you might drop “ok”, “yes”, etc.).

> Client capture helps a lot: request `echoCancellation:true`, `noiseSuppression:true`, and **`autoGainControl:false`** (see rationale below).

---

## Ready-to-use profiles

### 1) Balanced (good default)

```json
{
  "model_path": "/stt_server/data/models/whisper-large-v3-turbo",
  "host": "0.0.0.0",
  "port": 2700,
  "silence_duration": 0.9,
  "min_speech_duration": 0.45,
  "vad_threshold": 0.55,
  "processing_interval": 0.25,
  "max_workers": 2,
  "sample_rate": 16000,
  "enable_logging": true
}
```

* Cuts most phantom triggers while keeping short replies.
* Slightly higher end-of-utterance latency than the factory defaults.

### 2) Conservative (quiet rooms / reduce false positives)

```json
{
  "model_path": "/stt_server/data/models/whisper-large-v3-turbo",
  "host": "0.0.0.0",
  "port": 2700,
  "silence_duration": 1.0,
  "min_speech_duration": 0.6,
  "vad_threshold": 0.65,
  "processing_interval": 0.3,
  "max_workers": 2,
  "sample_rate": 16000,
  "enable_logging": true
}
```

* Much stricter VAD; handles background noise / fan hum better.
* Requires clearer speech; may miss very short utterances.

### 3) Permissive (hands-free, far mic, soft speech)

```json
{
  "model_path": "/stt_server/data/models/whisper-large-v3-turbo",
  "host": "0.0.0.0",
  "port": 2700,
  "silence_duration": 0.7,
  "min_speech_duration": 0.35,
  "vad_threshold": 0.45,
  "processing_interval": 0.2,
  "max_workers": 2,
  "sample_rate": 16000,
  "enable_logging": true
}
```

* Triggers easily; better for distant/quiet talkers.
* Expect more false positives; pair with client-side gating if possible.

### 4) Low-latency (snappier turn-taking)

```json
{
  "model_path": "/stt_server/data/models/whisper-large-v3-turbo",
  "host": "0.0.0.0",
  "port": 2700,
  "silence_duration": 0.8,
  "min_speech_duration": 0.4,
  "vad_threshold": 0.55,
  "processing_interval": 0.2,
  "max_workers": 1,
  "sample_rate": 16000,
  "enable_logging": true
}
```

* Faster segment closure checks; slightly higher CPU.
* Keep `max_workers: 1` on lower-end GPUs/CPUs to avoid contention.

---

## Why disable AGC (Automatic Gain Control)?

**TL;DR:** In this pipeline, **browser-level AGC tends to *increase* false positives**. Prefer `autoGainControl:false` and rely on VAD tuning.

**Reasoning:**

* **AGC “pumps” background noise.** During silence, it raises gain to hit a target loudness. Room noise gets amplified until it looks like speech, and the VAD fires.
* **Inconsistent levels across frames.** AGC’s fast attack/slow release can distort amplitude patterns the VAD relies on, especially around speech onsets/offsets.
* **You already control sensitivity.** With `vad_threshold`, `min_speech_duration`, and `silence_duration`, you can dial in sensitivity without AGC side-effects.

**When might AGC help?**

* Very low-output mics or mobile devices in highly variable conditions. If you must enable AGC, consider:

  * Raising `vad_threshold` by \~0.05–0.1,
  * Raising `min_speech_duration` (e.g., +0.1–0.2s),
  * Adding a tiny client-side **RMS gate** (discard frames where average absolute level is below a floor).

**Recommended browser constraints:**

```js
navigator.mediaDevices.getUserMedia({
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: false   // ← recommended
  },
  video: false
});
```

---

## Quick troubleshooting

* **Phantom “Thank you” / random words when you’re silent**

  * Raise `vad_threshold` (e.g., 0.6–0.65),
  * Raise `min_speech_duration` (0.5–0.7),
  * Ensure `autoGainControl:false`, and reduce fan/keyboard noise if possible.
* **Cuts you off mid-sentence**

  * Increase `silence_duration` (+0.1–0.3),
  * Increase `processing_interval` slightly (fewer premature closures).
* **Misses very short replies (“yes”, “ok”)**

  * Lower `min_speech_duration` (0.3–0.4),
  * Slightly lower `vad_threshold` (0.5 → 0.45).
* **Latency feels high**

  * Lower `silence_duration` (0.8 → 0.7),
  * Lower `processing_interval` (0.3 → 0.2) if CPU allows.

---

## Integration checklist (client side)

* Send **binary PCM 16-bit mono** at **16kHz** (resample locally if needed).
* Use the new payload shape:

  ```js
  sttSocket.emit("audio_data", {
    clientId: "<your-correlated-id>",
    audioData: pcm16.buffer // ArrayBuffer
  });
  ```
* Subscribe to:

  ```js
  sttSocket.emit("subscribe_transcripts", { clientId });
  sttSocket.on("transcription", ({ text, duration, client_id }) => { ... });
  ```
* Recommended capture flags: `echoCancellation:true`, `noiseSuppression:true`, `autoGainControl:false`.
