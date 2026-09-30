# stt_server.py

"""
pip install --upgrade pip 
pip install transformers accelerate python-socketio uvicorn silero-vad
"""

import asyncio
import math
import socketio
import uvicorn
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import silero_vad
from concurrent.futures import ThreadPoolExecutor
import time
import zlib
from collections import deque
from typing import Optional, Callable, Dict, Any
import logging
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Whisper hallucination defenses. Empirically validated against captured
# failures (see project memory / data/probes/stt_real_audio_probe.py):
#   - whisper-large-v3-turbo confidently emits "Thank you." / "Mm-hmm." on
#     silence and on bass-heavy mouth noise.
#   - P(<|nospeech|>) at the first decoder step is 0.0 for both cases —
#     the model is not trained to flag these via the nospeech token.
#   - Avg logprob is well above any usable threshold because once the
#     model emits " Thank", subsequent tokens "you", "." follow at ~99%
#     probability ("autoregressive lock-in").
# Three layered defenses each catch a different signature:
# ---------------------------------------------------------------------------

# Layer 1 — pre-Whisper RMS gate. A clip below this energy level cannot
# plausibly contain intelligible speech (typical conversational speech is
# -20 to -25 dBFS RMS; soft whisper into a mic is -35 to -40 dBFS).
# Catches the silence false-positives that escape Silero (e.g., the
# 20260504_033626 file: -63.9 dBFS RMS, transcribed as "Thank you").
MIN_AUDIO_RMS_DBFS = -45.0

# Layer 2 — post-Whisper first-content-token probability gate. The
# autoregressive lock-in inflates avg_logprob, but the FIRST content
# token's probability accurately reflects the decoder's confidence in
# what the audio actually says. On the captured failures: first-token
# P("Thank") = 0.10 (mouth noise) and 0.29 (silence) — both well below
# 0.6. On real speech the first content token is typically > 0.5.
# Combined with the duration cap below to avoid clipping legitimate
# short utterances where a real word has many plausible competitors.
FIRST_TOKEN_PROB_THRESHOLD = 0.3
# Was 1.5s. Lowered 2026-05-05 to stop swallowing brief intentional words
# like "yes", "okay", "continue", "stop". Whisper's silence/mouth-noise
# hallucinations are typically <1s anyway (the captured failures clustered
# around 0.4-0.8s), so the filter still catches them at 1.0s while letting
# real short replies through.
SHORT_AUDIO_DURATION_S = 1.0

# Layer 3 — post-Whisper canonical hallucination blocklist. When the
# model hallucinates on a short clip, the output is heavily concentrated
# on a small set of training-data sign-offs (YouTube end-of-video
# captions are the dominant source). Match is normalised: case + non-
# alphanumerics stripped, so "Thank you.", "thank you!", " Thank you "
# all collapse to "thankyou". Only triggers on clips < SHORT_AUDIO_DURATION_S
# so a real "Thank you" mid-conversation isn't dropped.
#
# Multi-language entries cover the canonical non-English Whisper
# hallucinations on silence — added when we un-pinned `language` from 'en'
# to support multilingual STT. Korean broadcast sign-offs and
# language-specific "thanks for watching" / "subscribe" patterns are the
# most reported. CJK characters survive the normalize-to-alphanum pass
# (Python's str.isalnum() accepts them), so the entries below match
# Whisper's typical normalisation output.
HALLUCINATION_BLOCKLIST = {
    # English (most common)
    "thankyou",
    "thanksforwatching",
    "thanksforwatchingthevideo",
    "thanksforlistening",
    "thanks",
    "you",
    "mmhmm",
    "uhhuh",
    "bye",
    "byebye",
    "goodbye",
    "subscribe",
    "subscribetothechannel",
    "subscribetomychannel",
    "amaraorg",
    "amarasubtitles",
    "transcriptionbyamaraorg",
    # Korean — the most-reported non-English Whisper hallucination on
    # silence (broadcast news sign-off pattern from training data).
    "mbc뉴스김성현입니다",
    "kbs뉴스",
    "ytn뉴스",
    "시청해주셔서감사합니다",  # "thanks for watching" sign-off
    "구독과좋아요부탁드립니다",  # "subscribe and like" sign-off
    # Japanese (similar end-of-video captions)
    "ご視聴ありがとうございました",   # "thank you for watching"
    "チャンネル登録お願いします",   # "please subscribe"
    # Spanish
    "graciasporver",
    "graciasporverelvideo",
    "suscribete",
    "suscríbete",
    # Brazilian Portuguese
    "obrigadoporassistir",
    "inscrevase",
    "inscreva",
    # French
    "mercidavoirregardé",
    "mercidavoirregarde",
    "abonnezvous",
    # Italian
    "grazieperlavisione",
    "iscriviti",
}


def _normalize_for_blocklist(s: str) -> str:
    """Lowercase + strip non-alphanumerics for blocklist comparison."""
    return "".join(c.lower() for c in s if c.isalnum())


# Maximum unprocessed audio retained in a client's rolling buffer.
# History:
#   Original:  10s rolling rebase (truncated >10s utterances at the front).
#   2026-05-05: bumped to 60s to fix the truncation. That introduced a
#               *VAD compute regression*: Silero VAD on a 60s buffer takes
#               ~500-800ms on CPU, which exceeds processing_interval (0.3s).
#               Once a long utterance accumulates, VAD runs back-to-back
#               on the full buffer, never idle, CPU pegs at ~100%, and
#               segment emissions stall by 15-20s.
#   2026-05-06: lowered to 30s — matches Whisper-large-v3-turbo's native
#               encoder cap (segments longer than 30s would be truncated
#               by the processor anyway, so the 30→60 headroom was
#               always wasted). Silero VAD on 30s runs in ~250-400ms,
#               comfortably bounded under processing_interval most of
#               the time. Full > 30s support requires chunked Whisper
#               decoding (backlog STT-1).
MAX_BUFFER_SECONDS = 30

# `transcribe` (whole clips, e.g. Cortex's diarized speaker turns). Measured on
# meeting audio: Whisper decoding each turn with the language fixed made 5-16
# errors per 5 min where the streaming recogniser made 30-37; the same audio
# given whole looped ("não, não, ..." x219). These guards are Whisper's own
# (temperature fallback on a compression ratio above 2.4) plus a speech-rate cap.
CLIP_PIECE_SECONDS = 28.0
CLIP_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CLIP_MAX_COMPRESSION = 2.4
CLIP_MAX_WORDS_PER_SECOND = 6.0
# Live segments of a sender that passes `language` in audio_data (opt-in; Cortex): decoded as clips
# (language fixed, the checks above) after raising quiet speech. Measured 2026-09-29 on phone and room
# recordings arriving at -42 to -60 dBFS: the live path's RMS gate dropped them as silence (a sonnet read
# into a phone: every segment), while the clip decoder with the language fixed and the audio raised
# transcribed the same readings with 2 errors in 99 words. One gain per segment: its RMS to
# LIVE_TARGET_DBFS, never lowered, capped so the loudest sample stays below LIVE_PEAK_DBFS and by
# LIVE_MAX_GAIN_DB; a segment below LIVE_MIN_RMS_DBFS is still silence.
LIVE_TARGET_DBFS = -20.0
LIVE_PEAK_DBFS = -1.0
LIVE_MAX_GAIN_DB = 40.0
LIVE_MIN_RMS_DBFS = -65.0
# ...and raised BEFORE the VAD decides where speech is: on audio arriving at -60 dBFS (a TV across a room)
# Silero found 15 s of 76 s of speech; raised first, all of it (measured 2026-09-30). The gain follows the
# stream: its 90th-percentile packet level over the last LIVE_LEVEL_WINDOW packets brought to
# LIVE_TARGET_DBFS (never lowered, at most LIVE_MAX_GAIN_DB), with a soft limiter against clipping.
LIVE_LEVEL_WINDOW = 300          # packets (~30 s of 100 ms packets)
# the gain is also capped by the loudest peak of the last LIVE_PEAK_WINDOW packets (the current one among
# them): at the start the level window holds only silence, and the first words, raised as much as that
# silence, were flattened by the limiter ("Como" lost). Over the whole level window, one click at -22 dBFS
# held a TV's speech (-60 dBFS) to +20 dB, too quiet for the VAD (76 s of speech -> 38 s found).
LIVE_PEAK_WINDOW = 10            # packets (~1 s)
# the pause that ends a segment for opt-in senders (they may choose it: `pause` in audio_data)
LIVE_PAUSE_RANGE = (0.3, 5.0)
# opt-in: speech shorter than this is not a segment. Raised background noise made 0.3-0.7 s segments that
# Whisper turned into "Obrigado.", "Nossa.", "Seus amigos." (a meeting room, measured 2026-09-30)
LIVE_MIN_SPEECH_SEC = 1.0
# opt-in: a segment forced out at the buffer cap is cut at the quietest LIVE_CUT_QUIET_SEC of its last
# LIVE_CUT_SEARCH_SEC (the rest stays in the buffer for the next one)
LIVE_CUT_QUIET_SEC = 0.3
LIVE_CUT_SEARCH_SEC = 4.0

# `transcribe`'s optional prompt (e.g. a vocabulary of names): Whisper was trained with at
# most 224 prompt tokens, and prompt + transcript share the decoder's 448 positions.
CLIP_MAX_PROMPT_TOKENS = 200


class STTServer:
    """
    Real-time speech-to-text server using Whisper and Silero VAD with Socket.IO.
    
    Usage:
        server = STTServer(
            model_path="models/whisper-large-v3-turbo",
            on_transcription=my_callback_function
        )
        await server.start()
    """
    
    def __init__(
        self,
        model_path: str = "models/whisper-large-v3-turbo",
        port: int = 2700,
        host: str = "0.0.0.0",
        on_transcription: Optional[Callable[[str, str, float], None]] = None,
        silence_duration: float = 0.8,
        min_speech_duration: float = 0.5,
        vad_threshold: float = 0.5,
        processing_interval: float = 0.3,
        max_workers: int = 2,
        sample_rate: int = 16000,
        enable_logging: bool = True
    ):
        """
        Initialize the Whisper server.
        
        Args:
            model_path: Path to Whisper model
            port: Server port
            host: Server host
            on_transcription: Callback function(text, client_id, duration) called when text is transcribed
            silence_duration: Seconds of silence before processing speech
            min_speech_duration: Minimum speech length to process
            vad_threshold: Voice activity detection sensitivity (0-1)
            processing_interval: How often to check for segments
            max_workers: Thread pool size for transcription
            sample_rate: Audio sample rate (Hz)
            enable_logging: Enable debug logging
        """
        self.model_path = model_path
        self.port = port
        self.host = host
        self.on_transcription = on_transcription
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.vad_threshold = vad_threshold
        self.processing_interval = processing_interval
        self.max_workers = max_workers
        self.sample_rate = sample_rate
        self.enable_logging = enable_logging
        
        # Initialize components
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.whisper_model = None
        self.processor = None
        self.vad_model = None
        self.clip_vad_model = None
        self.thread_pool = None
        self.server = None
        self.active_transcriptions = 0
        
        # Socket.IO setup
        self.sio = socketio.AsyncServer(
            cors_allowed_origins="*",
            logger=False,  # ✅ Disable Socket.IO internal logging
            engineio_logger=False,  # ✅ Disable EngineIO internal logging
            async_mode='asgi',
            # `transcribe` carries a whole speaker turn (PCM16 16 kHz = 32 KB/s); the
            # default 1 MB dropped the connection on turns over ~31 s (measured)
            max_http_buffer_size=32 * 1024 * 1024
        )
        self.app = socketio.ASGIApp(self.sio, other_asgi_app=None)
        
        # Client transcriber storage, keyed by the clientId a sender chose (or client-<sid>)
        self.client_transcribers = {}
        # the clientIds each connection sent audio under: freed when it disconnects (keyed by
        # clientId, a transcriber outlived its connection - memory kept for good, and a sender
        # reconnecting under the same clientId had the old connection's leftover audio
        # transcribed into its new session; measured 2026-09-29)
        self.sid_clients: Dict[str, set] = {}
        
        # Setup Socket.IO event handlers
        self._setup_socketio_handlers()
        
        # Setup logging
        if self.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)

    @classmethod
    def from_config(cls, config_path: str = "stt.server.settings.json"):
        """Create server instance from configuration file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"⚠️ Config file {config_path} not found, using defaults")
            return cls()
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return cls(**config)
    
    def _setup_socketio_handlers(self):
        """Setup Socket.IO event handlers."""
        
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection."""
            client_info = f"client-{sid[:8]}"
            # ✅ ESSENTIAL: Session status change
            if self.enable_logging:
                print(f"🔗 Session STARTED: {client_info}")
            
            # Create transcriber for this client
            self.client_transcribers[sid] = self._ClientTranscriber(self)
            
            # Send connection confirmation
            await self.sio.emit('connection_status', {'status': 'connected'}, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection."""
            client_info = f"client-{sid[:8]}"
            # ✅ ESSENTIAL: Session status change
            if self.enable_logging:
                print(f"❌ Session ENDED: {client_info}")
            
            # Clean up its transcribers: the one made on connect, and those of the clientIds it
            # sent audio under, unless another connection still sends under the same clientId
            self.client_transcribers.pop(sid, None)
            for client_id in self.sid_clients.pop(sid, set()):
                if not any(client_id in ids for ids in self.sid_clients.values()):
                    self.client_transcribers.pop(client_id, None)

        @self.sio.event
        async def client_disconnected(sid, data):
            """Handle notification from LLM Assistant that a client disconnected."""
            client_id = data.get('clientId')
            # ✅ ESSENTIAL: Session cleanup status
            if self.enable_logging:
                print(f"🧹 Session CLEANUP requested: {client_id}")
            
            # Find and clean up any transcriber sessions for this client
            # Since we store transcribers by Socket.IO session ID (sid), we need to
            # check if any transcriber corresponds to this client_id
            transcribers_to_remove = []
            
            for session_id, transcriber in self.client_transcribers.items():
                # The client_id from LLM Assistant corresponds to the original client
                # We might need to clean up based on the client_id pattern
                if session_id == client_id or f"client-{session_id[:8]}" == client_id:
                    transcribers_to_remove.append(session_id)
            
            # Clean up identified transcribers
            for session_id in transcribers_to_remove:
                if session_id in self.client_transcribers:
                    # ✅ ESSENTIAL: Cleanup confirmation
                    if self.enable_logging:
                        print(f"🧹 Session CLEANED: {session_id}")
                    del self.client_transcribers[session_id]
            
            # if self.enable_logging:
            #     print(f"🧹 Cleanup completed for client: {client_id}")                

        @self.sio.event
        async def cleanup_client(sid, data):
            """Handle cleanup request from LLM Assistant for a specific client."""
            client_id = data.get('clientId')
            # ✅ ESSENTIAL: Session cleanup status
            if self.enable_logging:
                print(f"🧹 Session CLEANUP requested: {client_id}")
            
            # Track which sessions were cleaned up
            cleaned_sessions = []
            
            # Remove transcribers that match this client_id
            # Since the LLM Assistant sends the exact client_id, we can match directly
            sessions_to_remove = []
            for session_id in list(self.client_transcribers.keys()):
                if session_id == client_id:
                    sessions_to_remove.append(session_id)
            
            # Clean up identified sessions
            for session_id in sessions_to_remove:
                if session_id in self.client_transcribers:
                    # ✅ ESSENTIAL: Cleanup confirmation
                    if self.enable_logging:
                        print(f"🧹 Session CLEANED: {session_id}")
                    del self.client_transcribers[session_id]
                    cleaned_sessions.append(session_id)
            
            # Confirm cleanup back to LLM Assistant
            await self.sio.emit('cleanup_confirmed', {
                'clientId': client_id,
                'cleanedSessions': cleaned_sessions,
                'timestamp': time.time()
            }, room=sid)
            
            # if self.enable_logging:
            #     print(f"✅ Cleanup completed for client: {client_id}, removed {len(cleaned_sessions)} sessions")


        async def emit_segment(sid, client_id, transcriber, audio_segment, span):
            """Transcribe one segment and send it to the sender and the clientId's room. Opt-in senders
            (transcriber.clip) get the clip decoder and a transcription even when it was rejected
            (empty text, `rejected` set), so they know the span was handled."""
            transcriber.inflight += 1
            try:
                await _emit_segment(sid, client_id, transcriber, audio_segment, span)
            finally:
                transcriber.inflight -= 1

        async def _emit_segment(sid, client_id, transcriber, audio_segment, span):
            duration = len(audio_segment) / self.sample_rate
            print(f"[DIAG] segment READY client={client_id[:8]} dur={duration:.2f}s — entering transcribe", flush=True)
            _t0 = time.time()
            rejected, result = None, {}
            if transcriber.clip:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.thread_pool, self._transcribe_live_clip_sync,
                                                    audio_segment.astype(np.float32), transcriber.language)
                text, rejected = (result.get("text") or "").strip(), result.get("rejected")
            else:
                text = await self._transcribe_async(audio_segment, client_id)
            print(f"[DIAG] transcribe DONE client={client_id[:8]} wall={time.time() - _t0:.2f}s text={text!r}", flush=True)
            if not transcriber.clip and not (text and len(text.strip()) > 1):
                return
            if text and self.enable_logging:
                print(f"🗣️ [{duration:.1f}s] {client_id}: {text}")
            payload = {"text": text, "duration": duration, "client_id": client_id, "ts": time.time(),
                       # the segment's place in the sender's stream, in seconds from its first audio
                       "start": round(span[0], 3), "end": round(span[1], 3)}
            if transcriber.clip:
                payload["rejected"] = rejected
                # each word [text, start, end] in stream seconds (Cortex gives each word its speaker)
                payload["words"] = [[w, round(span[0] + a, 3), round(span[0] + b, 3)]
                                    for seg in (result.get("segments") or []) for w, a, b in seg["words"]]
            # to everyone subscribed to this clientId (e.g., agent_server), the sender among them
            await self.sio.emit("transcription", payload, room=client_id)
            # LEGACY: also to the sender's own socket (so a legacy sender gets each one twice; kept
            # for the apps written against it). Opt-in senders get it once.
            if not transcriber.clip:
                await self.sio.emit("transcription", payload, room=sid)
            if text and self.on_transcription:
                try:
                    self.on_transcription(text, client_id, duration)
                except Exception as e:
                    self.logger.error(f"Error in transcription callback: {e}")

        @self.sio.event
        async def audio_data(sid, data):
            """Handle incoming audio data: {audioData: PCM16 16 kHz mono bytes, clientId, language?}
            (or raw bytes, legacy). `language` present (a Whisper code such as "pt", or null to
            detect) opts the clientId into the clip decoder (_transcribe_live_clip_sync)."""
            try:
                # Handle new format with client ID
                if isinstance(data, dict) and 'audioData' in data and 'clientId' in data:
                    audio_data = data['audioData']
                    client_id = data['clientId']
                else:
                    # Legacy format (raw audio data)
                    audio_data = data
                    client_id = f"client-{sid[:8]}"

                # Ensure the sender is in the client_id room (idempotent)
                await self.sio.enter_room(sid, client_id)
                self.sid_clients.setdefault(sid, set()).add(client_id)

                # CREATE OR GET TRANSCRIBER USING THE CLIENT_ID (not sid)
                if client_id not in self.client_transcribers:
                    if self.enable_logging:
                        print(f"🆕 Session CREATED: {client_id}")
                    self.client_transcribers[client_id] = self._ClientTranscriber(self)
                transcriber = self.client_transcribers[client_id]
                if isinstance(data, dict) and "language" in data:
                    transcriber.clip, transcriber.language = True, data.get("language") or None
                    transcriber.min_speech = LIVE_MIN_SPEECH_SEC
                    if data.get("pause") is not None:
                        transcriber.pause = float(np.clip(float(data["pause"]), *LIVE_PAUSE_RANGE))

                # Convert binary PCM16 -> float32 (-1..1)
                if isinstance(audio_data, (bytes, bytearray)):
                    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    return  # unsupported format (keep silent to avoid log spam)

                if transcriber.clip:
                    samples = transcriber.raise_level(samples)
                transcriber.add_audio(samples.tolist())

                # [DIAG] Audio flow: log every ~50 packets (~5s at 100ms packets) so
                # we can see audio is reaching us. Tracks buffer growth too.
                if not hasattr(transcriber, "_pkt_count"):
                    transcriber._pkt_count = 0
                transcriber._pkt_count += 1
                if transcriber._pkt_count % 50 == 0:
                    print(
                        f"[DIAG] audio_data pkt#{transcriber._pkt_count} "
                        f"client={client_id[:8]} buf_len={len(transcriber.audio_buffer)} "
                        f"buf_s={len(transcriber.audio_buffer)/self.sample_rate:.1f}s "
                        f"last_processed={transcriber.last_processed_time:.2f}s",
                        flush=True,
                    )

                # Check for ready segments
                audio_segment = transcriber.get_ready_segment()
                if audio_segment is not None:
                    await emit_segment(sid, client_id, transcriber, audio_segment, transcriber.last_span)

            except Exception as e:
                self.logger.error(f"Error processing audio from {sid}: {e}")

        @self.sio.event
        async def audio_end(sid, data):
            """The sender's audio for a clientId has ended ({clientId}, with an ack): the speech still
            in its buffer is transcribed and sent now, without waiting for a pause after it. Answers
            {"segments": n} once they are all sent."""
            client_id = (data or {}).get("clientId") or f"client-{sid[:8]}"
            transcriber = self.client_transcribers.get(client_id)
            if transcriber is None:
                return {"segments": 0}
            # segments already being transcribed are sent before this answers (the sender may disconnect then)
            waited = 0.0
            while transcriber.inflight and waited < 120:
                await asyncio.sleep(0.05)
                waited += 0.05
            held = len(transcriber.audio_buffer) / self.sample_rate
            last = transcriber.last_processed_time
            pending = transcriber.remaining_segments()
            print(f"[DIAG] audio_end client={client_id[:8]} buffer={held:.1f}s last_processed={last:.2f}s "
                  f"waited={waited:.2f}s segments={[(round(a, 1), round(b, 1)) for _, (a, b) in pending]}", flush=True)
            for audio_segment, span in pending:
                await emit_segment(sid, client_id, transcriber, audio_segment, span)
            return {"segments": len(pending)}

        @self.sio.event
        async def subscribe_transcripts(sid, data):
            """Allow a socket (e.g., agent_server) to subscribe to transcripts for a clientId."""
            client_id = (data or {}).get("clientId")
            # token = (data or {}).get("token")  # optional: verify if you add auth

            if not client_id:
                return await self.sio.emit("error", {"msg": "missing clientId"}, room=sid)

            await self.sio.enter_room(sid, client_id)
            if self.enable_logging:
                print(f"👂 Subscribed {sid[:8]} to transcripts room: {client_id}")

            await self.sio.emit("subscribed", {"clientId": client_id}, room=sid)

        @self.sio.event
        async def transcribe(sid, data):
            """Transcribe one finished clip (called with an ack, e.g. `await client.call(...)`).

            data: {"audio": PCM16 16 kHz mono bytes, "language": Whisper code such as "pt"
            or None (detect), "prompt": optional text Whisper takes as preceding context, e.g.
            a vocabulary of names}. Answers {"text": str, "rejected": reason or None}.
            Used by Cortex to re-transcribe each diarized speaker turn after its fast
            streaming draft; the live `audio_data` path above is unaffected."""
            data = data or {}
            audio = data.get("audio")
            if not isinstance(audio, (bytes, bytearray)) or len(audio) < 2:
                return {"text": "", "rejected": "no audio"}
            pcm = np.frombuffer(bytes(audio[: len(audio) - len(audio) % 2]), dtype=np.int16).astype(np.float32) / 32768.0
            language = data.get("language") or None
            prompt = (data.get("prompt") or "").strip() or None
            words = bool(data.get("words"))       # each piece's words with their times (segments)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_pool, self._transcribe_clip_sync, pcm, language, prompt, words)


    async def initialize(self):
        """Load models and initialize components."""
        print("🔄 Loading Whisper model...")
        
        # Load Whisper
        self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.whisper_model = torch.compile(self.whisper_model, mode="reduce-overhead", fullgraph=True)
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Load Silero VAD
        self.vad_model = silero_vad.load_silero_vad()
        
        # Create thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        print("✅ Models loaded successfully")
    
    def _log_thread_usage(self, action: str, client_info: str = ""):
        """Log thread pool usage changes."""
        if action == "start":
            self.active_transcriptions += 1
            # if self.enable_logging:
            #     print(f"🧵 Thread ACQUIRED {client_info} | Active: {self.active_transcriptions}/{self.max_workers}")
        elif action == "end":
            self.active_transcriptions -= 1
            # if self.enable_logging:
            #     print(f"🧵 Thread RELEASED {client_info} | Active: {self.active_transcriptions}/{self.max_workers}")
    

    def _transcribe_sync(self, audio_segment: np.ndarray, client_id: str) -> str:
        """Synchronous transcription function.

        Three-layer defense against the well-documented whisper-large-v3-turbo
        hallucination class ("Thank you." / "Mm-hmm." on silence and on
        bass-heavy mouth noise). See module-level constants and project
        memory for the empirical justification.

          1. Pre-Whisper RMS gate (MIN_AUDIO_RMS_DBFS): drop any segment
             whose energy is below the no-speech-possible floor.
          2. Whisper generate() with auto-language-detect (language=None)
             so Portuguese/Spanish/French/Italian/Japanese/etc. speech is
             transcribed in the source language. We previously pinned
             language='en' to suppress non-English hallucinations on
             silence (e.g., the Korean "MBC 뉴스 김성현입니다."), but the
             RMS gate (1) and first-token-probability gate (3b) are
             language-agnostic and catch those degenerate inputs at the
             source. Multi-language entries in HALLUCINATION_BLOCKLIST
             cover the canonical sign-off hallucinations across the
             languages we support.
          3. Post-Whisper short-clip filter: if duration < 1.5 s AND
             (transcript matches the canonical hallucination blocklist
             OR first-content-token probability < 0.6), return empty.
        """
        self._log_thread_usage("start", f"({client_id})")
        try:
            duration_s = len(audio_segment) / self.sample_rate

            # ---- Layer 1: pre-Whisper RMS gate ------------------------
            rms = float(np.sqrt(np.mean(audio_segment.astype(np.float32) ** 2)))
            rms_dbfs = 20.0 * math.log10(rms + 1e-12)
            if rms_dbfs < MIN_AUDIO_RMS_DBFS:
                print(
                    f"🔇 RMS_GATE_REJECT rms={rms_dbfs:.1f}dBFS "
                    f"dur={duration_s:.2f}s",
                    flush=True,
                )
                return ""

            inputs = self.processor(audio_segment, sampling_rate=self.sample_rate, return_tensors="pt") # type: ignore
            input_features = inputs["input_features"].to(self.device, dtype=self.torch_dtype)

            with torch.inference_mode():
                # Note: dropped return_timestamps=True and the
                # *_threshold filters. Those filters only function in
                # HF Whisper's long-form generation path (audio > 30s),
                # AND empirically don't catch v3-turbo's "Thank you on
                # silence" failures even when active. Keeping them with
                # return_dict_in_generate also auto-enables
                # return_segments which changes .sequences into a list
                # and breaks batch_decode. Our 3-layer external defense
                # is what actually catches the failures.
                gen_out = self.whisper_model.generate( # type: ignore
                    input_features,
                    max_new_tokens=200,
                    num_beams=1,
                    do_sample=False,
                    # language=None → Whisper auto-detects the spoken
                    # language (was pinned to 'en' previously to suppress
                    # non-English hallucinations on silence; that role is
                    # now covered by the RMS gate + first-token-P gate
                    # which are both language-agnostic). Multi-language
                    # entries in HALLUCINATION_BLOCKLIST cover the rest.
                    language=None,
                    task="transcribe",
                    condition_on_prev_tokens=False,
                    temperature=0.0,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            output_ids = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip() # type: ignore

            if not text:
                return ""

            # ---- Layer 3: post-Whisper short-clip filter --------------
            # Only applied to clips below SHORT_AUDIO_DURATION_S; longer
            # clips are trusted (a real long utterance with the canonical
            # phrase as part of it shouldn't be dropped).
            if duration_s < SHORT_AUDIO_DURATION_S:
                # 3a. Blocklist match (cheap; check first)
                normalised = _normalize_for_blocklist(text)
                if normalised in HALLUCINATION_BLOCKLIST:
                    print(
                        f"🚫 BLOCKLIST_REJECT text={text!r} "
                        f"dur={duration_s:.2f}s",
                        flush=True,
                    )
                    return ""

                # 3b. First-content-token probability gate. With
                # return_timestamps=True, the generated sequence begins
                # with a timestamp token (e.g. <|0.00|>); the first
                # non-special token after it is the first SPOKEN token.
                # Its probability accurately reflects decoder confidence
                # in what the audio actually says (avg logprob does not,
                # because of autoregressive lock-in).
                scores = getattr(gen_out, "scores", None)
                if scores:
                    seq = output_ids[0]
                    n_generated = len(scores)
                    gen_start = len(seq) - n_generated
                    special_ids = set(self.processor.tokenizer.all_special_ids) # type: ignore
                    first_token_p = None
                    for i in range(n_generated):
                        tok_id = int(seq[gen_start + i].item())
                        if tok_id in special_ids:
                            continue
                        # Some timestamp tokens may not be in
                        # all_special_ids depending on tokenizer; skip
                        # any token whose decoded form is wrapped in
                        # <|...|>.
                        decoded_one = self.processor.tokenizer.decode([tok_id]) # type: ignore
                        if decoded_one.startswith("<|") and decoded_one.endswith("|>"):
                            continue
                        # Found first content token at step i.
                        logp = torch.log_softmax(scores[i][0].float(), dim=-1)[tok_id].item()
                        first_token_p = math.exp(logp)
                        break
                    if first_token_p is not None and first_token_p < FIRST_TOKEN_PROB_THRESHOLD:
                        print(
                            f"❓ FIRST_TOKEN_REJECT P={first_token_p:.3f} "
                            f"text={text!r} dur={duration_s:.2f}s",
                            flush=True,
                        )
                        return ""

            return text
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""
        finally:
            self._log_thread_usage("end", f"({client_id})")
    
    def _clip_pieces(self, pcm: np.ndarray) -> list[tuple]:
        """A clip in pieces of at most CLIP_PIECE_SECONDS (Whisper's window is 30 s), as (first sample,
        samples), cut at the pauses Silero VAD finds; a longer unbroken run is cut at the limit."""
        limit = int(CLIP_PIECE_SECONDS * self.sample_rate)
        if len(pcm) <= limit:
            return [(0, pcm)]
        if self.clip_vad_model is None:     # its own instance: the live path uses vad_model on the event loop
            self.clip_vad_model = silero_vad.load_silero_vad()
        spans = silero_vad.get_speech_timestamps(torch.from_numpy(pcm), self.clip_vad_model,
                                                 sampling_rate=self.sample_rate, threshold=self.vad_threshold)
        if not spans:
            return []
        pieces, start, end = [], spans[0]["start"], spans[0]["start"]
        for s in spans:
            if s["end"] - start > limit and end > start:        # this span would overflow: close the piece
                pieces.append((start, end))
                start = s["start"]
            while s["end"] - start > limit:                     # one span longer than the limit
                pieces.append((start, start + limit))
                start += limit
            end = s["end"]
        pieces.append((start, end))
        pad = int(0.2 * self.sample_rate)
        return [(max(0, a - pad), pcm[max(0, a - pad):min(len(pcm), b + pad)]) for a, b in pieces if b > a]

    def _decode_piece(self, features, language: Optional[str], temperature: float, prompt: Optional[str] = None,
                      attention_mask=None):
        """The piece's text; with attention_mask given, (text, words): each word [text, start, end] in
        seconds from the piece's start, from Whisper's cross-attention alignment heads (the uncompiled
        model: the timings need the attention weights, which the compiled graph does not return)."""
        kw = dict(max_new_tokens=440, num_beams=1, task="transcribe", language=language,
                  condition_on_prev_tokens=False)
        if prompt:
            ids = self.processor.get_prompt_ids(prompt, return_tensors="pt").to(self.device)  # type: ignore
            if len(ids) > CLIP_MAX_PROMPT_TOKENS:
                raise ValueError(f"prompt is {len(ids)} tokens; at most {CLIP_MAX_PROMPT_TOKENS}")
            # the decoder holds 448 positions: start tokens + prompt + the transcript
            kw.update(prompt_ids=ids, max_new_tokens=444 - len(ids))
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        if attention_mask is None:
            with torch.inference_mode():
                out = self.whisper_model.generate(features, **kw)  # type: ignore
            return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()  # type: ignore
        model = getattr(self.whisper_model, "_orig_mod", self.whisper_model)
        with torch.inference_mode():
            out = model.generate(features, attention_mask=attention_mask, return_token_timestamps=True, **kw)  # type: ignore
        seq, stamps = out["sequences"][0], out["token_timestamps"][0]
        text = self.processor.batch_decode(seq.unsqueeze(0), skip_special_tokens=True)[0].strip()  # type: ignore
        return text, self._words(seq, stamps)

    def _words(self, seq, stamps) -> list:
        """Tokens grouped into words (a token starting with a space starts a word): [text, start, end],
        a token's time being where it ends; a word starts where the token before it ended."""
        tok = self.processor.tokenizer  # type: ignore
        special = set(tok.all_special_ids)
        words, cur, start, end = [], "", None, None
        for i, tid in enumerate(seq.tolist()):
            if tid in special or tid >= tok.eos_token_id:
                continue
            piece = tok.decode([tid])
            if piece.startswith(" ") and cur.strip():
                words.append([cur.strip(), round(start, 3), round(end, 3)])
                cur, start = "", None
            if start is None:
                start = float(stamps[i - 1]) if i > 0 else float(stamps[i])
            cur += piece
            end = float(stamps[i])
        if cur.strip():
            words.append([cur.strip(), round(start, 3), round(end, 3)])
        return words

    def _transcribe_clip_sync(self, pcm: np.ndarray, language: Optional[str], prompt: Optional[str] = None,
                              words: bool = False) -> dict:
        """Whisper on one clip with the language given. Each piece is decoded greedily and,
        as in Whisper's own decoding, again at rising temperatures while the text repeats
        itself (zlib compression ratio above CLIP_MAX_COMPRESSION) or holds more words than
        the audio could (CLIP_MAX_WORDS_PER_SECOND); a piece that never passes is rejected."""
        self._log_thread_usage("start", "(clip)")
        try:
            rms = float(np.sqrt(np.mean(pcm ** 2)))
            if 20.0 * math.log10(rms + 1e-12) < MIN_AUDIO_RMS_DBFS:
                return {"text": "", "rejected": "silence"}
            texts, segments = [], []
            for offset, piece in self._clip_pieces(pcm):
                seconds = len(piece) / self.sample_rate
                inputs = self.processor(piece, sampling_rate=self.sample_rate, return_tensors="pt",  # type: ignore
                                        return_attention_mask=words)
                features = inputs["input_features"].to(self.device, dtype=self.torch_dtype)
                mask = inputs["attention_mask"].to(self.device) if words else None
                for temperature in CLIP_TEMPERATURES:
                    text = self._decode_piece(features, language, temperature, prompt, attention_mask=mask)
                    if words:
                        text, piece_words = text
                    raw = text.encode("utf-8")
                    ratio = len(raw) / max(1, len(zlib.compress(raw)))
                    if ratio <= CLIP_MAX_COMPRESSION and len(text.split()) <= CLIP_MAX_WORDS_PER_SECOND * seconds + 2:
                        break
                else:
                    print(f"🔁 CLIP_REJECT repetition dur={seconds:.1f}s text={text[:80]!r}", flush=True)
                    if words:       # with words (a long recording in parts): only this piece is lost
                        at = offset / self.sample_rate
                        segments.append({"start": round(at, 3), "end": round(at + seconds, 3), "text": "",
                                         "words": [], "rejected": "repetition"})
                        continue
                    return {"text": "", "rejected": "repetition"}
                texts.append(text)
                if words:
                    at = offset / self.sample_rate
                    segments.append({"start": round(at, 3), "end": round(at + seconds, 3), "text": text,
                                     "words": [[w, round(a + at, 3), round(b + at, 3)] for w, a, b in piece_words]})
            out = {"text": " ".join(t for t in texts if t), "rejected": None}
            if words:
                out["segments"] = segments
            return out
        except Exception as e:
            self.logger.error(f"Clip transcription error: {e}")
            return {"text": "", "rejected": f"error: {e}"}
        finally:
            self._log_thread_usage("end", "(clip)")

    def _transcribe_live_clip_sync(self, pcm: np.ndarray, language: Optional[str]) -> dict:
        """A live segment of an opt-in sender: raised to a speaking level, then the clip decoder."""
        duration_s = len(pcm) / self.sample_rate
        rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(pcm ** 2))) + 1e-12)
        if rms_db < LIVE_MIN_RMS_DBFS:
            print(f"🔇 LIVE_CLIP_SILENCE rms={rms_db:.1f}dBFS dur={duration_s:.2f}s", flush=True)
            return {"text": "", "rejected": "silence"}
        peak_db = 20.0 * math.log10(float(np.abs(pcm).max()) + 1e-12)
        gain = max(0.0, min(LIVE_TARGET_DBFS - rms_db, LIVE_PEAK_DBFS - peak_db, LIVE_MAX_GAIN_DB))
        if gain >= 1.0:
            pcm = (pcm * 10 ** (gain / 20)).astype(np.float32)
        result = self._transcribe_clip_sync(pcm, language, words=True)
        text = result.get("text") or ""
        if text and duration_s < SHORT_AUDIO_DURATION_S and _normalize_for_blocklist(text) in HALLUCINATION_BLOCKLIST:
            print(f"🚫 BLOCKLIST_REJECT text={text!r} dur={duration_s:.2f}s", flush=True)
            return {"text": "", "rejected": "blocklist"}
        return result

    async def _transcribe_async(self, audio_segment: np.ndarray, client_id: str) -> str:
        """Async wrapper for transcription."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, self._transcribe_sync, audio_segment, client_id)
    
    class _ClientTranscriber:
        """Per-client transcription state."""
        
        def __init__(self, server_instance):
            self.server = server_instance
            self.audio_buffer = []
            self.last_processed_time = 0
            self.last_check_time = 0
            # samples dropped from the front of the buffer so far: buffer position + dropped =
            # position in the client's stream (the `start`/`end` sent with each transcription)
            self.dropped = 0
            self.last_span = (0.0, 0.0)          # stream seconds of the segment last returned
            # set when the sender passes `language` in audio_data (null = detect): its segments
            # go through the clip decoder (see _transcribe_live_clip_sync); unset = as before
            self.clip = False
            self.language: Optional[str] = None
            self.pause = server_instance.silence_duration      # the silence that ends a segment
            self.min_speech = server_instance.min_speech_duration  # the shortest speech that is a segment
            self.inflight = 0                    # segments being transcribed (audio_end waits for them)
            self.levels: deque = deque(maxlen=LIVE_LEVEL_WINDOW)  # opt-in: recent packet RMS levels (dBFS)
            self.peaks: deque = deque(maxlen=LIVE_PEAK_WINDOW)    # the last packets' peaks (dBFS)

        def raise_level(self, samples: np.ndarray) -> np.ndarray:
            """Opt-in senders: quiet speech raised before the VAD sees it (LIVE_LEVEL_WINDOW)."""
            if not len(samples):
                return samples
            self.levels.append(10.0 * math.log10(float(np.mean(samples.astype(np.float64) ** 2)) + 1e-12))
            self.peaks.append(20.0 * math.log10(float(np.abs(samples).max()) + 1e-12))
            gain_db = float(np.clip(min(LIVE_TARGET_DBFS - np.percentile(self.levels, 90),
                                        LIVE_PEAK_DBFS - max(self.peaks)), 0.0, LIVE_MAX_GAIN_DB))
            if gain_db < 1.0:
                return samples
            return (np.tanh(samples * 10 ** (gain_db / 20)) * 0.9).astype(np.float32)

        def _span(self, start_sample: int, end_sample: int) -> tuple:
            sr = self.server.sample_rate
            return ((self.dropped + start_sample) / sr, (self.dropped + end_sample) / sr)

        def add_audio(self, samples):
            """Add audio samples to the per-client rolling buffer.

            Two-phase trim:
              1. Drop audio that's already been emitted as a transcribed
                 segment. last_processed_time is the buffer-coordinate
                 END time of the most recently processed VAD segment;
                 everything before it is safe to discard. After the drop
                 we rebase last_processed_time to 0 so the buffer is
                 always in current coordinates.
              2. Hard safety cap at MAX_BUFFER_SECONDS to bound memory
                 and VAD compute on pathologically continuous speech
                 (no silence_duration-long pause within this window).
                 Reaching this cap re-introduces the original "drop the
                 front of the unprocessed audio" behaviour, but only at
                 the extreme tail (>60s of continuous speech) instead of
                 the previous 10s.

            Replaces the old "always keep only last 10s" rebase that
            silently truncated any utterance longer than ~10 seconds.
            """
            self.audio_buffer.extend(samples)

            # Phase 1: drop already-processed audio.
            if self.last_processed_time > 0:
                n_processed = int(self.last_processed_time * self.server.sample_rate)
                if 0 < n_processed <= len(self.audio_buffer):
                    self.audio_buffer = self.audio_buffer[n_processed:]
                    self.dropped += n_processed
                    self.last_processed_time = 0.0

            # Phase 2: safety cap.
            max_samples = self.server.sample_rate * MAX_BUFFER_SECONDS
            if len(self.audio_buffer) > max_samples:
                excess = len(self.audio_buffer) - max_samples
                self.audio_buffer = self.audio_buffer[excess:]
                self.dropped += excess
                # last_processed_time is already 0 from Phase 1; nothing
                # else to adjust.
        
        def done_before(self, start_time: float) -> bool:
            """Whether a VAD segment starting at start_time (buffer seconds) was already emitted. After a
            segment the buffer is cut at its end and restarts at 0, so speech that runs straight on starts
            at 0.0: counted as done by `<=`, it was never emitted, the buffer filled to MAX_BUFFER_SECONDS
            and dropped it from the front (a phone recording: 16 s lost at audio_end; measured 2026-09-30).
            Opt-in senders get `<`; the others keep the previous behaviour."""
            if self.clip:
                return start_time < self.last_processed_time
            return start_time <= self.last_processed_time

        def should_check_for_segments(self) -> bool:
            """Rate limit segment checking."""
            current_time = time.time()
            if current_time - self.last_check_time >= self.server.processing_interval:
                self.last_check_time = current_time
                return True
            return False
        
        def get_ready_segment(self) -> Optional[np.ndarray]:
            """Get speech segment if ready for processing."""
            if not self.should_check_for_segments():
                return None
                
            if len(self.audio_buffer) < self.server.sample_rate * 1.0:
                return None
            
            # Run VAD
            audio_tensor = torch.FloatTensor(self.audio_buffer)
            
            try:
                with torch.no_grad():
                    segments = silero_vad.get_speech_timestamps(
                        audio_tensor,
                        self.server.vad_model,
                        sampling_rate=self.server.sample_rate,
                        threshold=self.server.vad_threshold,
                        min_speech_duration_ms=int(self.min_speech * 1000),
                        min_silence_duration_ms=int(self.pause * 1000)
                    )
            except Exception:
                return None
            
            if not segments:
                return None
            
            # Find first ready segment
            buffer_duration = len(self.audio_buffer) / self.server.sample_rate
            
            for segment in segments:
                start_sample = segment['start']
                end_sample = segment['end']
                
                start_time = start_sample / self.server.sample_rate
                end_time = end_sample / self.server.sample_rate
                
                # Skip if already processed
                if self.done_before(start_time):
                    continue
                
                # Check for enough silence
                silence_after = buffer_duration - end_time
                
                if silence_after >= self.pause:
                    duration = (end_sample - start_sample) / self.server.sample_rate
                    if duration >= self.min_speech:
                        audio_data = np.array(self.audio_buffer[start_sample:end_sample])
                        self.last_processed_time = end_time
                        self.last_span = self._span(start_sample, end_sample)
                        return audio_data

            # Force-flush safety net: the buffer is at/near the hard cap and no
            # segment was silence-terminated (continuous speech with no
            # >= silence_duration gap). Emit the first unprocessed speech
            # segment anyway so the buffer drains instead of pinning at the
            # cap — which otherwise pegs VAD on a full 30s buffer every
            # interval, stalls all transcription for this client, and starves
            # the event loop. Coarser (segment isn't silence-terminated) but
            # far better than never transcribing.
            if buffer_duration >= MAX_BUFFER_SECONDS - 1.0:
                for segment in segments:
                    start_sample = segment['start']
                    end_sample = segment['end']
                    start_time = start_sample / self.server.sample_rate
                    end_time = end_sample / self.server.sample_rate
                    if self.done_before(start_time):
                        continue
                    duration = (end_sample - start_sample) / self.server.sample_rate
                    if duration >= self.min_speech:
                        if self.clip:
                            end_sample = self.quiet_cut(start_sample, end_sample)
                            end_time = end_sample / self.server.sample_rate
                        audio_data = np.array(self.audio_buffer[start_sample:end_sample])
                        self.last_processed_time = end_time
                        self.last_span = self._span(start_sample, end_sample)
                        print(
                            f"[DIAG] FORCE-FLUSH at {buffer_duration:.1f}s cap: "
                            f"emitting {duration:.1f}s segment (no trailing silence)",
                            flush=True,
                        )
                        return audio_data

            return None

        def quiet_cut(self, start_sample: int, end_sample: int) -> int:
            """Opt-in senders: where to end a segment forced out by the buffer cap (nobody paused): the
            middle of the quietest LIVE_CUT_QUIET_SEC within the last LIVE_CUT_SEARCH_SEC, not wherever the
            cap fell. Cut mid-word, Whisper lost the words around the cut and often the start of the next
            segment (a second reader's 5 s, measured 2026-09-30)."""
            sr = self.server.sample_rate
            win, lo = int(LIVE_CUT_QUIET_SEC * sr), max(start_sample + sr, end_sample - int(LIVE_CUT_SEARCH_SEC * sr))
            if end_sample - lo <= win:
                return end_sample
            x = np.asarray(self.audio_buffer[lo:end_sample], dtype=np.float64) ** 2
            csum = np.concatenate([[0.0], np.cumsum(x)])
            energy = csum[win:] - csum[:-win]
            return lo + int(np.argmin(energy)) + win // 2

        def remaining_segments(self) -> list:
            """Every unprocessed speech segment left in the buffer, with its stream span, whatever
            the silence after it (the sender has ended its audio: `audio_end`). Empties the buffer."""
            buf = self.audio_buffer
            self.audio_buffer, out = [], []
            if len(buf) < self.server.sample_rate * self.min_speech:
                self.dropped += len(buf)
                return out
            with torch.no_grad():
                segments = silero_vad.get_speech_timestamps(
                    torch.FloatTensor(buf), self.server.vad_model, sampling_rate=self.server.sample_rate,
                    threshold=self.server.vad_threshold,
                    min_speech_duration_ms=int(self.min_speech * 1000),
                    min_silence_duration_ms=int(self.pause * 1000))
            for seg in segments:
                if self.done_before(seg["start"] / self.server.sample_rate):
                    continue
                out.append((np.array(buf[seg["start"]:seg["end"]]), self._span(seg["start"], seg["end"])))
            self.dropped += len(buf)
            self.last_processed_time = 0.0
            return out
    
    async def start(self):
        """Start the server."""
        if not self.whisper_model:
            await self.initialize()
        
        print(f"🟢 Server listening on {self.host}:{self.port}")
        
        # Create and configure uvicorn server
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning"  # Set to warning to reduce uvicorn noise
        )
        self.server = uvicorn.Server(config)
        
        if self.enable_logging:
            print("🚀 Ready for real-time transcription")
        
        # Start the server
        await self.server.serve()
    
    async def stop(self):
        """Stop the server and cleanup resources."""
        if self.server:
            self.server.should_exit = True
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        print("🛑 Server stopped")
    
    async def run_forever(self):
        """Start server and run indefinitely."""
        try:
            await self.start()
        except KeyboardInterrupt:
            print("🛑 Shutting down...")
        finally:
            await self.stop()

# Example usage and callback function
def on_speech_transcribed(text: str, client_id: str, duration: float):
    """
    Callback function called when speech is transcribed.
    This is where you'd integrate with your LLM chatbot.
    """
    print(f"📞 CALLBACK: Client {client_id} said: '{text}' ({duration:.1f}s)")
    
    # 1. Send the text to the LLM
    # 2. Get the LLM response
    # 3. Convert LLM response to speech (TTS)
    # 4. Send audio back to client or play locally

# Example standalone usage
async def main():
    
    """
    server = STTServer(
        model_path="/stt_server/data/models/whisper-large-v3-turbo",
        port=2700,
        on_transcription=on_speech_transcribed,  # Callback function
        silence_duration=0.8,
        enable_logging=True
    )
    """

    try:
        server = STTServer.from_config(str(Path.cwd()) + "/data/configuration/stt.server.settings.json")
        # Assign the callback function
        server.on_transcription = on_speech_transcribed 
        await server.run_forever()

    except:
        print("Error loading configuration from " + str(Path.cwd()) + "/data/configuration/stt.server.settings.json")
    
    
if __name__ == "__main__":
    asyncio.run(main())
