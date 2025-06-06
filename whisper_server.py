# whisper_server.py

"""
pip install --upgrade pip 
pip install transformers accelerate python-socketio uvicorn silero-vad
"""

import asyncio
import socketio
import uvicorn
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import silero_vad
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Optional, Callable, Dict, Any
import logging


class RealtimeWhisperServer:
    """
    Real-time speech-to-text server using Whisper and Silero VAD with Socket.IO.
    
    Usage:
        server = RealtimeWhisperServer(
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
        min_speech_duration: float = 0.4,
        vad_threshold: float = 0.4,
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
        self.thread_pool = None
        self.server = None
        self.active_transcriptions = 0
        
        # Socket.IO setup
        self.sio = socketio.AsyncServer(
            cors_allowed_origins="*",
            logger=False,  # ✅ Disable Socket.IO internal logging
            engineio_logger=False,  # ✅ Disable EngineIO internal logging
            async_mode='asgi'
        )
        self.app = socketio.ASGIApp(self.sio, other_asgi_app=None)
        
        # Client transcriber storage
        self.client_transcribers = {}
        
        # Setup Socket.IO event handlers
        self._setup_socketio_handlers()
        
        # Setup logging
        if self.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)
    
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
            
            # Clean up transcriber
            if sid in self.client_transcribers:
                del self.client_transcribers[sid]

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

        @self.sio.event
        async def audio_data(sid, data):
            """Handle incoming audio data."""
            try:
                # Handle new format with client ID
                if isinstance(data, dict) and 'audioData' in data and 'clientId' in data:
                    audio_data = data['audioData']
                    client_id = data['clientId']
                else:
                    # Legacy format (raw audio data)
                    audio_data = data
                    client_id = f"client-{sid[:8]}"
                
                # ✅ CREATE OR GET TRANSCRIBER USING THE CLIENT_ID (not sid)
                if client_id not in self.client_transcribers:
                    # ✅ ESSENTIAL: New session creation
                    if self.enable_logging:
                        print(f"🆕 Session CREATED: {client_id}")
                    self.client_transcribers[client_id] = self._ClientTranscriber(self)
                
                # Convert binary data to numpy array
                if isinstance(audio_data, (bytes, bytearray)):
                    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    return
                
                transcriber = self.client_transcribers[client_id]
                transcriber.add_audio(samples.tolist())
                
                # Check for ready segments
                audio_segment = transcriber.get_ready_segment()
                
                if audio_segment is not None:
                    # Process segment
                    duration = len(audio_segment) / self.sample_rate
                    text = await self._transcribe_async(audio_segment, client_id)
                    
                    if text and len(text.strip()) > 1:
                        # ✅ ESSENTIAL: Transcription result
                        if self.enable_logging:
                            print(f"🗣️ [{duration:.1f}s] {client_id}: {text}")
                        
                        # Send transcription with the provided client ID
                        await self.sio.emit('transcription', {
                            'text': text,
                            'duration': duration,
                            'client_id': client_id  # Use the provided client ID
                        }, room=sid)
                        
                        # Call user callback
                        if self.on_transcription:
                            try:
                                self.on_transcription(text, client_id, duration)
                            except Exception as e:
                                self.logger.error(f"Error in transcription callback: {e}")
                
            except Exception as e:
                self.logger.error(f"Error processing audio from {sid}: {e}")
    
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
        """Synchronous transcription function."""
        self._log_thread_usage("start", f"({client_id})")
        try:
            inputs = self.processor(audio_segment, sampling_rate=self.sample_rate, return_tensors="pt")
            input_features = inputs["input_features"].to(self.device, dtype=self.torch_dtype)
            
            with torch.inference_mode():
                output_ids = self.whisper_model.generate(
                    input_features,
                    max_new_tokens=200,
                    num_beams=1,
                    do_sample=False
                )
            
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return text
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""
        finally:
            self._log_thread_usage("end", f"({client_id})")
    
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
        
        def add_audio(self, samples):
            """Add audio samples to buffer."""
            self.audio_buffer.extend(samples)
            
            # Keep only last 10 seconds
            max_samples = self.server.sample_rate * 10
            if len(self.audio_buffer) > max_samples:
                excess = len(self.audio_buffer) - max_samples
                self.audio_buffer = self.audio_buffer[excess:]
                removed_time = excess / self.server.sample_rate
                self.last_processed_time = max(0, self.last_processed_time - removed_time)
        
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
                        min_speech_duration_ms=int(self.server.min_speech_duration * 1000),
                        min_silence_duration_ms=int(self.server.silence_duration * 1000)
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
                if start_time <= self.last_processed_time:
                    continue
                
                # Check for enough silence
                silence_after = buffer_duration - end_time
                
                if silence_after >= self.server.silence_duration:
                    duration = (end_sample - start_sample) / self.server.sample_rate
                    if duration >= self.server.min_speech_duration:
                        audio_data = np.array(self.audio_buffer[start_sample:end_sample])
                        self.last_processed_time = end_time
                        
                        # if self.server.enable_logging:
                        #     print(f"🎤 Found segment: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s)")
                        return audio_data
            
            return None
    
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
            log_level="warning"  # ✅ Set to warning to reduce uvicorn noise
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
    
    # Here you would:
    # 1. Send the text to your LLM
    # 2. Get the LLM response
    # 3. Convert LLM response to speech (TTS)
    # 4. Send audio back to client or play locally

# Example standalone usage
async def main():
    server = RealtimeWhisperServer(
        model_path="/home/logus2k/env/jarbas/data/models/whisper-large-v3-turbo",
        port=2700,
        on_transcription=on_speech_transcribed,  # Your callback function
        silence_duration=0.8,
        enable_logging=True
    )
    
    await server.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
