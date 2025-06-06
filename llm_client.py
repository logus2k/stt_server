from whisper_server import RealtimeWhisperServer

"""
# Your LLM chatbot integration
async def handle_speech(text: str, client_id: str, duration: float):
    # Called when speech is transcribed - integrate with your LLM here
    print(f"User said: {text}")
    
    # 1. Send to your LLM
    llm_response = await your_llm.chat(text)
    
    # 2. Optional: Convert to speech and send back
    # audio = await your_tts.generate(llm_response)
    # await send_audio_to_client(client_id, audio)

# Start the server
server = RealtimeWhisperServer(
    model_path="models/whisper-large-v3-turbo",
    on_transcription=handle_speech,  # Your callback
    silence_duration=0.8,
    enable_logging=True
)

await server.start()
"""
