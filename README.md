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
| `audio_data` | Client -\> Server | `bytes` or `{'audioData': bytes, 'clientId': str}` | Stream raw PCM16 (16kHz) audio data. |
| `transcription` | Server -\> Client | `{'text': str, 'duration': float, 'client_id': str, ...}` | A confirmed speech segment has been transcribed. |
| `subscribe_transcripts` | Client -\> Server | `{'clientId': str}` | Allows a monitoring client (e.g., an LLM Assistant) to receive all transcripts for a specific user ID. |
| `cleanup_client` | Client -\> Server | `{'clientId': str}` | Request the server to clear the audio buffer for a specific client session. |

-----

## License

This project is licensed under the **Apache License 2.0**.

---