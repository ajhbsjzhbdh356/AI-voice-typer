import streamlit as st
import asyncio
import websockets
import json
import threading
from configure import auth_key
import time
import queue
import logging
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use session state to manage the app's state
if 'full_transcript' not in st.session_state:
    st.session_state.full_transcript = []
if 'error' not in st.session_state:
    st.session_state.error = None
if 'queue' not in st.session_state:
    st.session_state.queue = queue.Queue()

URL = f"wss://streaming.assemblyai.com/v3/ws?sample_rate=16000"
LIBRARY_FILE = "library.json"

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.transcription_thread = None
        self.lock = threading.Lock()

    def recv(self, frame):
        self.audio_queue.put(frame.to_ndarray())
        return frame

    def start(self):
        with self.lock:
            self.stop_event.clear()
            self.transcription_thread = threading.Thread(target=self.transcribe)
            self.transcription_thread.start()

    def stop(self):
        with self.lock:
            self.stop_event.set()
            if self.transcription_thread:
                self.transcription_thread.join()

    def transcribe(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._transcribe())

    async def _transcribe(self):
        try:
            async with websockets.connect(
                URL,
                extra_headers={"Authorization": auth_key},
                ping_interval=5,
                ping_timeout=20
            ) as ws:
                sender_task = asyncio.create_task(self.sender(ws))
                receiver_task = asyncio.create_task(self.receiver(ws))
                await asyncio.gather(sender_task, receiver_task)
        except Exception as e:
            st.session_state.queue.put({"error": f"WebSocket connection error: {e}"})

    async def sender(self, ws):
        while not self.stop_event.is_set():
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                await ws.send(audio_chunk.tobytes())
            except queue.Empty:
                continue
            except Exception as e:
                st.session_state.queue.put({"error": f"Send error: {e}"})
                break
            await asyncio.sleep(0.01)

    async def receiver(self, ws):
        while not self.stop_event.is_set():
            try:
                result_str = await ws.recv()
                logger.info(f"Received from AssemblyAI: {result_str}")
                result_json = json.loads(result_str)
                if "words" in result_json:
                    st.session_state.queue.put(result_json)
            except websockets.exceptions.ConnectionClosedOK:
                break
            except Exception as e:
                st.session_state.queue.put({"error": f"Receive error: {e}"})
                break

def save_to_library(transcript):
    if not os.path.exists(LIBRARY_FILE):
        with open(LIBRARY_FILE, "w") as f:
            json.dump([], f)

    with open(LIBRARY_FILE, "r+") as f:
        library = json.load(f)
        library.append({
            "timestamp": datetime.now().isoformat(),
            "transcript": transcript
        })
        f.seek(0)
        json.dump(library, f, indent=4)

def load_library():
    if not os.path.exists(LIBRARY_FILE):
        return []
    with open(LIBRARY_FILE, "r") as f:
        return json.load(f)

def transcription_page():
    st.title("Real-time Speech Recognition")

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": False, "audio": True},
    )

    if webrtc_ctx.state.playing:
        if webrtc_ctx.audio_processor:
            with webrtc_ctx.audio_processor.lock:
                if not hasattr(webrtc_ctx.audio_processor, "transcription_thread") or not webrtc_ctx.audio_processor.transcription_thread.is_alive():
                    webrtc_ctx.audio_processor.start()
    elif webrtc_ctx.audio_processor:
        webrtc_ctx.audio_processor.stop()

    clear_button = st.button("Clear Transcript")
    if clear_button:
        st.session_state.full_transcript = []
        st.rerun()

    st.subheader("Transcription Output:")
    transcript_text = "\n\n".join(st.session_state.full_transcript)
    st.markdown(transcript_text)

    if not webrtc_ctx.state.playing and transcript_text:
        if st.button("Save to Library"):
            save_to_library(transcript_text)
            st.success("Transcription saved to library.")

    if st.session_state.error:
        st.error(st.session_state.error)

    try:
        message = st.session_state.queue.get_nowait()
        if "words" in message:
            turn_order = message.get("turn_order", -1)
            words = message.get("words", [])
            transcript = " ".join([word.get("text", "") for word in words])
            if turn_order >= len(st.session_state.full_transcript):
                st.session_state.full_transcript.append(transcript)
            else:
                st.session_state.full_transcript[turn_order] = transcript
        if "error" in message:
            st.session_state.error = message["error"]
        st.rerun()
    except queue.Empty:
        pass

def library_page():
    st.title("Transcription Library")
    library = load_library()
    if not library:
        st.info("No transcriptions saved yet.")
    else:
        for item in reversed(library):
            st.subheader(f"Transcription from {item['timestamp']}")
            st.markdown(item['transcript'])
            st.markdown("---")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Transcription", "Library"])

if page == "Transcription":
    transcription_page()
elif page == "Library":
    library_page()
