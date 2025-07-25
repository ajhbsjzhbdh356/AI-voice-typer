import streamlit as st
import asyncio
import websockets
import base64
import json
import threading
from configure import auth_key
import pyaudio
import time
import queue
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use session state to manage the app's state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'full_transcript' not in st.session_state:
    st.session_state.full_transcript = []
if 'thread' not in st.session_state:
    st.session_state.thread = None
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'error' not in st.session_state:
    st.session_state.error = None
if 'device_index' not in st.session_state:
    st.session_state.device_index = None
if 'queue' not in st.session_state:
    st.session_state.queue = queue.Queue()

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
URL = f"wss://streaming.assemblyai.com/v3/ws?sample_rate={RATE}"
LIBRARY_FILE = "library.json"

p = pyaudio.PyAudio()

def audio_stream_start():
    try:
        st.session_state.stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
            input_device_index=st.session_state.device_index
        )
        return True
    except Exception as e:
        st.session_state.queue.put({"error": f"Failed to open audio stream: {e}"})
        return False

async def send_receive(ws, stop_event, q, stream):
    async def send():
        while not stop_event.is_set():
            try:
                data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                await ws.send(data)
            except websockets.exceptions.ConnectionClosedOK:
                break
            except Exception as e:
                q.put({"error": f"Send error: {e}"})
                break
            await asyncio.sleep(0.01)

    async def receive():
        while not stop_event.is_set():
            try:
                result_str = await ws.recv()
                logger.info(f"Received from AssemblyAI: {result_str}")
                result_json = json.loads(result_str)
                if "words" in result_json:
                    q.put(result_json)
            except websockets.exceptions.ConnectionClosedOK:
                break
            except Exception as e:
                q.put({"error": f"Receive error: {e}"})
                break

    await asyncio.gather(send(), receive())

def transcription_thread(stop_event, q, stream):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def connect_ws():
        try:
            async with websockets.connect(
                URL,
                extra_headers={"Authorization": auth_key},
                ping_interval=5,
                ping_timeout=20
            ) as ws:
                await send_receive(ws, stop_event, q, stream)
        except Exception as e:
            q.put({"error": f"WebSocket connection error: {e}"})
        finally:
            q.put({"status": "stopped"})

    loop.run_until_complete(connect_ws())

def start_transcription():
    if not auth_key or auth_key == "YOUR_ASSEMBLYAI_API_KEY":
        st.session_state.error = "AssemblyAI API key is missing. Please add it to configure.py."
        return

    st.session_state.recording = True
    st.session_state.error = None
    st.session_state.full_transcript = []
    st.session_state.stop_event.clear()

    if not audio_stream_start():
        st.session_state.recording = False
        return

    st.session_state.thread = threading.Thread(target=transcription_thread, args=(st.session_state.stop_event, st.session_state.queue, st.session_state.stream), daemon=True)
    st.session_state.thread.start()

def stop_transcription():
    st.session_state.stop_event.set()
    if st.session_state.thread:
        st.session_state.thread.join(timeout=1)
    if st.session_state.stream:
        st.session_state.stream.stop_stream()
        st.session_state.stream.close()
        st.session_state.stream = None
    st.session_state.recording = False
    time.sleep(0.1)

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

    if st.session_state.device_index is None:
        try:
            default_device_info = p.get_default_input_device_info()
            st.session_state.device_index = default_device_info['index']
        except IOError:
            st.warning("No default audio input device found. Please ensure a microphone is connected and configured.")
            return

    col1, col2, col3 = st.columns(3)
    with col1:
        start_button = st.button("Start Transcription")
    with col2:
        stop_button = st.button("Stop Transcription")
    with col3:
        clear_button = st.button("Clear Transcript")

    if start_button and not st.session_state.recording:
        start_transcription()
        if not st.session_state.error:
            st.success("Transcription started.")
        st.rerun()

    if stop_button and st.session_state.recording:
        stop_transcription()
        st.rerun()

    if clear_button:
        st.session_state.full_transcript = []
        st.rerun()

    st.subheader("Transcription Output:")
    transcript_text = "\n\n".join(st.session_state.full_transcript)
    st.markdown(transcript_text)

    if not st.session_state.recording and transcript_text:
        if st.button("Save to Library"):
            save_to_library(transcript_text)
            st.success("Transcription saved to library.")

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.recording:
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
            if "status" in message and message["status"] == "stopped":
                st.session_state.recording = False
            st.rerun()
        except queue.Empty:
            if st.session_state.thread and not st.session_state.thread.is_alive():
                st.session_state.recording = False
                st.rerun()
            else:
                time.sleep(0.1)
                st.rerun()

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
