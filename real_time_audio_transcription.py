import logging
import queue
import threading
import time
from typing import Type

import streamlit as st
import assemblyai as aai
from assemblyai.streaming.v3 import StreamingEvents
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
    TerminationEvent,
)
from configure import auth_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"Session started: {event.id}")


def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(
        f"Session terminated: {event.audio_duration_seconds} seconds of audio processed"
    )


def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"Error occurred: {error!r}")


import pyaudio

def stream_audio_worker(client: StreamingClient, transcript_queue: queue.Queue, stop_event: threading.Event):
    """
    Configures and runs the AssemblyAI client in a background thread.
    This function should not use any Streamlit commands.
    """
    def on_turn_callback(self: Type[StreamingClient], event: TurnEvent):
        if hasattr(event, 'words'):
            transcript = " ".join(word.text for word in event.words)
            if transcript:
                if event.end_of_turn:
                    transcript_queue.put(("final", transcript))
                else:
                    transcript_queue.put(("partial", transcript))

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn_callback)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    def get_audio_stream():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)
        while not stop_event.is_set():
            data = stream.read(1024)
            yield data
        stream.stop_stream()
        stream.close()
        p.terminate()

    try:
        client.connect(
            StreamingParameters(
                sample_rate=16000,
            )
        )
        client.stream(get_audio_stream())
    except Exception as e:
        print(f"An error occurred during streaming: {e}")
    finally:
        try:
            client.disconnect(terminate=True)
        except Exception:
            pass


def main():
    st.title("Real-Time Audio Transcription with AssemblyAI")

    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
        st.session_state.full_transcript = []
        st.session_state.partial_transcript = ""
        st.session_state.transcript_queue = queue.Queue()
        st.session_state.worker_thread = None
        st.session_state.client = None
        st.session_state.stop_event = None

    col1, col2 = st.columns(2)

    if col1.button("Start Listening"):
        if not st.session_state.is_recording:
            st.session_state.is_recording = True
            st.session_state.full_transcript = []
            st.session_state.partial_transcript = ""
            while not st.session_state.transcript_queue.empty():
                st.session_state.transcript_queue.get()

            client = StreamingClient(StreamingClientOptions(api_key=auth_key))
            st.session_state.client = client
            st.session_state.stop_event = threading.Event()

            thread = threading.Thread(
                target=stream_audio_worker,
                args=(client, st.session_state.transcript_queue, st.session_state.stop_event),
            )
            st.session_state.worker_thread = thread
            thread.start()
            st.rerun()

    if col2.button("Stop Listening"):
        if st.session_state.is_recording:
            st.session_state.is_recording = False
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            if st.session_state.client:
                try:
                    st.session_state.client.disconnect(terminate=True)
                except Exception:
                    pass
            st.session_state.client = None
            st.rerun()

    while not st.session_state.transcript_queue.empty():
        msg_type, text = st.session_state.transcript_queue.get()
        if msg_type == "partial":
            st.session_state.partial_transcript = text
        elif msg_type == "final":
            st.session_state.full_transcript.append(text)
            st.session_state.partial_transcript = ""

    transcript_display = " ".join(st.session_state.full_transcript)
    if st.session_state.partial_transcript:
        transcript_display += " " + st.session_state.partial_transcript

    st.text_area("Transcript", transcript_display.strip(), height=300)

    if st.session_state.is_recording:
        st.write("🔴 Listening...")
        if st.session_state.worker_thread and not st.session_state.worker_thread.is_alive():
            st.session_state.is_recording = False
            st.rerun()
        else:
            time.sleep(0.1)
            st.rerun()


if __name__ == "__main__":
    main()
