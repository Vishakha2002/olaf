import os
import time
import streamlit.components.v1 as components
import streamlit as st
import requests
from io import BytesIO
import numpy as np
import whisper
from pprint import pprint


def whisper_transcription(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    pprint(result)
    return result["text"]

def transcribe_question(video_event, default="assemblyai"):
    text = ""
    audio_file = "audio.wav"

    ind, val = zip(*video_event['arr'].items())
    ind = np.array(ind, dtype=int)  # convert to np array
    val = np.array(val)             # convert to np array
    sorted_ints = val[ind]
    stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
    # This wav_bytes has the audio

    wav_bytes = stream.read()
    with open(audio_file, "wb") as binary_file:
        binary_file.write(wav_bytes)
    # audio = np.array(list(video_event['arr'].items()))
    

    if default == "assemblyai":
        time.sleep(3)
        text = "Assembly AI Not Implemented"
    elif default == "whisper":
        # text = "Whisper Not Implemented"
        text = whisper_transcription(audio_file)
    else:
        text = "Not Implemented"
    return text    

def parse_video_event(video_event):
    print(st.session_state.is_video_paused)
    if video_event and video_event.get('name') == "onPause":
            st.session_state.is_video_paused = True 
            print("video got paused")
    if video_event and video_event.get('name') == "onProgress":
        # print("Inside onProgres")
        if video_event.get('data') and isinstance(video_event.get('data'), dict):
            frame_stopped_at = video_event["data"].get("playedSeconds")
            # print(f"you paused the video at {frame_stopped_at} session state {st.session_state.is_video_paused}")
            if st.session_state.is_video_paused:
                print(f"you paused the video at {frame_stopped_at}")
                st.session_state.frame_stopped_at = frame_stopped_at
                st.session_state.is_video_paused = False
                st.write(st.session_state)


def initialize_session_state():
    """
    This method sets the variables required for olaf to work in
    streamlit session state.
    """
    if "is_video_paused" not in st.session_state:
        st.session_state['is_video_paused'] = False
    if "frame_stopped_at" not in st.session_state:   
        st.session_state['frame_stopped_at'] = 0


def main():
    """
    Olaf main application script.
    """
    initialize_session_state()

    st.set_page_config(
        layout="wide",
    )

    # Setup for streamlit_player
    _SUPPORTED_EVENTS = [
        "onStart", "onPlay", "onProgress", "onDuration", "onPause",
        "onBuffer", "onBufferEnd", "onSeek", "onEnded", "onError"
    ]
    options = {
        # "events": ["onProgress", "onPause"],
        "events": _SUPPORTED_EVENTS,
        "progress_interval": 1000
    }

    # XXX Take user input instead of harcoding.
    # yt_url = "https://www.youtube.com/watch?v=bmIVWe3Cux8&iv_load_policy=3"
    yt_url = "https://www.youtube.com/watch?v=nK1r_9hPWuI"

    av_player_parent_dir = os.path.dirname(os.path.abspath(__file__))
    av_player_build_dir = os.path.join(av_player_parent_dir, "av_player/frontend/build")
    av_player = components.declare_component("streamlit_player", path=av_player_build_dir)

    # STREAMLIT Video Player Instance
    video_event = av_player(url=yt_url, width ="500px", height = "300px", **options)

    parse_video_event(video_event)
    placeholder = st.empty()
    if isinstance(video_event, dict):  # retrieve audio data
        if "arr" in video_event.keys():
            with st.spinner('Transcribing Question...'):
                # transcription = transcribe_question(video_event)
                transcription = transcribe_question(video_event, default="whisper")
                placeholder.text(transcription)


if __name__ == "__main__":
    main()
