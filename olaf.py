"""
https://github.com/suyashkumar/ssl-proxy If you need reverse proxy in front of streamlit server
"""
from fileinput import filename
import re
import os
import time
import streamlit.components.v1 as components
import streamlit as st
import subprocess
import requests
from io import BytesIO
import numpy as np
import whisper
from pprint import pprint
from pytube import YouTube
from moviepy.editor import VideoFileClip


def whisper_transcription(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    pprint(result)
    return result["text"]


def transcribe_question(video_event, default="assemblyai"):
    text = ""
    # XXX Vishakha move this to a temp file.
    audio_file = "question_audio.wav"

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
                # st.write(st.session_state)


def initialize_session_state():
    """
    This method sets the variables required for olaf to work in
    streamlit session state.
    """
    if "is_video_paused" not in st.session_state:
        st.session_state['is_video_paused'] = False
    if "frame_stopped_at" not in st.session_state:   
        st.session_state['frame_stopped_at'] = 0


def extract_audio_features():
    pass


def get_audio_wav(audio_filename, video_path) -> str:
    """
    """
    raw_audio_dir = "data/raw_audio"
    saved_audio = os.path.join(os.getcwd(), raw_audio_dir, audio_filename)
    
    if audio_filename not in os.listdir(raw_audio_dir):
        video = VideoFileClip(video_path)
        audio_file = video.audio
        audio_file.write_audiofile(saved_audio, fps=16000)
        print(f"finish video id: {saved_audio}")

    return saved_audio


def extract_frames(video, dst):
    command1 = 'ffmpeg '
    command1 += '-i ' + video + " "
    command1 += '-y' + " "
    command1 += "-r " + "1 "
    command1 += '{0}/%06d.jpg'.format(dst)
    print(command1)
    #    print command1
    os.system(command1)

    return


def preprocess_youtube_video(yt_url):
    """
    To take user input it is import to preprocess the video. Steps:
    1. Download youtube video  (720p resoultion / mp4 format) and save it in data/raw_video
    2. Extract audio (.wav) from the downloaded video and save it in data/raw_audio
    3. Extract frames from video using ffmeg
    
    """
    if yt_url in st.session_state:
        return
    video_object = YouTube(yt_url)
    video_title = re.sub(r'[^A-Za-z0-9 ]+', '', video_object.title)
    video_title = video_title.replace(" ", "_")
    video_frame_path = os.path.join(os.getcwd(), "data/frames/video", video_title)
    
    video_filename = video_title + ".mp4"
    audio_filename = video_title + ".wav"

    if yt_url not in st.session_state:
        extracted_frames = False
        with st.spinner('Video Preprocessing...'):
            # start video download
            saved_video = video_object.streams.filter(mime_type="video/mp4", res="720p").first().download(filename=video_filename, output_path="data/raw_video")
            st.write(f"Downloaded video at {saved_video}")
            # Extract Audio from the saved video
            saved_audio = get_audio_wav(audio_filename=audio_filename, video_path=saved_video)
            st.write(f"Extracted Audio saved at {saved_audio}")

            # Here we will extract frames from the saved video file
            try:
                if not os.path.exists(video_frame_path):
                    print(f"Creating a new folder for extracting video frames {video_frame_path}")
                    os.makedirs(video_frame_path)
                frame_count = len(os.listdir(video_frame_path))
                if frame_count == 0:
                    extract_frames(saved_video, video_frame_path)
                    frame_count = len(os.listdir(video_frame_path))
                st.write(f"Extracted Video frames saved at {video_frame_path}. Count: {frame_count}")
                extracted_frames = True
            except Exception:
                raise
    
        st.session_state[yt_url] = {
            "raw_audio": saved_audio,
            "raw_video": saved_video,
            "video_frame_path": video_frame_path,
            'extracted_frames': extracted_frames
        }
        st.success('Pre processing Done!')
        st.write(st.session_state[yt_url])
        time.sleep(5)


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
    yt_urls = ["https://www.youtube.com/watch?v=6gQ7m0c4ReI","https://youtu.be/is68rlOzEio", "https://www.youtube.com/watch?v=nK1r_9hPWuI"]
    yt_url = st.selectbox("Please select a video to be play", options=yt_urls)

    preprocess_youtube_video(yt_url)
    # pprint(st.session_state.get(yt_url))

    # video_uri = st.session_state[yt_url].get('raw_video')
    

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
