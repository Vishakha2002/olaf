import os
import time
import streamlit.components.v1 as components
import streamlit as st
import requests
from io import BytesIO
import numpy as np

# try:
#     if isinstance(video_event, dict):  # retrieve audio data
#         if "arr" in video_event.keys():
#             with st.spinner('Transcribing audio-recording...'):
#                 placeholder.text('Question recevieved ...')
#                 ind, val = zip(*video_event['arr'].items())
#                 ind = np.array(ind, dtype=int)  # convert to np array
#                 val = np.array(val)             # convert to np array
#                 sorted_ints = val[ind]
#                 stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
#                 # This wav_bytes has the audio
#                 wav_bytes = stream.read()
#                 print(type(wav_bytes))

#                 if 'status' not in st.session_state:
#                     st.session_state['status'] = 'submitted'

#                 CHUNK_SIZE= 5242880
#                 upload_endpoint = "https://api.assemblyai.com/v2/upload"
#                 transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

#                 headers = {
#                     "authorization":"34e0f41f2b84459db61bb5da43686224",
#                     "contect-type": "application/json"
#                 }

#                 def read_file(filename):
#                     with open(filename, 'rb') as _file:
#                         while True:
#                             data = _file.read(CHUNK_SIZE)
#                             if not data:
#                                 break
#                             yield data

#                 def start_transcription():
#                     # upload_response = requests.post(
#                     #     upload_endpoint,
#                     #     headers=headers, data=read_file(audio_file)
#                     # )
#                     upload_response = requests.post(
#                         upload_endpoint,
#                         headers=headers, data=wav_bytes
#                     )
#                     audio_url = upload_response.json()['upload_url']
#                     print(f"Uploaded to {audio_url}")

#                     transcript_request = {
#                         'audio_url': audio_url,
#                         'iab_categories': 'False',
#                     }
#                     transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)

#                     transcript_id = transcript_response.json()['id']
#                     polling_endpoint = transcript_endpoint + "/" + transcript_id
#                     return polling_endpoint

#                 polling_endpoint = start_transcription()
#                 transcript = None
#                 while st.session_state['status'] != 'completed':
#                     polling_response = requests.get(polling_endpoint, headers=headers)
#                     st.session_state['status'] = polling_response.json()['status']
#                     # print(f"polling status {polling_response.json()['status']}")
#                     transcript = polling_response.json()['text']

#                 placeholder.text('Your Question -')
#                 st.markdown(transcript)
#                 st.session_state['status'] = "submitted"
#                 # transcript = "This is a test"
#                 # placeholder.text('Your Question -')
#                 # st.markdown(transcript)
#                 # st.session_state['status'] = "submitted"

# except KeyError:
#     print("\n")

def transcribe_question(video_event, default="assemblyai"):
    if default == "assemblyai":
        time.sleep(3)
        return "Not Implemented"
    

def parse_video_event(video_event):
    # print(st.session_state.is_video_paused)
    if video_event and video_event.get('name') == "onPause":
            st.session_state.is_video_paused = True 
            # print("video got paused")
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
    if "is_video_paused" not in st.session_state:
        st.session_state['is_video_paused'] = False
    if "frame_stopped_at" not in st.session_state:   
        st.session_state['frame_stopped_at'] = 0


def main():
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
                transcription = transcribe_question(video_event)
                placeholder.text(transcription)


if __name__ == "__main__":
    main()
