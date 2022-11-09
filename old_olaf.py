import os
import streamlit.components.v1 as components
import streamlit as st
import requests
from io import BytesIO
import numpy as np

st.set_page_config(
    layout="wide",
)

# Setup for streamlit_player
_SUPPORTED_EVENTS = [
    "onStart", "onPlay", "onProgress", "onDuration", "onPause",
    "onBuffer", "onBufferEnd", "onSeek", "onEnded", "onError"
]
options = {
    "events": ["onProgress", "onPause"],
    "progress_interval": 1000
}

yt_url = "https://www.youtube.com/watch?v=bmIVWe3Cux8&iv_load_policy=3"

av_player_parent_dir = os.path.dirname(os.path.abspath(__file__))
av_player_build_dir = os.path.join(av_player_parent_dir, "av_player/frontend/build")
av_player = components.declare_component("streamlit_player", path=av_player_build_dir)

# STREAMLIT Video Player Instance
video_event = av_player(url=yt_url, width ="800px", height = "400px", **options)
if isinstance(video_event, dict):
    try:
        if video_event.get('name') == "onPause":
            print("video got paused")
        if video_event.get('name') == "onProgress":
            if video_event.get('data') and type(video_event.get('data') == dict):
                frameStoppedAt = video_event["data"].get("playedSeconds")
                st.text(f"you paused the video at {frameStoppedAt}")
    except KeyError:
        print("\n")

placeholder = st.empty()
try:
    if isinstance(video_event, dict):  # retrieve audio data
        if "arr" in video_event.keys():
            with st.spinner('Transcribing audio-recording...'):
                placeholder.text('Question recevieved ...')
                ind, val = zip(*video_event['arr'].items())
                ind = np.array(ind, dtype=int)  # convert to np array
                val = np.array(val)             # convert to np array
                sorted_ints = val[ind]
                stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
                wav_bytes = stream.read()
                print(type(wav_bytes))

                if 'status' not in st.session_state:
                    st.session_state['status'] = 'submitted'

                CHUNK_SIZE= 5242880
                upload_endpoint = "https://api.assemblyai.com/v2/upload"
                transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

                headers = {
                    "authorization":"34e0f41f2b84459db61bb5da43686224",
                    "contect-type": "application/json"
                }

                def read_file(filename):
                    with open(filename, 'rb') as _file:
                        while True:
                            data = _file.read(CHUNK_SIZE)
                            if not data:
                                break
                            yield data

                def start_transcription():
                    # upload_response = requests.post(
                    #     upload_endpoint,
                    #     headers=headers, data=read_file(audio_file)
                    # )
                    upload_response = requests.post(
                        upload_endpoint,
                        headers=headers, data=wav_bytes
                    )
                    audio_url = upload_response.json()['upload_url']
                    print(f"Uploaded to {audio_url}")

                    transcript_request = {
                        'audio_url': audio_url,
                        'iab_categories': 'False',
                    }
                    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)

                    transcript_id = transcript_response.json()['id']
                    polling_endpoint = transcript_endpoint + "/" + transcript_id
                    return polling_endpoint

                # polling_endpoint = start_transcription()
                transcript = None
                # while st.session_state['status'] != 'completed':
                #     polling_response = requests.get(polling_endpoint, headers=headers)
                #     st.session_state['status'] = polling_response.json()['status']
                #     print(f"polling status {polling_response.json()['status']}")
                #     transcript = polling_response.json()['text']

                # placeholder.text('Your Question -')
                # st.markdown(transcript)
                # st.session_state['status'] = "submitted"
                transcript = "This is a test"
                placeholder.text('Your Question -')
                st.markdown(transcript)
                st.session_state['status'] = "submitted"

except KeyError:
    print("\n")


# Modified version from https://github.com/stefanrmmr/streamlit_audio_recorder
# Setup for st_audio
# st_audio_parent_dir = os.path.dirname(os.path.abspath(__file__))
# st_audio_build_dir = os.path.join(st_audio_parent_dir, "st_audiorec/frontend/build")
# st_audiorec = components.declare_component("st_audiorec", path=st_audio_build_dir)

# STREAMLIT AUDIO RECORDER Instance
# val = st_audiorec()

# placeholder = st.empty()
# if isinstance(val, dict):  # retrieve audio data
#     placeholder.text('Question recevieved ...')
#     with st.spinner('Transcribing audio-recording...'):
# #         ind, val = zip(*val['arr'].items())
# #         ind = np.array(ind, dtype=int)  # convert to np array
# #         val = np.array(val)             # convert to np array
# #         sorted_ints = val[ind]
# #         stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
# #         wav_bytes = stream.read()
# # #         print(type(wav_bytes))

# #         if 'status' not in st.session_state:
# #             st.session_state['status'] = 'submitted'

# #         CHUNK_SIZE= 5242880
# #         audio_file="/Users/vtyagi/Desktop/sky_is_blue.wav"
# #         upload_endpoint = "https://api.assemblyai.com/v2/upload"
# #         transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

# #         headers = {
# #             "authorization":"34e0f41f2b84459db61bb5da43686224",
# #             "contect-type": "application/json"
# #         }

# #         def read_file(filename):
# #             with open(filename, 'rb') as _file:
# #                 while True:
# #                     data = _file.read(CHUNK_SIZE)
# #                     if not data:
# #                         break
# #                     yield data

# #         def start_transcription():
# #             # upload_response = requests.post(
# #             #     upload_endpoint,
# #             #     headers=headers, data=read_file(audio_file)
# #             # )
# #             upload_response = requests.post(
# #                 upload_endpoint,
# #                 headers=headers, data=wav_bytes
# #             )
# #             audio_url = upload_response.json()['upload_url']
# #             print(f"Uploaded to {audio_url}")

# #             transcript_request = {
# #                 'audio_url': audio_url,
# #                 'iab_categories': 'False',
# #             }
# #             transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)

# #             transcript_id = transcript_response.json()['id']
# #             polling_endpoint = transcript_endpoint + "/" + transcript_id
# #             return polling_endpoint

# #         polling_endpoint = start_transcription()
# #         transcript = None
# #         while st.session_state['status'] != 'completed':
# #             polling_response = requests.get(polling_endpoint, headers=headers)
# #             st.session_state['status'] = polling_response.json()['status']
# #             print(f"polling status {polling_response.json()['status']}")
# #             transcript = polling_response.json()['text']
        # transcript = "This is a test"
        # placeholder.text('Your Question -')
        # st.markdown(transcript)
        # st.session_state['status'] = "submitted"