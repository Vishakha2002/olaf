import os
import requests
import streamlit.components.v1 as components
import streamlit as st
from io import BytesIO
import numpy as np

from streamlit_player import st_player, _SUPPORTED_EVENTS


st.title("Olaf - The AVQA Application")

# yt_url = st.text_input("Enter Youtube URL", "https://www.youtube.com/watch?v=DwxmKKD8c3s")
options = {
                "events": st.multiselect("Events to listen", _SUPPORTED_EVENTS, ["onProgress", "onPause"]),
                "progress_interval": 1000,
                # "volume": st.slider("Volume", 0.0, 1.0, 1.0, .01),
                # "playing": st.checkbox("Playing", False),
                # "loop": st.checkbox("Loop", False),
                # "controls": st.checkbox("Controls", True),
                # "muted": st.checkbox("Muted", False),
            }
yt_url = "https://www.youtube.com/watch?v=DwxmKKD8c3s"

event = st_player(yt_url, **options)
print(event.name)
if event.name == "onPause":
    print("video got paused")
if event.data:
    # st.text(f"you paused the video at {event.data.get('playedSeconds')}")
    print(event.data.values())

# Setup for st_audio
parent_dir = os.path.dirname(os.path.abspath(__file__))
print (parent_dir)
build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
st_audiorec = components.declare_component("st_audiorec", path=build_dir)

# STREAMLIT AUDIO RECORDER Instance
val = st_audiorec()

placeholder = st.empty()
if isinstance(val, dict):  # retrieve audio data
    placeholder.text('Question recevieved ...')
    with st.spinner('Transcribing audio-recording...'):
        # yo.empty()
        ind, val = zip(*val['arr'].items())
        ind = np.array(ind, dtype=int)  # convert to np array
        val = np.array(val)             # convert to np array
        sorted_ints = val[ind]
        stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
        wav_bytes = stream.read()
#         print(type(wav_bytes))

        if 'status' not in st.session_state:
            st.session_state['status'] = 'submitted'

        CHUNK_SIZE= 5242880
        audio_file="/Users/vtyagi/Desktop/sky_is_blue.wav"
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

        polling_endpoint = start_transcription()
        transcript = None
        while st.session_state['status'] != 'completed':
            polling_response = requests.get(polling_endpoint, headers=headers)
            st.session_state['status'] = polling_response.json()['status']
            print(f"polling status {polling_response.json()['status']}")
            transcript = polling_response.json()['text']

        placeholder.text('Your Question -')
        st.markdown(transcript)
        st.session_state['status'] = "submitted"