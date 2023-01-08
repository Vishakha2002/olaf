"""
https://github.com/suyashkumar/ssl-proxy If you need reverse proxy in front of streamlit server
"""

import json
import logging
import os
import re
import tarfile
import time
import shutil
from io import BytesIO

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import whisper

from moviepy.editor import VideoFileClip
from pytube import YouTube
from torch.utils.data import DataLoader
from torchvision import transforms

from audio_feature.extract_audio_vggish_feat import generate_audio_vggish_features
from music_avqa import OlafBatchInput, OlafSingleInput, ToTensor, test
from net_grd_avst.net_avst import AVQA_Fusion_Net
from video_feature.extract_resnet18_14x14 import extract_video_feature

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
log = logging.getLogger(__name__)
log.addHandler(ch)

# This we are doing for vggish
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # set gpu number


def whisper_transcription(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    # pprint(result)
    return result["text"]


def transcribe_question(video_event, default="assemblyai"):
    text = ""
    # XXX Vishakha move this to a temp file.
    audio_file = "data/user_question/question_audio.wav"

    ind, val = zip(*video_event["arr"].items())
    ind = np.array(ind, dtype=int)  # convert to np array
    val = np.array(val)  # convert to np array
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
    if video_event and video_event.get("name") == "onPause":
        st.session_state.is_video_paused = True
        log.debug("video got paused")
    if video_event and video_event.get("name") == "onProgress":

        if video_event.get("data") and isinstance(video_event.get("data"), dict):
            frame_stopped_at = video_event["data"].get("playedSeconds")
            log.debug(
                f"you paused the video at {frame_stopped_at} session state {st.session_state.is_video_paused}"
            )
            if st.session_state.is_video_paused:
                log.debug(f"you paused the video at {frame_stopped_at}")
                st.session_state.frame_stopped_at = frame_stopped_at
                st.session_state.is_video_paused = False


def extract_audio_features():
    """
    VGGish:
    The initial AudioSet release included 128-dimensional embeddings of each
    AudioSet segment produced from a VGG-like audio classification model that was
    trained on a large YouTube dataset.
    We provide a TensorFlow definition of this model, which we call VGGish, as
    well as supporting code to extract input features for the model from audio
    waveforms and to post-process the model embedding output into the same format as
    the released embedding features.

    VGGish can be used in two ways:

    As a feature extractor: VGGish converts audio input features into a
    semantically meaningful, high-level 128-D embedding which can be fed as input
    to a downstream classification model. The downstream model can be shallower
    than usual because the VGGish embedding is more semantically compact than raw
    audio features.
    So, for example, you could train a classifier for 10 of the AudioSet classes
    by using the released embeddings as features.  Then, you could use that
    trained classifier with any arbitrary audio input by running the audio through
    the audio feature extractor and VGGish model provided here, passing the
    resulting embedding features as input to your trained model.
    vggish_inference_demo.py shows how to produce VGGish embeddings from
    arbitrary audio.


    As part of a larger model: Here, we treat VGGish as a "warm start" for the
    lower layers of a model that takes audio features as input and adds more
    layers on top of the VGGish embedding. This can be used to fine-tune VGGish
    (or parts thereof) if you have large datasets that might be very different
    from the typical YouTube video clip. vggish_train_demo.py shows how to add
    layers on top of VGGish and train the whole model.

    Source:
    https://git.dst.etit.tu-chemnitz.de/external/tf-models/-/tree/1d057dfc32f515a63ab1e23fd72052ab2a954952/research/audioset/vggish

    Tutorial https://www.youtube.com/watch?v=PYlr8ayHb4g
    Vishakha - Do we even need to do this?
    https://colab.research.google.com/drive/1TbX92UL9sYWbdwdGE0rJ9owmezB-Rl1C?usp=sharing

    Pre-requisite
    !curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
    !curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz

    we are going to download them in
    $ ls
    models/
    vggish_model.ckpt
    vggish_pca_params.npz

    """
    pass


def extract_frames(video, dst):

    command1 = "ffmpeg "
    command1 += "-i " + video + " "
    command1 += "-y" + " "
    command1 += "-r " + "1 "
    command1 += "{0}/%06d.jpg".format(dst)

    log.info(command1)

    exit_code = os.system(command1)

    if exit_code != 0:
        log.error(f"Not able to run ffmpeg. command {command1}")
        raise Exception("Not able to run ffmpeg")

    return


def get_audio_wav(audio_filename, video_path) -> str:
    """
    Given the video file(i.e. video_path), extract audio and save it as audio_filename
    """
    raw_audio_dir = "data/raw_audio"
    saved_audio = os.path.join(os.getcwd(), raw_audio_dir, audio_filename)

    if audio_filename not in os.listdir(raw_audio_dir):
        video = VideoFileClip(video_path)
        audio_file = video.audio
        audio_file.write_audiofile(saved_audio, fps=16000)
        log.info(f"finish video id: {saved_audio}")

    return saved_audio


def preprocess_youtube_video(yt_url, frontend_dev):
    """
    To take user input it is import to preprocess the video. Steps:
    1. Download youtube video  (720p resoultion / mp4 format) and save it in data/raw_video
    2. Extract audio (.wav) from the downloaded video and save it in data/raw_audio
    3. Extract frames from video using ffmeg
    4. Run VGGish
    5. Run Resnet18
    """
    saved_audio = None
    yt_urls = []
    dictionary = {}

    # Check if the youtube url has been preprocessed
    with open("data/preprocessed_urls.txt") as my_file:
        yt_urls = [line.rstrip("\n") for line in my_file]

        if yt_url in yt_urls:
            log.info(f"{yt_url} is already processed.")
            return

    video_object = YouTube(yt_url)
    video_title = re.sub(r"[^A-Za-z0-9 ]+", "", video_object.title)
    video_title = video_title.replace(" ", "_")
    video_frame_path = os.path.join(os.getcwd(), "data/frames/video", video_title)

    video_filename = video_title + ".mp4"
    audio_filename = video_title + ".wav"

    extracted_frames = False
    log.info(f"Preprocessing Video {video_title}")
    with st.spinner(f"Preprocessing Video {video_title}"):
        torch.cuda.empty_cache()
        # start video download
        saved_video = (
            video_object.streams.filter(mime_type="video/mp4", res="720p")
            .first()
            .download(filename=video_filename, output_path="data/raw_video")
        )
        log.info(f"Downloaded video at {saved_video}")

        if not frontend_dev:
            # Extract Audio from the saved video
            saved_audio = get_audio_wav(
                audio_filename=audio_filename, video_path=saved_video
            )
            log.info(f"Extracted audio saved at {saved_audio}")

            # Here we will extract frames from the saved video file
            try:
                if not os.path.exists(video_frame_path):
                    log.info(
                        f"Creating a new folder for extracting video frames {video_frame_path}"
                    )
                    os.makedirs(video_frame_path)
                frame_count = len(os.listdir(video_frame_path))
                if frame_count == 0:
                    extract_frames(saved_video, video_frame_path)
                    frame_count = len(os.listdir(video_frame_path))
                log.info(
                    f"Extracted Video frames saved at {video_frame_path}. Count: {frame_count}"
                )
                extracted_frames = True

                vggish_audio_feature_file_path = generate_audio_vggish_features(
                    saved_audio
                )
                resnet_video_feature_file_path = extract_video_feature(video_frame_path)
            except Exception:
                log.exception(f"Unable to process video {video_title}/ url {yt_url}")
                raise

        preprocess_state = {
            "video_title": video_title,
            "raw_audio": saved_audio,
            "raw_video": saved_video,
            "video_frame_path": video_frame_path,
            "vggish_audio_feature_file_path": os.path.join(
                os.getcwd(), vggish_audio_feature_file_path
            ),
            "resnet_video_feature_file_path": os.path.join(
                os.getcwd(), resnet_video_feature_file_path
            ),
            "extracted_frames": extracted_frames,
        }

        with open("data/preprocessed_urls.txt", "a", encoding="utf-8") as my_file:
            my_file.write(yt_url + "\n")

        with open("data/preprocessed_urls_metadata.txt") as my_file:
            dictionary = json.load(my_file)

        dictionary[yt_url] = preprocess_state

        with open("data/preprocessed_urls_metadata.txt", "w") as outfile:
            json.dump(dictionary, outfile)


def run_batch_pre_processing(frontend_dev):
    """
        XXX Questions for video 19   // Video 19
    //   {
    //         "video_id": "00000272",
    //         "question_id": 50,
    //         "type": "[\"Audio-Visual\", \"Temporal\"]",
    //         "question_content": "Where is the <FL> sounding instrument?",
    //         "templ_values": "[\"first\"]",
    //         "question_deleted": 0,
    //         "anser": "left"
    //     },


    //   {
    //     "video_id": "00000259",
    //     "question_id": 157,
    //     "type": "[\"Audio-Visual\", \"Counting\"]",
    //     "question_content": "How many types of musical instruments sound in the video?",
    //     "templ_values": "[]",
    //     "question_deleted": 0,
    //     "anser": "two"
    //   },

    //   {
    //     "video_id": "00007052",
    //     "question_id": 1751,
    //     "type": "[\"Audio-Visual\", \"Existential\"]",
    //     "question_content": "Is this sound from the instrument in the video?",
    //     "templ_values": "[]",
    //     "question_deleted": 0,
    //     "anser": "yes"
    //   },
     this needs to be removed later.
    """
    yt_urls = [
        "https://www.youtube.com/watch?v=6gQ7m0c4ReI",  # Video1
        "https://youtu.be/is68rlOzEio",  # Video2
        "https://youtu.be/5sHwuARMXj0",  # Video3
        "https://youtu.be/zcyatK8qc7c",  # Video4
        "https://youtu.be/3dOATkCOguI",  # Video5
        "https://youtu.be/2m9fqUQgzUA",  # Video6
        "https://youtu.be/mQpzKwSxiOw",  # Video7
        "https://youtu.be/NqSKXimnl6Y",  # Video8
        "https://youtu.be/P3LjmYl4Yd8",  # Video9
        # "https://youtu.be/R3SOHWhJ398",
        # "https://youtu.be/MCO2-ikxe1I",
        "https://youtu.be/rsJgPnWDflU",  # Video10
        "https://youtu.be/AHlG_PwwvGY",  # Video11
        "https://youtu.be/BIPTE84l9ls",  # Video12
        "https://youtu.be/iveZwfEhZYI",  # Video13
        "https://youtu.be/Oipg71dSem0",  # Video14
        "https://youtu.be/-PWiegZQfAY",  # Video15
        "https://youtu.be/1Asc6IaPunU",  # Video16
        "https://youtu.be/l0JKkmztuRE",  # Video17
        "https://youtu.be/pxoW-00Zyho",  # Video18
        # "https://youtu.be/W37hiyMDJnE",  # Video19
        "https://youtu.be/WRe2sz5l9JE",  # Video20
    ]
    video_count = len(yt_urls)
    placeholder = st.empty()
    with st.spinner(
        f"Starting batch pre processing on {video_count} youtube videos..."
    ):
        for count, url in enumerate(yt_urls):
            placeholder.write(f"Preprocessing url {count+1}/{video_count}. Url - {url}")
            preprocess_youtube_video(url, frontend_dev)

    placeholder.write(
        f"Batch pre processing of {video_count} videos finished successfully"
    )


def run_batch_through_avqa_model():
    test_dataset = OlafBatchInput(
        label="./data/pretrained/olaf-test.json",
        audio_dir="/home/vishakha/olaf/data/features/audio_vggish",
        video_res14x14_dir="/home/vishakha/olaf/data/features/video_resnet18",
        transform=transforms.Compose([ToTensor()]),
        mode_flag="test",
    )
    print(test_dataset.__len__())
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    model = AVQA_Fusion_Net()
    model = nn.DataParallel(model)
    model = model.to("cuda")
    model.load_state_dict(torch.load("net_grd_avst/avst_models/avst.pt"))
    test(model, test_loader)


def initialize_session_state():
    """
    This method sets the variables required for olaf to work in
    streamlit session state.
    """
    if "is_video_paused" not in st.session_state:
        st.session_state["is_video_paused"] = False
    if "frame_stopped_at" not in st.session_state:
        st.session_state["frame_stopped_at"] = 0


def get_resp_from_avqa_model(olaf_pre_process_context, transcription):
    # Getting Olafdataset
    placeholder_2 = st.empty()
    with st.spinner("Getting response from MUSIC_AVQA Model..."):


        olaf_input_obj = OlafSingleInput(
            vocab_label="data/json/avqa-test.json",
            audio_vggish_features_dir="data/features/audio_vggish",
            video_res14x14_dir="data/features/video_resnet18",
            current_answer="one",
            transform=transforms.Compose([ToTensor()]),
            # current_question = transcription,
            current_question="How many instruments are being played?",
            olaf_context=olaf_pre_process_context,
            is_batch=False,
        )
        # answer_labels = olaf_input_obj.answer_label
        model = AVQA_Fusion_Net()
        model = nn.DataParallel(model)
        model = model.to("cuda")
        test_loader = DataLoader(olaf_input_obj, batch_size=1, pin_memory=True)
        model.load_state_dict(torch.load("net_grd_avst/avst_models/avst.pt"))
        model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                print(f"Vishakha batch id here is {batch_idx}")
                audio = sample["audio"].to("cuda")
                visual_posi = sample["visual_posi"].to("cuda")
                visual_nega = sample["visual_nega"].to("cuda")
                question = sample["question"].to("cuda")
                target = sample["label"].to("cuda")
                preds_qa, out_match_posi, out_match_nega = model(
                    audio, visual_posi, visual_nega, question
                )
                preds = preds_qa
                # print(preds)
                _, predicted = torch.max(preds.data, 1)
                is_correct = target == predicted

                print(type(preds))
                print(preds)
                print(predicted)
                placeholder_2.write(is_correct)

def main(frontend_dev):
    """
    Olaf main application script.
    """
    initialize_session_state()

    is_batch_input = st.radio("Is batch processing?", ["No", "Yes"], horizontal=True)
    is_batch = is_batch_input == "Yes"
    # is_batch =False

    # Setup for streamlit_player
    _SUPPORTED_EVENTS = [
        "onStart",
        "onPlay",
        "onProgress",
        "onDuration",
        "onPause",
        "onBuffer",
        "onBufferEnd",
        "onSeek",
        "onEnded",
        "onError",
    ]
    options = {
        "events": _SUPPORTED_EVENTS,
        "progress_interval": 1000,
    }

    if is_batch:
        run_batch_pre_processing(frontend_dev)
        run_batch_through_avqa_model()

    else:
        # Variable that need to be fed to AVQA Model.
        olaf_pre_process_context = {}

        # XXX Take user input instead of harcoding.
        yt_urls = [
            "https://www.youtube.com/watch?v=6gQ7m0c4ReI",
            "https://youtu.be/is68rlOzEio",
        ]
        yt_url = st.selectbox("Please select a video to be play", options=yt_urls)

        preprocess_youtube_video(yt_url, frontend_dev)
        with open("data/preprocessed_urls_metadata.txt") as my_file:
            olaf_pre_process_context = json.load(my_file)

        if frontend_dev:
            st.write(
                f"Here the video is saved at {olaf_pre_process_context[yt_url]['raw_video']}"
            )

        av_player_parent_dir = os.path.dirname(os.path.abspath(__file__))
        av_player_build_dir = os.path.join(
            av_player_parent_dir, "av_player/frontend/build"
        )
        av_player = components.declare_component(
            "streamlit_player", path=av_player_build_dir
        )

        # STREAMLIT Video Player Instance
        video_event = av_player(url=yt_url, width="500px", height="300px", **options)

        parse_video_event(video_event)
        placeholder = st.empty()
        if isinstance(video_event, dict):  # retrieve audio data
            if "arr" in video_event.keys():
                with st.spinner("Transcribing Question..."):
                    if not frontend_dev:
                        transcription = transcribe_question(
                            video_event, default="whisper"
                        )
                        st.session_state["current_question"] = transcription
                        log.info(f"User Question: {transcription}")

                        placeholder.text(f"User Question: {transcription}")
                    else:
                        placeholder.text("Testing Frontend code")

                get_resp_from_avqa_model(olaf_pre_process_context, transcription)

@st.cache
def setup_frontend():
    shutil.rmtree('av_player/frontend/')
    my_tar = tarfile.open('av_player/frontend.tar.gz')
    my_tar.extractall("./av_player/")
    my_tar.close()

@st.cache
def download_required_files():
    if os.path.exists("vggish_model.ckpt"):
        log.info("Found vggish_model.ckpt, no need to download it")
    else:
        URL = "https://storage.googleapis.com/audioset/vggish_model.ckpt"
        response = requests.get(URL)
        open("vggish_model.ckpt", "wb").write(response.content)
        log.info("Downloaded vggish_model.ckpt")
    if os.path.exists("ffmpeg_dir/ffmpeg-git-20220910-i686-static/ffmpeg"):
        log.info("ffmpeg is already downloaded")
    else:
        log.info("Downloading ffmpeg")
        os.makedirs("ffmpeg_dir")

        URL = "https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-i686-static.tar.xz"
        response = requests.get(URL)
        open("ffmpeg-git-i686-static.tar.xz", "wb").write(response.content)

        ffmpeg_file = tarfile.open('ffmpeg-git-i686-static.tar.xz')
        ffmpeg_file.extractall('./ffmpeg_dir')
        ffmpeg_file.close()
        log.info("Setting ffmpeg file path")

    if "ffmpeg_dir" not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + os.path.join(os.getcwd(), "ffmpeg_dir/ffmpeg-git-20220910-i686-static/")


@st.cache
def setup_directory() -> None:
    """
    Before you begin setup data directories
    data/raw_audio                      : Path for extracted Audio from the youtube video
    data/raw_video                      : Path for downloaded Video from youtube
    data/frames/audio
    data/frames/video                   : Path for extracted frames from video
    data/features/audio_vggish          : Path for extracted VGGish features from audia waveform
    data/features/video_resnet18        : PAth for extracted Video features using resent features
    data/user_question                  : Path to store user question audio
    data/preprocessed_urls.txt          : Path to append preprocessed yt urls
    data/preprocessed_urls_metadata.txt : Path to add preprocessing metadata for a processed yt url
    logs                                : Path to save olaf logs

    """
    if not os.path.exists("data/raw_audio"):
        os.makedirs("data/raw_audio")
    if not os.path.exists("data/raw_video"):
        os.makedirs("data/raw_video")
    if not os.path.exists("data/user_question"):
        os.makedirs("data/user_question")

    if not os.path.exists("data/frames"):
        os.makedirs("data/frames")
    if not os.path.exists("data/frames/audio"):
        os.makedirs("data/frames/audio")
    if not os.path.exists("data/frames/video"):
        os.makedirs("data/frames/video")

    if not os.path.exists("data/features"):
        os.makedirs("data/features")
    if not os.path.exists("data/features/audio_vggish"):
        os.makedirs("data/features/audio_vggish")
    if not os.path.exists("data/features/video_resnet18"):
        os.makedirs("data/features/video_resnet18")
    if not os.path.exists("data/pretrained"):
        os.makedirs("data/pretrained")
    if not os.path.exists("data/preprocessed_urls.txt"):
        open("data/preprocessed_urls.txt", "a").close()
    if not os.path.exists("data/preprocessed_urls_metadata.txt"):
        with open("data/preprocessed_urls_metadata.txt", "w") as foo:
            tmp = {}
            json.dump(tmp, foo)
    if not os.path.exists("logs"):
        os.makedirs("logs")


# if __name__ == "__main__":
#     """
#     Olaf main script
#     """
log.info("Staring Olaf Application")
log.info("Setting up directories for Olaf Application")
setup_directory()
log.info("Running Olaf main script")
download_required_files()
setup_frontend()
print(os.environ["PATH"])
main(frontend_dev=False)
