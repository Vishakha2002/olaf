import contextlib
import logging
import os
import wave

import numpy as np
import tensorflow as tf

from . import vggish_input, vggish_params, vggish_slim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # set gpu number

log = logging.getLogger(__name__)

# get audio length
def get_audio_len(audio_file):
    # audio_file = os.path.join(audio_path, audio_name)
    with contextlib.closing(wave.open(audio_file, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        wav_length = int(frames / float(rate))
        # print("wave_len: ", wav_length)

        return wav_length


def generate_audio_vggish_features(filename):
    """
    Function extracts the audio features and retrun the path where these
    features are saved
    """
    # Paths to downloaded VGGish files.
    checkpoint_path = "vggish_model.ckpt"

    audio_dir = "data/raw_audios"  # .wav audio files
    save_dir = "data/features/audio_vggish/"

    log.info(f"Starting to extract viggish audio features for {filename}")
    raw_name = filename.split("/")[-1]
    raw_name = raw_name.split(".")[0]

    # log(f"Vishakha raw file name {raw_name}")

    outfile = os.path.join(save_dir, raw_name + ".npy")
    if os.path.exists(outfile):
        log.info(f" {outfile} already exist! ")
        return outfile

    """feature learning by VGG-net trained by audioset"""
    audio_index = os.path.join(audio_dir, filename)  # path of your audio files
    # num_secs = 60
    num_secs_real = get_audio_len(audio_index)

    input_batch = vggish_input.wavfile_to_examples(audio_index, num_secs_real)
    np.testing.assert_equal(
        input_batch.shape,
        [num_secs_real, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS],
    )

    # Define VGGish, load the checkpoint, and run the batch through the model to
    # produce embeddings.
    # with tf.Graph().as_default(), tf.Session() as sess:
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME
        )
        [embedding_batch] = sess.run(
            [embedding_tensor], feed_dict={features_tensor: input_batch}
        )
        # print(f'VGGish embedding: {embedding_batch[0]}')

        np.save(outfile, embedding_batch)
        log.info(f" save info: {outfile} ---> {embedding_batch.shape}")
        return outfile
