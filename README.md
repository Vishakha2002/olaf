# Olaf - The AVQA Application
A web app project for real time question-answering on videos. The app can let users do the following :

- Play and Pause videos
- Take User's voice input (for asking question)
- Returns the answer to the question in speech.

# Steps to setup your application environment in linux
Follow docker steps manually for now.

# Docker installation steps:
## Install docker on your host
Follow instruction on https://docs.docker.com/desktop/install/mac-install/

## Clone the repo once.
`git clone https://github.com/Vishakha2002/olaf.git`

## move into the olaf directory
`cd olaf/`

## Build your docker image
`docker build -t olaf_app .`

## Run your docker image
`docker run -p 8501:8501 olaf_app`

You would have application running on your localhost.


### Install Whisper dependecies
The codebase also depends on a few Python packages, most notably [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) for their fast tokenizer implementation and [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) for reading audio files. The following command will pull and install the latest commit from this repository, along with its Python dependencies 

    pip install git+https://github.com/openai/whisper.git 

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

```

You may need [`rust`](http://rust-lang.org) installed as well, in case [tokenizers](https://pypi.org/project/tokenizers/) does not provide a pre-built wheel for your platform. If you see installation errors during the `pip install` command above, please follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install Rust development environment. Additionally, you may need to configure the `PATH` environment variable, e.g. `export PATH="$HOME/.cargo/bin:$PATH"`. If the installation fails with `No module named 'setuptools_rust'`, you need to install `setuptools_rust`, e.g. by running:

```bash
pip install setuptools-rust
```

Python packages required
```
pip3 install numpy
pip3 install torch
pip3 install tqdm
pip3 install more-itertools
pip3 install transformers==4.19.0
pip3 install 
ffmpeg-python==0.2.0
```


# Notes

Update  `/Users/vtyagi/code/olaf/olaf_dev/lib/python3.6/site-packages/pafy/backend_youtube_dl.py` look for fixing the youtube download lib issue
`self._dislikes = 0  #self._ydl_info['dislike_count']`

If you need to disable bootstrap.css.map error use https://stackoverflow.com/questions/21773376/bootstrap-trying-to-load-map-file-how-to-disable-it-do-i-need-to-do-it
