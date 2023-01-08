# Olaf - The AVQA Application
A web app project for real time question-answering on videos. The app can let users do the following :

- Play and Pause videos
- Take User's voice input (for asking question)
- Returns the answer to the question in speech.

## Data Directory Structure
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

# Manual steps to setup Olaf

Requirements:
Python 3.7+
```
git clone https://github.com/Vishakha2002/olaf.git
cd olaf
python3 -m venv dev_venv
source dev_venv/bin/activate
pip3 install -r requirements.txt
streamlit run olaf.py
```

Note: If torch fails that is okay for frontend dev


# Notes

Update  `/Users/vtyagi/code/olaf/olaf_dev/lib/python3.6/site-packages/pafy/backend_youtube_dl.py` look for fixing the youtube download lib issue
`self._dislikes = 0  #self._ydl_info['dislike_count']`

If you need to disable bootstrap.css.map error use https://stackoverflow.com/questions/21773376/bootstrap-trying-to-load-map-file-how-to-disable-it-do-i-need-to-do-it


### Untar a file
tar -xvf archive.tar.gz
### Tar a file

### Pip install flag to without cache
--no-cache-dir

### fix for urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1091)>
(olaf_3_7_9_env) vtyagi@Vishakhas-MacBook-Pro olaf % open /Applications/Python\ 3.7/Install\ Certificates.command

### ssl proxy
https://github.com/suyashkumar/ssl-proxy




# tunnel to agave compute host
% ssh vtyagi14@agave.asu.edu -L 8501:cg28-2.agave.rc.asu.edu:850
ssh vtyagi14@agave.asu.edu -L 8501:agave.asu.edu:8501


https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

# To download ffmpeg and deploy without root
download `i686-static` from https://www.johnvansickle.com/ffmpeg/
then extract it using
`tar xvf ffmpeg-git-i686-static.tar.xz`
Finally add it to the os path
`export PATH=/home/vishakha/olaf/ffmpeg-git-20220910-i686-static:$PATH`


## How to open port in linux
https://www.digitalocean.com/community/tutorials/opening-a-port-on-linux


## to setup the service

(venv) vishakha@Perl-Lab-PC:~/puzzle$ sudo vim /etc/systemd/system/puzzle.service
(venv) vishakha@Perl-Lab-PC:~/puzzle$ sudo systemctl daemon-reload
(venv) vishakha@Perl-Lab-PC:~/puzzle$ sudo systemctl start puzzle
(venv) vishakha@Perl-Lab-PC:~/puzzle$ sudo systemctl status puzzle