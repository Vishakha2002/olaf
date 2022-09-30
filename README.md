# Olaf - The AVQA Application

# Steps to setup your application environment

## Create a new Virtual environment

python3 -m venv olaf_dev

## Activate your virtual environment

source ./olaf_dev/bin/activate

## Install packages
> **_NOTE:_** It is currently work in progress. Hence its not a complete/exhaustive list of packages required by Olaf app.
pip3 install streamlit


# How to run your app.
Once all the packages are installed, we can run the app using following command.
`streamlit run olaf.py`

Update  `/Users/vtyagi/code/olaf/olaf_dev/lib/python3.6/site-packages/pafy/backend_youtube_dl.py` look for
```
    self._dislikes = 0  #self._ydl_info['dislike_count']

```
