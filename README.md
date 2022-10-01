# Olaf - The AVQA Application
A web app project for real time question-answering on videos. The app can let users do the following :

- Play and Pause videos
- Take User's voice input (for asking question)
- Returns the answer to the question in speech.

# Steps to setup your application environment

## Create a new Virtual environment
```
python3 -m venv olaf_dev
```

## Activate your virtual environment
```
source ./olaf_dev/bin/activate
```

## Install packages
> **_NOTE:_** It is currently work in progress. Hence its not a complete/exhaustive list of packages required by Olaf app.
```
pip3 install streamlit
pip install streamlit-player
```
# Build and install your st_audiorec frontend app
```
cd st_audiorec/frontend
npm install
npm run build
cd ../..
```

# How to run your app.
Once all the packages are installed, we can run the app using following command.
```
streamlit run olaf.py
```

Update  `/Users/vtyagi/code/olaf/olaf_dev/lib/python3.6/site-packages/pafy/backend_youtube_dl.py` look for fixing the youtube download lib issue
`self._dislikes = 0  #self._ydl_info['dislike_count']`
