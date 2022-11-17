# Adding the Python base file
FROM python:3.7-slim

# Default Streamlit port
EXPOSE 8501

# setup python
RUN pip install --upgrade pip

# create python3 virtual envionment
RUN python3 -m venv olaf_dev
RUN . olaf_dev/bin/activate

# Set application working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    ffmpeg \
    npm \
    && rm -rf /var/lib/apt/lists/*

# CLone our github repo
RUN git clone https://github.com/Vishakha2002/olaf.git .


# Set application working directory
WORKDIR /app/av_player/frontend
RUN npm install && npm run build

# Back to main app directory
WORKDIR /app

# Install all the python requirements
RUN pip3 install -r requirements.txt

# run the application
ENTRYPOINT ["streamlit", "run", "olaf.py", "--server.port=8501", "--server.address=0.0.0.0"]