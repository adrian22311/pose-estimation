FROM python:3.10-slim-bullseye

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y ffmpeg libsm6 libxext6

# Clone the repository
RUN git clone https://github.com/yangsenius/TransPose.git transpose
ENV PYTHONPATH "/app/transpose"

# Set up the environment
WORKDIR /app/transpose
RUN pip install torch torchvision torchaudio
RUN pip install -r requirements.txt

# Prepare the model
WORKDIR /app
COPY lib.py lib_.py
RUN mkdir data

# Set up the model (download the weights)
RUN python lib_.py

ENTRYPOINT ["bash"]
