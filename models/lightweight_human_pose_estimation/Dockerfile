FROM python:3.8

WORKDIR /app

# Clone the repository
RUN git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch.git lightweight

# Download the pretrained model
RUN pip install gdown
RUN gdown 1niBUbUecPhKt3GyeDNukobL4OQ3jqssH

# CMake and opencv
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y libopencv-dev

# Set up the environment
WORKDIR /app/lightweight
RUN pip install -r requirements.txt
RUN python setup.py build_ext
ENV PYTHONPATH "/app/lightweight/pose_extractor/build:/app/lightweight"

# Prepare the model
WORKDIR /app
COPY lib.py .

ENTRYPOINT ["bash"]
