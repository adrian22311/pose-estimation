FROM python:3.9-slim-bullseye

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git

# cv2 dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6

RUN git clone https://github.com/open-mmlab/mmpose.git tmp && mv tmp/* .
# fix wrong config file
# /app/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py
# uncomment line 110, comment lines 111-116
RUN sed -i '110s/^# //' /app/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py && \
    sed -i '111,116s/^/#/' /app/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py

# fix the error: FileNotFoundError: [Errno 2] No such file or directory: '/app/mmpose/.mim/model-index.yml'
RUN mkdir -p /app/mmpose/.mim && cp /app/model-index.yml /app/mmpose/.mim/model-index.yml && cp -r /app/configs /app/mmpose/.mim/configs

COPY lib.py .


# Install pip requirements
# INFO: chumpy package from /app/requirements/runtime.txt may cause errors on Windows
RUN --mount=type=cache,target=/root/.cache/pip sed -i 's/torchvision/#torchvision/g' /app/requirements/runtime.txt && \
    sed -i 's/mmcv/#mmcv/g' /app/requirements/mminstall.txt && \
    python -m pip install -r /app/requirements/runtime.txt && \
    python -m pip install -r /app/requirements/mminstall.txt && \
    python -m pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    python -m pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8/index.html


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Creates a non-root user
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

ENTRYPOINT [ "bash" ]
