FROM python:3.9

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get -y install libgl1
RUN pip install torch requests opencv-python

# Prepare the model
RUN git clone https://github.com/michellelychan/posenet-pytorch.git posenet_lib
COPY lib.py .
ENV PYTHONPATH "/app/posenet_lib"

# Make RUN commands use the new environment:
ENTRYPOINT ["bash"]
