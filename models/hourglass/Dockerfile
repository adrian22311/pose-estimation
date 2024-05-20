FROM python:3.10.4

# Clone the repository
WORKDIR /app
RUN git clone https://github.com/dkurzend/Soft_Gated_Pose_Estimation_Pytorch.git soft_gated

# Setup the environment
ENV PYTHONPATH "/app/soft_gated:/app/soft_gated/utils"
RUN pip install -r soft_gated/requirements.txt
RUN apt-get update && apt-get install -y libopencv-dev
COPY lib.py .

# Download the pretrained model
WORKDIR /app/checkpoints
RUN wget https://matix.li/9680ec1eb999 -O SHG_model_weights.tar
WORKDIR /app

# Switch the model.py file
COPY model.py /app/soft_gated/utils/model.py
COPY config.py /app/soft_gated/config.py

CMD [ "bash" ]
