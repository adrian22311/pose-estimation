FROM python:3.8

# Clone the repository
WORKDIR /app
RUN git clone https://github.com/daniegr/EfficientPose.git efficient_pose

# Setup the environment
RUN pip install -r efficient_pose/requirements.txt
RUN mv efficient_pose/* . && rm -rf efficient_pose
COPY lib.py .

CMD ["bash"]