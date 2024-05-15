import cv2
import numpy as np

from lightweight.modules.input_reader import ImageReader
from lightweight.modules.parse_poses import parse_poses
from lightweight.modules.inference_engine_pytorch import InferenceEnginePyTorch

net = InferenceEnginePyTorch("human-pose-estimation-3d.pth", 'CPU', use_tensorrt=False)


def interference(filename):
    """
    Interference the image and return the 2D poses
    :param filename: str, the path of the image
    :return: list, the 2D poses in the format of [x1, y1, score1, ..., x19, y19, score19, score of the pose]
        score is the confidence of the keypoint, -1 means the keyypoint is not detected
    """
    is_video = False
    base_height = 256
    fx = -1
    stride = 8
    frame_provider = ImageReader([filename])
    for frame in frame_provider:
        if frame is None:
            raise ValueError('Frame is None')
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])
        inference_result = net.infer(scaled_img)
        _, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
    return poses_2d[0]
        

if __name__ == "__main__":
    poses_2d = interference("data/sampled_images/17905_2157397.jpg")
    print(poses_2d)
    print(f"Total keypoints: {(len(poses_2d) - 1) // 3}")
    print(f"Numbers of keypoints: {len(list(filter(lambda x: x != -1, poses_2d))) // 3}")
