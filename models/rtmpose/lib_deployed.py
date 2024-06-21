import os

import numpy as np
import openvino as ov

# from deploy_infer import (
from models.rtmpose.deploy_infer import (
    Compose,
    GetBBoxCenterScale,
    LoadImage,
    PackPoseInputs,
    PoseDataPreprocessor,
    TopdownAffine,
    prepare_data,
    restore_keypoints,
)

MODEL_NM = os.getenv("MODEL_NM")
assert MODEL_NM is not None, "MODEL_NM environment variable is not set."

if "384x288" in MODEL_NM:
    INPUT_SIZE = 288, 384
    OUTPUT_SIZE = 576, 768
else:
    INPUT_SIZE = 192, 256
    OUTPUT_SIZE = 384, 512

if "26keypoints" in MODEL_NM:
    N_KEYPOINTS = 26
else:
    N_KEYPOINTS = 17

core = ov.Core()
model = f"./pose-estimation/out/{MODEL_NM}/{MODEL_NM}.xml"
compiled_model = core.compile_model(
    model=model, device_name="CPU"#, config={"DYN_BATCH_ENABLED": "YES"}
)


# config from POSE_CONFIG
pipeline = Compose(
    [
        LoadImage(),
        GetBBoxCenterScale(),
        TopdownAffine(input_size=INPUT_SIZE),
        PackPoseInputs(),
    ]
)

data_preprocessor = PoseDataPreprocessor(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,  # if isinstance(img, np.ndarray) then check if bgr_to_rgb is required
)


def inference(img: str | np.ndarray, bboxes: np.ndarray = None):
    if bboxes is None:
        h, w, _ = img.shape
        bboxes = np.array([[0, 0, w, h]], dtype=np.float32)

    data_list, preprocessed_image = prepare_data(
        img, bboxes, pipeline, data_preprocessor
    )

    infer_request = compiled_model.create_infer_request()

    input_tensor = ov.Tensor(array=preprocessed_image.numpy(), shared_memory=True)
    infer_request.set_input_tensor(input_tensor)

    infer_request.set_output_tensor(
        0,
        ov.Tensor(
            np.zeros((bboxes.shape[0], N_KEYPOINTS, OUTPUT_SIZE[0]), dtype=np.float32)
        ),
    )
    infer_request.set_output_tensor(
        1,
        ov.Tensor(
            np.zeros((bboxes.shape[0], N_KEYPOINTS, OUTPUT_SIZE[1]), dtype=np.float32)
        ),
    )

    infer_request.start_async()
    infer_request.wait()

    simcc_x = infer_request.get_output_tensor(0).data
    simcc_y = infer_request.get_output_tensor(1).data

    openvino_pred, openvino_scores = restore_keypoints(simcc_x, simcc_y, data_list)

    return {
        "keypoints": openvino_pred[0],
        "keypoint_scores": openvino_scores[0],
    }

if __name__ == "__main__":
    import cv2
    filename = "./sampled/sampled_images/108495_487402.jpg"
    def load_img(filename: str) -> np.ndarray:

        with open(filename, 'rb') as f:
            value = f.read()
        img_np = np.frombuffer(value, np.uint8)
        flag = cv2.IMREAD_COLOR
        img = cv2.imdecode(img_np, flag)
        return img

    img = load_img(filename)
    print(inference(img))
