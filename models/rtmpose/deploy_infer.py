from typing import Union

import numpy as np
import openvino as ov

#
# from deploy_mmpose_replacement import (
from models.rtmpose.deploy_mmpose_replacement import (
    Compose,
    GetBBoxCenterScale,
    LoadImage,
    PackPoseInputs,
    PoseDataPreprocessor,
    TopdownAffine,
    get_simcc_maximum,
    pseudo_collate,
)


def prepare_data(
    img: Union[str, np.ndarray],
    bboxes: np.ndarray,
    pipeline: Compose,
    data_preprocessor: PoseDataPreprocessor,
):
    data_list = []
    for bbox in bboxes:
        if isinstance(img, str):
            data_info = dict(img_path=img)
        else:
            data_info = dict(img=img)
        data_info["bbox"] = bbox[None]  # shape (1, 4)
        data_info["bbox_score"] = np.ones(1, dtype=np.float32)  # shape (1,)
        data_list.append(pipeline(data_info))
    if data_list:
        batch = pseudo_collate(data_list)

    preprocessed_image = data_preprocessor(batch)["inputs"]
    return data_list, preprocessed_image


def restore_keypoints(output_1, output_2, data_list):
    keypoints, scores = get_simcc_maximum(output_1, output_2)
    simcc_split_ratio = 2  # 2 for all models (source: config `codec`)
    keypoints = keypoints / simcc_split_ratio

    n_pred = len(data_list)
    n_keypoints = keypoints.shape[1]
    samples_metainfo = [data_list[i]["data_samples"]["metainfo"] for i in range(n_pred)]

    input_centers = np.array(
        [samples_metainfo[i]["input_center"] for i in range(n_pred)]
    )
    input_scales = np.array([samples_metainfo[i]["input_scale"] for i in range(n_pred)])
    input_sizes = np.array([samples_metainfo[i]["input_size"] for i in range(n_pred)])

    input_centers = np.repeat(
        np.expand_dims(input_centers, axis=1), n_keypoints, axis=1
    )
    input_scales = np.repeat(np.expand_dims(input_scales, axis=1), n_keypoints, axis=1)
    input_sizes = np.repeat(np.expand_dims(input_sizes, axis=1), n_keypoints, axis=1)

    openvino_pred = (
        keypoints / input_sizes * input_scales + input_centers - 0.5 * input_scales
    )
    openvino_scores = scores

    return openvino_pred, openvino_scores


if __name__ == "__main__":
    import os

    MODEL_NM = os.getenv("MODEL_NM", "rtm_body8_det-m_pose-m")
    IMG_PATH = "./sampled/sampled_images/108495_487402.jpg"

    assert MODEL_NM is not None, "MODEL_NM environment variable is not set."

    if "384x288" in MODEL_NM:
        input_size = 288, 384
        output_size = 576, 768
    else:
        input_size = 192, 256
        output_size = 384, 512
    if "26keypoints" in MODEL_NM:
        n_keypoints = 26
    else:
        n_keypoints = 17


    core = ov.Core()
    model = f"./out/{MODEL_NM}/{MODEL_NM}.xml"
    compiled_model = core.compile_model(
        model=model, device_name="CPU"#, config={"DYN_BATCH_ENABLED": "YES"}
    )

    # pipeline = Compose(pose_estimator.cfg.test_dataloader.dataset.pipeline)
    # pose_estimator.cfg.test_dataloader.dataset.pipeline -> val_pipeline = [
    #     dict(type='LoadImage', backend_args=backend_args),
    #     dict(type='GetBBoxCenterScale'),
    #     dict(type='TopdownAffine', input_size=codec['input_size']),
    #     dict(type='PackPoseInputs')
    # ]
    # config from POSE_CONFIG
    pipeline = Compose(
        [
            LoadImage(),
            GetBBoxCenterScale(),
            TopdownAffine(input_size=input_size),
            PackPoseInputs(),
        ]
    )

    data_preprocessor = PoseDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,  # if isinstance(img, np.ndarray) then check if bgr_to_rgb is required
    )

    # data_info = dict(img=img)
    w = 260
    h = 492
    # bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
    # bboxes in format (N, 4) [x, y, w, h] (left, top, width, height)
    bboxes = np.array(
        [[200, 200, w - 200, h - 200], [50, 50, w - 100, h - 100]], dtype=np.float32
    )

    img = IMG_PATH

    data_list, preprocessed_image = prepare_data(
        img, bboxes, pipeline, data_preprocessor
    )

    infer_request = compiled_model.create_infer_request()

    input_tensor = ov.Tensor(array=preprocessed_image.numpy(), shared_memory=True)
    infer_request.set_input_tensor(input_tensor)

    infer_request.set_output_tensor(
        0,
        ov.Tensor(
            np.zeros((bboxes.shape[0], n_keypoints, output_size[0]), dtype=np.float32)
        ),
    )
    infer_request.set_output_tensor(
        1,
        ov.Tensor(
            np.zeros((bboxes.shape[0], n_keypoints, output_size[1]), dtype=np.float32)
        ),
    )

    infer_request.start_async()
    infer_request.wait()

    simcc_x = infer_request.get_output_tensor(0).data
    simcc_y = infer_request.get_output_tensor(1).data

    openvino_pred, openvino_scores = restore_keypoints(simcc_x, simcc_y, data_list)

    print(openvino_pred)
    print(openvino_scores)
