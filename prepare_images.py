import json
import os
import pickle
import logging
import cv2
import numpy as np
from draw_pose import get_pose, draw_pose
from typing import Any


import models.efficient_pose_ii.utils as eff_utils
import models.hourglass.utils as hourglass_utils
import models.lightweight_human_pose_estimation.utils as lwhpe_utils
import models.posenet_50.utils as posenet_utils
import models.rtmpose.utils as rtm_utils
import models.trans_pose_tph_a4_256x192.utils as tp_utils

OUTPUT_ROOT = "wycinki_output"
DATA_OUTPUT_ROOT = "out"
DATA_ROOT = "wycinki"
LOGGER = logging.getLogger(__name__)

MODEL_UTILS_MAPPING = {
    "rtm_body8_26keypoints_det-m_pose-m_256x192": (rtm_utils, {"threshold": 0.3}),
    "rtm_coco_det-m_pose-l": (rtm_utils, {"threshold": 0.3}),
    "rtm_body8_26keypoints_det-m_pose-m_384x288": (rtm_utils, {"threshold": 0.3}),
    "rtm_coco_det-nano_pose-m": (rtm_utils, {"threshold": 0.3}),
    "rtm_body8_det-m_pose-s": (rtm_utils, {"threshold": 0.3}),
    "rtm_body8_det-m_pose-m": (rtm_utils, {"threshold": 0.3}),
    "lightweight_human_pose_estimation": (lwhpe_utils, {}),
    "trans_pose_tpr_a4_256x192": (tp_utils, {}),
    "hourglass": (hourglass_utils, {}),
    "posenet_50": (posenet_utils, {}),
    "efficient_pose_rt_lite": (eff_utils, {}),
    "posenet_101": (posenet_utils, {}),
    "trans_pose_tph_a4_256x192": (tp_utils, {}),
    "soft_gated_pose_estimation": (hourglass_utils, {}),
    "efficient_pose_iii": (eff_utils, {}),
    "efficient_pose_iv": (eff_utils, {}),
    "efficient_pose_ii": (eff_utils, {}),
}

def load_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as file:
        return pickle.load(file)
    
def load_json(file_path: str) -> Any:
    with open(file_path, "r") as file:
        return json.load(file)
    
def load_image(file_path: str) -> np.ndarray:
    return cv2.imread(file_path)

def map_keypoints(keypoints: list[tuple[float, float]], position: dict[str, int]) -> list[tuple[float, float]]:
    x_min, y_min, _, _ = position["xmin"], position["ymin"], position["xmax"], position["ymax"]
    return [
        (
            x_min + x if x is not None else None,
            y_min + y if y is not None else None
        )
        for x, y in keypoints
    ]


def processing_camera(camera_path: str):
    camera_dir= os.path.basename(camera_path)
    position_of_objects = load_json(os.path.join(camera_path, "objects_pos.json"))
    LOGGER.debug(f"{camera_dir=}")

    for model_name in MODEL_UTILS_MAPPING:
        LOGGER.info(f"Processing {model_name}")
        predictions = load_pickle(os.path.join(DATA_OUTPUT_ROOT, model_name, "prediction.pkl"))
        camera_img = load_image(os.path.join(camera_path, "image.jpg"))
        output_dir = os.path.join(OUTPUT_ROOT, model_name, camera_dir)
        LOGGER.debug(f"{output_dir=}")
        os.makedirs(output_dir, exist_ok=True)

        for object_name in filter(lambda x: x.startswith("object") & x.endswith("jpg"), os.listdir(camera_path)):
            LOGGER.debug(f"Processing {object_name}")
            filepath = f"{camera_path}/{object_name}"

            LOGGER.debug(f"{filepath=}")
            prediction = predictions[f"app/{filepath}"]
            utils, _ = MODEL_UTILS_MAPPING[model_name]
            height, width = cv2.imread(filepath).shape[:2]
            keypoints, config = utils.to_key_points(prediction, image_width=width, image_height=height)
            image = get_pose(filepath, keypoints, config, line_width=2)
            cv2.imwrite(os.path.join(output_dir, object_name), image)

            LOGGER.debug(f"Processed {object_name.removesuffix('.jpg')}")
            new_keypoints = map_keypoints(keypoints, position_of_objects[object_name.removesuffix(".jpg")])
            camera_img = get_pose(camera_img, new_keypoints, config, line_width=2)
        
        cv2.imwrite(os.path.join(output_dir, "image.jpg"), camera_img)
        LOGGER.info(f"Processed {model_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for _camera in os.listdir(DATA_ROOT):
        LOGGER.info(f"Processing {_camera}")
        processing_camera(f"{DATA_ROOT}/{_camera}")

    LOGGER.info("Done")
