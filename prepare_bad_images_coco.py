import json
import os
import pickle
import logging
from typing import Any
from draw_pose import get_pose


import cv2
import models.efficient_pose_ii.utils as eff_utils
import models.hourglass.utils as hourglass_utils
import models.lightweight_human_pose_estimation.utils as lwhpe_utils
import models.posenet_50.utils as posenet_utils
import models.rtmpose.utils as rtm_utils
import models.trans_pose_tph_a4_256x192.utils as tp_utils
from pckh_scoring import calculate_pckh


OUTPUT_ROOT = "out"
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


def list_models() -> list[str]:
    return [
        nm
        for nm in os.listdir(OUTPUT_ROOT)
        if os.path.isdir(os.path.join(OUTPUT_ROOT, nm))
    ]



def get_predictions(model: str, deployed: bool = False) -> dict[str, Any]:
    filename = "scores.pkl" if not deployed else "scores_deployed.pkl"
    utils, _ = MODEL_UTILS_MAPPING[model]
    if utils is None:
        raise NotImplementedError(f"Model {model} not implemented")
    with open(os.path.join(OUTPUT_ROOT, model, filename), "rb") as f:
        predictions = pickle.load(f)
    return predictions


def calculate_score(
    gt_location: list[float],
    query_location: Any,
    model,
    image_id: str,
    bbox: list[float],
    strict: bool = False,
) -> float:
    utils_mod, kwargs = MODEL_UTILS_MAPPING[model]
    return calculate_pckh(
        gt_location,
        utils_mod.to_pckh(query_location, **kwargs, image_id=image_id),
        bbox=bbox,
        strict=strict,
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    models = list_models()

    with open("sampled/samples_info.json") as f:
        samples_info = json.load(f)
    results_score = []
    results_score_deployed = []
    for model in models:
        LOGGER.info(f"Processing {model}")
        os.makedirs(f"bad_images/{model}", exist_ok=True)
        utils, _ = MODEL_UTILS_MAPPING[model]
        predictions = get_predictions(model)
        for image_nm, query_location in predictions.items():
            image_id = image_nm.split(".")[0]
            gt = samples_info[image_id]["keypoints"]
            bbox = samples_info[image_id]["bbox"]
            num_correct_keypoints_strict, num_valid_keypoints_strict = calculate_score(gt, query_location, model, image_id, bbox, strict=True)
            if num_valid_keypoints_strict == 0:
                continue
            if num_correct_keypoints_strict / num_valid_keypoints_strict < 0.5:
                logging.info(f"Bad image: {image_id}")
                image = cv2.imread(f"sampled/sampled_images/{image_id}.jpg")
                height, width, _ = image.shape
                keypoints, config = utils.to_key_points(query_location, image_width=width, image_height=height)
                image = get_pose(image, keypoints, config, line_width=1)
                cv2.imwrite(f"bad_images/{model}/{image_id}.jpg", image)
    LOGGER.info("Done")
