import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

import models.efficient_pose_ii.utils as eff_utils
import models.hourglass.utils as hourglass_utils
import models.lightweight_human_pose_estimation.utils as lwhpe_utils
import models.posenet_50.utils as posenet_utils
import models.rtmpose.utils as rtm_utils
import models.trans_pose_tph_a4_256x192.utils as tp_utils
from pckh_scoring import calculate_pckh

OUTPUT_ROOT = "out"

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


def get_times(model: str, deployed: bool = False) -> np.ndarray:
    filename = "times.txt" if not deployed else "times_deployed.txt"
    with open(os.path.join(OUTPUT_ROOT, model, filename), "r") as f:
        return np.array([float(line) * 1000 for line in f.readlines()])


def get_cpu(model: str, deployed: bool = False) -> np.ndarray:
    filename = "cpu.txt" if not deployed else "cpu_deployed.txt"
    with open(os.path.join(OUTPUT_ROOT, model, filename), "r") as f:
        return np.array([float(line) for line in f.readlines()])


def calculate_statistics(times: np.ndarray) -> dict[str, Any]:
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "P95": np.percentile(times, 95),
        "P98": np.percentile(times, 98),
        "P99": np.percentile(times, 99),
    }


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

def calc_scores(model: str, samples_info: dict[str, Any], deployed: bool = False):
    total_correct_keypoints = 0
    total_strict_correct_keypoints = 0
    total_valid_keypoints = 0
    total_strict_valid_keypoints = 0
    predictions = get_predictions(model, deployed=deployed)

    for image_nm, query_location in predictions.items():
        image_id = image_nm.split(".")[0]
        gt = samples_info[image_id]["keypoints"]
        bbox = samples_info[image_id]["bbox"]
        # pckh_nonstrict = calculate_score(gt, query_location, model, image_id, bbox)
        num_correct_keypoints, num_valid_keypoints = calculate_score(gt, query_location, model, image_id, bbox)
        num_correct_keypoints_strict, num_valid_keypoints_strict = calculate_score(gt, query_location, model, image_id, bbox, strict=True)
        # pckh_strict = calculate_score(gt, query_location, model, image_id, bbox, strict=True)
        total_correct_keypoints += num_correct_keypoints
        total_valid_keypoints += num_valid_keypoints
        total_strict_correct_keypoints += num_correct_keypoints_strict
        total_strict_valid_keypoints += num_valid_keypoints_strict
    return (total_correct_keypoints,
        total_strict_correct_keypoints,
        total_valid_keypoints,
        total_strict_valid_keypoints
    )


if __name__ == "__main__":
    models = list_models()

    results = []
    results_cpu = []
    results_deployed = []
    results_cpu_deployed = []
    models_deployed = []
    for model in models:
        times = get_times(model)
        stats = calculate_statistics(times)
        results.append(stats)
        cpu = get_cpu(model)
        stats_cpu = calculate_statistics(cpu)
        results_cpu.append(stats_cpu)
        try:
            times = get_times(model, deployed=True)
            stats = calculate_statistics(times)
            results_deployed.append(stats)
            cpu = get_cpu(model, deployed=True)
            stats_cpu = calculate_statistics(cpu)
            results_cpu_deployed.append(stats_cpu)
            models_deployed.append(model)
        except FileNotFoundError:
            continue

    results_pd = pd.DataFrame(results, index=models)
    results_cpu_pd = pd.DataFrame(results_cpu, index=models)
    results_pd_deployed = pd.DataFrame(results_deployed, index=models_deployed)
    results_cpu_pd_deployed = pd.DataFrame(results_cpu_deployed, index=models_deployed)

    results_pd.to_csv(os.path.join(OUTPUT_ROOT, "results_time.csv"), index=True)
    results_cpu_pd.to_csv(os.path.join(OUTPUT_ROOT, "results_cpu.csv"), index=True)
    results_pd_deployed.to_csv(os.path.join(OUTPUT_ROOT, "results_time_deployed.csv"), index=True)
    results_cpu_pd_deployed.to_csv(os.path.join(OUTPUT_ROOT, "results_cpu_deployed.csv"), index=True)

    with open("sampled/samples_info.json") as f:
        samples_info = json.load(f)
    results_score = []
    results_score_deployed = []
    for model in models:
        try:
            total_correct_keypoints, total_strict_correct_keypoints, total_valid_keypoints, total_strict_valid_keypoints = calc_scores(
                model, samples_info, deployed=False
            )
        except:
            print(f"Error in model {model}")
            continue
        results_score.append(
            {
                "model": model,
                "pckh": total_correct_keypoints / total_valid_keypoints if total_valid_keypoints > 0 else 0,
                "pckh_strict": total_strict_correct_keypoints / total_strict_valid_keypoints if total_strict_valid_keypoints > 0 else 0
            }
        )
        try:
            total_correct_keypoints, total_strict_correct_keypoints, total_valid_keypoints, total_strict_valid_keypoints = calc_scores(
                model, samples_info, deployed=True
            )
        except Exception as e:
            print(f"Error in model {model}. Model might not be deployed.")
            print(str(e))
            continue
        results_score_deployed.append(
            {
                "model": model,
                "pckh": total_correct_keypoints / total_valid_keypoints if total_valid_keypoints > 0 else 0,
                "pckh_strict": total_strict_correct_keypoints / total_strict_valid_keypoints if total_strict_valid_keypoints > 0 else 0
            }
        )
    results_score_pd = pd.DataFrame(results_score)
    results_score_pd.to_csv(os.path.join(OUTPUT_ROOT, "results_score.csv"), index=False)
    results_score_pd_deployed = pd.DataFrame(results_score_deployed)
    results_score_pd_deployed.to_csv(os.path.join(OUTPUT_ROOT, "results_score_deployed.csv"), index=False)
