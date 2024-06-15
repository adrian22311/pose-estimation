import importlib
import os
import pickle
import random
import sys
import time

import cv2
import numpy as np
import psutil

WARMUP_RUNS = 3
DATA_ROOT = "/home/kaminskia/studies/s10/pose-estimation/sampled"
OUTPUT_ROOT = "/home/kaminskia/studies/s10/pose-estimation/out"

IMAGES_ROOT = os.path.join(DATA_ROOT, "sampled_images")
# MODEL_NM = os.environ.get("MODEL_NM", None)
MODELS_NM = [
    "rtm_body8_26keypoints_det-m_pose-m_256x192",
    "rtm_coco_det-m_pose-l",
    "rtm_body8_26keypoints_det-m_pose-m_384x288",
    "rtm_coco_det-nano_pose-m",
    "rtm_body8_det-m_pose-s",
    "rtm_body8_det-m_pose-m",
]

def timeit(func):
    def wrapper(*args, **kwargs):
        model_nm = os.environ.get("MODEL_NM")
        process = psutil.Process()
        cpu_start = process.cpu_percent()
        start = time.time()
        _score = func(*args, **kwargs)
        end = time.time()
        cpu_end = process.cpu_percent()
        with open(os.path.join(OUTPUT_ROOT, model_nm, "times_deployed.txt"), "a") as f:
            f.write(f"{end-start}\n")
        with open(os.path.join(OUTPUT_ROOT, model_nm, "cpu_deployed.txt"), "a") as f:
            f.write(f"{cpu_end-cpu_start}\n")
        # print(f"Time taken: {end-start}")
        # print(f"Cpu usage: {cpu_end-cpu_start}")
        return _score

    return wrapper


@timeit
def inference_it(img):
    return inference(img)

def load_img(filename: str) -> np.ndarray:

    with open(filename, 'rb') as f:
        value = f.read()
    img_np = np.frombuffer(value, np.uint8)
    flag = cv2.IMREAD_COLOR
    img = cv2.imdecode(img_np, flag)
    return img

if __name__ == "__main__":
    images = [
        os.path.join(IMAGES_ROOT, _file_nm) for _file_nm in os.listdir(IMAGES_ROOT)
    ]

    for model_nm in MODELS_NM:
        os.environ["MODEL_NM"] = model_nm
        from models.rtmpose.lib_deployed import inference
        importlib.reload(sys.modules['models.rtmpose.lib_deployed'])
        from models.rtmpose.lib_deployed import inference

        for i in range(WARMUP_RUNS):
            _ = inference(load_img(random.choice(images)))

        scores = {}
        for filename in images:
            file_id = os.path.basename(filename).split(".")[0]
            img = load_img(filename)
            scores[file_id] = inference_it(img)

        with open(os.path.join(OUTPUT_ROOT, model_nm, "scores_deployed.pkl"), "wb") as f:
            pickle.dump(scores, f)

# Loading example
# import pickle
# with open("out/rtm_coco_det-m_pose-l/scores.pkl", "rb") as f:
#     x = pickle.load(f)
# x -> dict[key -> file id, value -> inference function output]
# x -> dict[key -> file id, value -> inference function output]
