try:
    from lib import inference
except ImportError:
    from lib_ import inference

import os
import pickle
import random

WARMUP_RUNS = 1
DATA_ROOT = "/app"
OUTPUT_ROOT = "/app/out"

IMAGES_ROOT = os.path.join(DATA_ROOT, "wycinki")
MODEL_NM = os.environ.get("MODEL_NM", "None")

assert MODEL_NM, "MODEL_NM environment variable not provided"

os.makedirs(os.path.join(OUTPUT_ROOT, MODEL_NM), exist_ok=True)

def inference_it(filename):
    return inference(filename)


if __name__ == "__main__":
    images = [
        os.path.join(IMAGES_ROOT, _dir_nm, _file_nm) for _dir_nm in os.listdir(IMAGES_ROOT)
            for _file_nm in os.listdir(os.path.join(IMAGES_ROOT, _dir_nm))
            if _file_nm.endswith(".jpg") and _file_nm.startswith("object")
    ]

    for i in range(WARMUP_RUNS):
        _ = inference(random.choice(images))

    scores = {}
    for filename in images:
        file_id = "/".join(filename.split("/")[1:])
        scores[file_id] = inference_it(filename)

    with open(os.path.join(OUTPUT_ROOT, MODEL_NM, "prediction.pkl"), "wb") as f:
        pickle.dump(scores, f)

# Loading example
# import pickle
# with open("out/rtm_coco_det-m_pose-l/scores.pkl", "rb") as f:
#     x = pickle.load(f)
# x -> dict[key -> file id, value -> inference function output]
