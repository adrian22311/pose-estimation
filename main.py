try:
    from lib import inference
except ImportError:
    from lib_ import inference

import os
import pickle
import random
import time

import psutil

WARMUP_RUNS = 3
DATA_ROOT = "/app/data"
OUTPUT_ROOT = "/app/out"

IMAGES_ROOT = os.path.join(DATA_ROOT, "sampled_images")
MODEL_NM = os.environ.get("MODEL_NM", "None")

assert MODEL_NM, "MODEL_NM environment variable not provided"

os.makedirs(os.path.join(OUTPUT_ROOT, MODEL_NM), exist_ok=True)


def timeit(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        cpu_start = process.cpu_percent()
        start = time.time()
        _score = func(*args, **kwargs)
        end = time.time()
        cpu_end = process.cpu_percent()
        with open(os.path.join(OUTPUT_ROOT, MODEL_NM, "times.txt"), "a") as f:
            f.write(f"{end-start}\n")
        with open(os.path.join(OUTPUT_ROOT, MODEL_NM, "cpu.txt"), "a") as f:
            f.write(f"{cpu_end-cpu_start}\n")
        # print(f"Time taken: {end-start}")
        # print(f"Cpu usage: {cpu_end-cpu_start}")
        return _score

    return wrapper


@timeit
def inference_it(filename):
    return inference(filename)


if __name__ == "__main__":
    images = [
        os.path.join(IMAGES_ROOT, _file_nm) for _file_nm in os.listdir(IMAGES_ROOT)
    ]

    for i in range(WARMUP_RUNS):
        _ = inference(random.choice(images))

    scores = {}
    for filename in images:
        file_id = os.path.basename(filename).split(".")[0]
        scores[file_id] = inference_it(filename)

    with open(os.path.join(OUTPUT_ROOT, MODEL_NM, "scores.pkl"), "wb") as f:
        pickle.dump(scores, f)
# Loading example
# import pickle
# with open("out/rtm_coco_det-m_pose-l/scores.pkl", "rb") as f:
#     x = pickle.load(f)
# x -> dict[key -> file id, value -> inference function output]
