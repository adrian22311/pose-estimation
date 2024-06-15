### Setup model

import os
from collections import namedtuple

import mmcv
import numpy as np
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples

assert os.getenv("POSE_CONFIG") is not None, "POSE_CONFIG environment variable is not set."
assert os.getenv("POSE_CHECKPOINT") is not None, "POSE_CHECKPOINT environment variable is not set."
assert os.getenv("MODEL_NM") is not None, "MODEL_NM environment variable is not set."

Config = namedtuple(
    "Config",
    [
        "pose_config",
        "pose_checkpoint",
        "show",
        "device",
        "kpt_thr",
        "draw_heatmap",
        "show_kpt_idx",
        "skeleton_style",
        "show_interval",
        "draw_bbox",
    ],
)

args = Config(
    pose_config=os.getenv("POSE_CONFIG"),
    pose_checkpoint=os.getenv("POSE_CHECKPOINT"),
    show=False,  # whether to show img
    device="cpu",  # Device used for inference
    kpt_thr=0.3,  # Visualizing keypoint thresholds
    draw_heatmap=False,  # Draw heatmap predicted by the model
    show_kpt_idx=False,  # Whether to show the index of keypoints
    skeleton_style="mmpose",  # Skeleton style selection
    show_interval=0,  # Sleep seconds per frame
    draw_bbox=False,  # Draw bboxes of instances
)


def process_one_image(
    args,
    img,
    pose_estimator,
    visualizer=None,
    show_interval=0,
):
    """Visualize predicted keypoints (and heatmaps) of one image."""
    pose_results = inference_topdown(pose_estimator, img, bboxes=None)
    data_samples = merge_data_samples(pose_results)
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order="rgb")
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)
    if visualizer is not None:
        visualizer.add_datasample(
            "result",
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr,
        )
    return data_samples.get("pred_instances", None)

pose_estimator = init_pose_estimator(
    args.pose_config,
    args.pose_checkpoint,
    device=args.device,
    cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))),
)

pose_estimator.eval()


### Convert model

import warnings

import openvino as ov
import torch

MODEL_DIRECTORY_PATH = f'/app/out/{os.getenv("MODEL_NM")}'
ONNX_CV_MODEL_PATH = os.path.join(MODEL_DIRECTORY_PATH,f"{os.getenv('MODEL_NM')}.onnx")

if os.path.exists(ONNX_CV_MODEL_PATH):
    print(f"ONNX model {ONNX_CV_MODEL_PATH} already exists.")
else:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        w, h = (288, 384) if "384x288" in os.getenv("MODEL_NM") else (192, 256)
        args = (torch.randn(1, 3, h, w), None, "tensor") # predict
        dynamic_axes = {"image": {0: "batch_size"}}
        torch.onnx.export(
            model=pose_estimator, args=args, f=ONNX_CV_MODEL_PATH, input_names = ["image"], dynamic_axes=dynamic_axes
        )
    print(f"ONNX model exported to {ONNX_CV_MODEL_PATH}")

ov_model = ov.convert_model(ONNX_CV_MODEL_PATH)
# then model can be serialized to *.xml & *.bin files
ov.save_model(ov_model, os.path.join(MODEL_DIRECTORY_PATH,f"{os.getenv('MODEL_NM')}.xml"))
