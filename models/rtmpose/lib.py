# This code is based on https://github.com/open-mmlab/mmpose/blob/main/demo/topdown_demo_with_mmdet.py
# The original code is licensed under the Apache License 2.0

import os
from collections import namedtuple

import mmcv
import numpy as np
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples, split_instances

assert os.getenv("POSE_CONFIG") is not None, "POSE_CONFIG environment variable is not set."
assert os.getenv("POSE_CHECKPOINT") is not None, "POSE_CHECKPOINT environment variable is not set."


Config = namedtuple(
    "Config",
    [
        "pose_config",
        "pose_checkpoint",
        "show",
        # "output_root",
        # "save_predictions",
        "device",
        # "det_cat_id",
        # "bbox_thr",
        # "nms_thr",
        "kpt_thr",
        "draw_heatmap",
        "show_kpt_idx",
        "skeleton_style",
        # "radius",
        # "thickness",
        "show_interval",
        # "alpha",
        "draw_bbox",
    ],
)

args = Config(
    # Config file for pose
    pose_config=os.getenv("POSE_CONFIG"),
    # Checkpoint file for pose
    pose_checkpoint=os.getenv("POSE_CHECKPOINT"),
    # input_file='' # Image/Video file
    show=False,  # whether to show img
    # output_root="",  # root of the output img file.
    # save_predictions=True,  # whether to save predicted results
    device="cpu",  # Device used for inference
    # det_cat_id=0,  # Category id for bounding box detection model
    # bbox_thr=0.3,  # Bounding box score threshold
    # nms_thr=0.3,  # IoU threshold for bounding box NMS
    kpt_thr=0.3,  # Visualizing keypoint thresholds
    draw_heatmap=False,  # Draw heatmap predicted by the model
    show_kpt_idx=False,  # Whether to show the index of keypoints
    skeleton_style="mmpose",  # Skeleton style selection
    # radius=3,  # Keypoint radius for visualization
    # thickness=1,  # Link thickness for visualization
    show_interval=0,  # Sleep seconds per frame
    # alpha=0.8,  # The transparency of bboxes
    draw_bbox=False,  # Draw bboxes of instances
)


def process_one_image(
    args,
    img,
    #   detector,
    pose_estimator,
    visualizer=None,
    show_interval=0,
):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes=None)
    data_samples = merge_data_samples(pose_results)

    # show the results
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

    # if there is no instance detected, return None
    return data_samples.get("pred_instances", None)


# build pose estimator
pose_estimator = init_pose_estimator(
    args.pose_config,
    args.pose_checkpoint,
    device=args.device,
    cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))),
)


def inference(filename: str):
    """
    Inference function for a single image.
    Prediction for each point is always returned - threshold should be selected to filter out low confidence points.

    Parameters
    ----------
    filename : str
        The path to the image file.

    Returns
    -------
    pred_keypoints : dict
        A dictionary containing the predicted keypoints.
        The dictionary has the following keys:
        - "keypoints": list[list[float]] A list of keypoints -- [[x1,y1], ..., [x17,y17]] or [[x1,y1], ..., [x26,y26]].
        - "keypoint_scores": list[float] A list of scores for each keypoint -- [score1, ..., score17] or [score1, ..., score26].
        - "bbox": tuple[list[float]] A one-element tuple of bounding boxes -- ([x,y,w,h],) - in this case [0, 0, <image-width>, <image-height>].
        - "bbox_score": (float) A score for bounding box -- score.
    """
    pred_instances = process_one_image(args, filename, pose_estimator, None)
    pred_instances_list = split_instances(pred_instances)

    return pred_instances_list[0]


if __name__ == "__main__":
    if os.path.exists("/app/data/sampled_images/17905_2157397.jpg"):
        pred = inference("/app/data/sampled_images/17905_2157397.jpg")
        print(f"Keypoints: {pred['keypoints']}")
        print(f"Keypoints num: {len(pred['keypoints'])}")
        print(f"Keypoints scores: {pred['keypoint_scores']}")
        print(f"Bbox: {pred['bbox']}")
        print(f"Bbox scores: {pred['bbox_score']}")
        print()
    if os.path.exists("/app/data/sampled_images/559160_546425.jpg"):
        pred = inference("/app/data/sampled_images/559160_546425.jpg")
        print(f"Keypoints: {pred['keypoints']}")
        print(f"Keypoints num: {len(pred['keypoints'])}")
        print(f"Keypoints scores: {pred['keypoint_scores']}")
        print(f"Bbox: {pred['bbox']}")
        print(f"Bbox scores: {pred['bbox_score']}")
