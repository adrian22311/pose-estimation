import json

import numpy as np

with open("sampled/samples_info.json") as f:
    SAMPLES_INFO = json.load(f)


def get_image_width_height(image_id: str) -> tuple[int, int]:
    """
    Get the width and height of the image
    :param image_id: str: the image id
    :return: tuple: the width and height of the image
    """
    width, height = SAMPLES_INFO[image_id]["bbox"][2], SAMPLES_INFO[image_id]["bbox"][3]
    return int(width) + 1, int(height) + 1


def to_key_points(query_locations, **kwargs) -> tuple[list[tuple[float, float]], list[tuple[tuple[int, int], int]]]:
    """
    Convert the 2D key points to edges
    :param query_locations: list: coordinates of the keypoints (name of the keypoint, x, y coordinates) x and y coordinates are percentages of the image width and height where the keypoint is located
    :return: tuple: the list of keypoints and the configuration of the edges - 2D coordinates of the key points in the format of (x, y) and the configuration of the edges
    """
    if "image_id" in kwargs:
        image_width, image_height = get_image_width_height(kwargs["image_id"])

    query_locations = query_locations[0]

    keypoints: list[tuple[float, float]] = [(x * image_width, y * image_height) for key, x, y in query_locations]  
    config: list[tuple[tuple[float, float], float]] = [
        ((0, 1), 0), # head_top -> upper_neck 
        ((1, 5), 0), # upper_neck -> thorax
        ((5, 2), 1),  # thorax -> r_shoulder
        ((5, 6), -1),  # thorax -> l_shoulder
        ((5, 9), 0),  # thorax -> pelvis
        ((2, 3), 1),  # r_shoulder -> r_elbow
        ((3, 4), 1),  # r_elbow -> r_wrist
        ((6, 7), -1),  # l_shoulder -> l_elbow
        ((7, 8), -1),  # l_elbow -> l_wrist
        ((9, 10), 1),  # pelvis -> r_hip
        ((9, 13), -1),  # pelvis -> l_hip
        ((10, 11), 1),  # r_hip -> r_knee
        ((11, 12), 1), # r_knee -> r_ankle
        ((13, 14), -1), # l_hip -> l_knee
        ((14, 15), -1) # l_knee -> l_ankle
    ]
    
    return keypoints, config


def to_pckh(query_locations, **kwargs) -> list[int]:
    """
    Convert the 2D key points to proper PCKh format
    :param query_locations: list: coordinates of the keypoints (name of the keypoint, x, y coordinates) x and y coordinates are percentages of the image width and height where the keypoint is located
    :return: list: the list of keypoints (length 17 * 3) - 2D coordinates of the key points in the format of (x, y, s) where x and y are the coordinates of the key points and s indicates the presence of the key point
    """
    if "image_id" in kwargs:
        image_width, image_height = get_image_width_height(kwargs["image_id"])

    query_locations = query_locations[0]

    pckh = [0] * 51
    config = {
        # 'head_top': 0,
        # 'upper_neck': 0,
        # 'thorax': 0,
        'right_shoulder': 6,
        'right_elbow': 8,
        'right_wrist': 10,
        'left_shoulder': 5,
        'left_elbow': 7,
        'left_wrist': 9,
        # 'pelvis': 0,
        'right_hip': 12,
        'right_knee': 14,
        'right_ankle': 16,
        'left_hip': 11,
        'left_knee': 13,
        'left_ankle': 15
    }

    for key, x, y in query_locations:
        if key in config:
            pckh[config[key] * 3] = x * image_width
            pckh[config[key] * 3 + 1] = y * image_height
            pckh[config[key] * 3 + 2] = 1

    potential_points = set(config.values())
    for i in range(17):
        if i not in potential_points:
            pckh[i * 3 + 2] = 2

    return pckh