import numpy as np

def to_key_points(query_locations: np.array, **kwargs) -> tuple[list[tuple[float, float]], list[tuple[tuple[int, int], int]]]:
    """
    Convert the 2D key points to edges
    :param query_locations: np.array: coordinates of the keypoints (x, y coordinates) Shape: (16, 2)
    :return: tuple: the list of keypoints and the configuration of the edges - 2D coordinates of the key points in the format of (x, y) and the configuration of the edges
    """

    keypoints: list[tuple[float, float]] = [(x[0], x[1]) for x in query_locations]
    config: list[tuple[tuple[int, int], int]] = [
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


def to_pckh(query_locations: np.array, **kwargs) -> list[int]:
    """
    Convert the 2D key points to proper PCKh format
    :param query_locations: list: coordinates of the keypoints (name of the keypoint, x, y coordinates) x and y coordinates are percentages of the image width and height where the keypoint is located
    :return: list: the list of keypoints (length 17 * 3) - 2D coordinates of the key points in the format of (x, y, s) where x and y are the coordinates of the key points and s indicates the presence of the key point
    """
    pckh = [0] * 51
    config = {
        # 'head_top': 0,
        # 'upper_neck': 0,
        # 'thorax': 0,
        3: 6,
        4: 8,
        5: 10,
        6: 5,
        7: 7,
        8: 9,
        # 'pelvis': 0,
        10: 12,
        11: 14,
        12: 16,
        13: 11,
        14: 13,
        15: 15
    }

    for i, keypoint in enumerate(query_locations):
        if i in config:
            pckh[config[i] * 3] = keypoint[0]
            pckh[config[i] * 3 + 1] = keypoint[1]
            pckh[config[i] * 3 + 2] = 1
    return pckh