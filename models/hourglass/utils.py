import numpy as np

def to_key_points(query_locations: np.array, **kwargs) -> tuple[list[tuple[float, float]], list[tuple[tuple[int, int], int]]]:
    """
    Convert the 2D key points to edges
    :param query_locations: np.array: coordinates of the keypoints (x, y coordinates) Shape: (16, 2)
    :return: tuple: the list of keypoints and the configuration of the edges - 2D coordinates of the key points in the format of (x, y) and the configuration of the edges
    """

    keypoints: list[tuple[float, float]] = [(x[0], x[1]) if x[0] >= 1 and x[1] >= 1 else (None, None) for x in query_locations]
    config: list[tuple[tuple[int, int], int]] = [
        ((0, 1), 1), # rank - rkne
        ((3, 4), -1), # lhip - lkne
        ((4, 5), -1), # lkne - lank
        ((8, 9), 0), # neck - head
        ((7, 8), 0), # thrx - neck
        ((7, 12), 0),  # thrx - rsho
        ((7, 13), 0), # thrx - lsho
        ((11, 12), 1), # relb - rsho
        ((13, 14), -1), # lsho - lelb
        ((10, 11), 1), # rwri - relb
        ((14, 15), -1), # lelb - lwri
        ((6, 7), 0), # pelv - thrx
        ((2, 6), 0), # rhip - pelv
        ((3, 6), 0), # lhip - pelv
        ((1, 2), 1), # rkne - rhip
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
        0: 16, # rank
        1: 14, # rkne
        2: 12, # rhip
        3: 9, # lhip
        4: 13, # lkne
        5: 15, # lank
        10: 10, # rwri
        11: 8, # relb
        12: 6, # rsho
        13: 5, # lsho
        14: 7, # lelb
        15: 9 # lwri
    }

    for i, keypoint in enumerate(query_locations):
        if i in config and keypoint[0] >= 1 and keypoint[1] >= 1:
            pckh[config[i] * 3] = keypoint[0]
            pckh[config[i] * 3 + 1] = keypoint[1]
            pckh[config[i] * 3 + 2] = 1
    return pckh