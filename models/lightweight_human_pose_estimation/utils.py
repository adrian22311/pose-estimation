import numpy as np

def to_key_points(query_locations: np.array, **kwargs) -> tuple[list[tuple[float, float]], list[tuple[tuple[int, int], int]]]:
    """
    Convert the 2D key points to edges
    :param query_locations: np.array: coordinates of the keypoints 
    :return: tuple: the list of keypoints and the configuration of the edges - 2D coordinates of the key points in the format of (x, y) and the configuration of the edges
    """
    query_locations = query_locations[:-1]
    keypoints: list[tuple[float, float]] = []

    for i in range(len(query_locations) // 3):
        x, y, score = query_locations[3 * i], query_locations[3 * i + 1], query_locations[3 * i + 2]
        if score != -1:
            keypoints.append((x, y))
        else:
            keypoints.append((None, None))
    
    config: list[tuple[tuple[int, int], int]] = [
        ((0, 1), 0), # neck - nose
        ((1, 16), -1), ((16, 18), 1), # nose - l_eye - l_ear
        ((1, 15), 1), ((15, 17), 1), # nose - r_eye - r_ear
        ((0, 3), 0), ((3, 4), -1), ((4, 5), -1), # neck - l_shoulder - l_elbow - l_wrist
        ((0, 9), 0), ((9, 10), 1), ((10, 11), 1), # neck - r_shoulder - r_elbow - r_wrist
        ((3, 6), 0), ((6, 7), -1), ((7, 8), -1), # l_shoulder - l_hip - l_knee - l_ankle
        ((9, 12), 0), ((12, 13), 1), ((13, 14), 1), # r_shoulder - r_hip - r_knee - r_ankle
        ((6, 2), 0), ((2, 12), 0), # l_hip - pelvis - r_hip
        ((1, 2), 0) # neck - pelvis
    ]  
    return keypoints, config



def to_pckh(query_locations: np.array, **kwargs) -> list[int]:
    """
    Convert the 2D key points to proper PCKh format
    :param query_locations: tuple: first element is the scores of the keypoints and the second element is the coordinates of the keypoints (x, y coordinates)
    :return: list: the list of keypoints (length 17 * 3) - 2D coordinates of the key points in the format of (x, y, s) where x and y are the coordinates of the key points and s indicates the presence of the key point
    """
    pckh = [0] * 51

    mapping = {
        0: -1,
        1: 0,
        2: -1,
        16: 1,
        15: 2,
        18: 3,
        17: 4,
        3: 5,
        9: 6,
        4: 7,
        10: 8,
        5: 9,
        11: 10,
        6: 11,
        12: 12,
        7: 13,
        13: 14,
        8: 15,
        14: 16
    }

    new_query_locations = query_locations[:-1]
    for i in range(len(new_query_locations) // 3):
        mapping_idx = mapping[i]
        if mapping_idx != -1:
            if new_query_locations[3 * i + 2] == -1:
                continue
            pckh[mapping_idx * 3] = round(new_query_locations[3 * i])
            pckh[mapping_idx * 3 + 1] = round(new_query_locations[3 * i + 1])
            pckh[mapping_idx * 3 + 2] = 1
    return pckh