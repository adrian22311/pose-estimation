import numpy as np

def to_key_points(query_locations: np.array, **kwargs) -> tuple[list[tuple[float, float]], list[tuple[tuple[int, int], int]]]:
    """
    Convert the 2D key points to edges
    :param query_locations: tuple: first element is the scores of the keypoints and the second element is the coordinates of the keypoints (x, y coordinates)
    :return: tuple: the list of keypoints and the configuration of the edges - 2D coordinates of the key points in the format of (x, y) and the configuration of the edges
    """

    config = [
        ([0, 1], -1), # nose -> leftEye
        ([1, 3], -1), # leftEye -> leftEar
        ([0, 2], 1), # nose -> rightEye
        ([2, 4], 1), # rightEye -> rightEar
        ([0, 5], 0), # nose -> leftShoulder
        ([5, 7], -1), # leftShoulder -> leftElbow
        ([7, 9], -1), # leftElbow -> leftWrist
        ([5, 11], 0), # leftShoulder -> leftHip
        ([11, 13], -1), # leftHip -> leftKnee
        ([13, 15], -1), # leftKnee -> leftAnkle
        ([0, 6], 0), # nose -> rightShoulder
        ([6, 8], 1), # rightShoulder -> rightElbow
        ([8, 10], 1), # rightElbow -> rightWrist
        ([6, 12], 0), # rightShoulder -> rightHip
        ([11, 13], 1), # rightHip -> rightKnee
        ([13, 15], 1) # rightKnee -> rightAnkle
    ]
    scores, coordinates =  query_locations
    selected_keypoints = np.where(scores > 0.25, np.arange(17), -1)
    keypoints = []
    for idx in selected_keypoints:
        if idx == -1:
            continue
        keypoints.append((coordinates[idx, 0], coordinates[idx, 1]))
    return keypoints, config



def to_pckh(query_locations: np.array, **kwargs) -> list[int]:
    """
    Convert the 2D key points to proper PCKh format
    :param query_locations: tuple: first element is the scores of the keypoints and the second element is the coordinates of the keypoints (x, y coordinates)
    :return: list: the list of keypoints (length 17 * 3) - 2D coordinates of the key points in the format of (x, y, s) where x and y are the coordinates of the key points and s indicates the presence of the key point
    """
    pckh = [0] * 51
    scores, coordinates =  query_locations
    selected_keypoints = np.where(scores > 0.25, np.arange(17), -1)
    for idx in selected_keypoints:
        if idx == -1:
            continue
        pckh[idx * 3] = round(coordinates[idx, 0])
        pckh[idx * 3 + 1] = round(coordinates[idx, 1])
        pckh[idx * 3 + 2] = 1
    return pckh