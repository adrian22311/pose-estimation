from itertools import chain

THRESHOLD = 0.3


def _to_key_points(
    query_locations, **kwargs
) -> tuple[list[tuple[float, float]], list[tuple[tuple[int, int], int]]]:
    """
    Convert query locations to key points.
    """
    if (threshold := kwargs.get("threshold", None)) is None:
        threshold = THRESHOLD

    keypoints = query_locations["keypoints"]  # works for both 17 and 26 keypoints
    scores = query_locations["keypoint_scores"]  # works for both 17 and 26 keypoints
    key_points = [
        (loc[0], loc[1]) if score >= threshold else (None, None)
        for loc, score in zip(keypoints, scores)
    ]
    if len(keypoints) == 17:
        config = [
            ((15, 13), -1),  # left_ankle -> left_knee
            ((13, 11), -1),  # left_knee -> left_hip
            ((16, 14), 1),  # right_ankle -> right_knee
            ((14, 12), 1),  # right_knee -> right_hip
            ((11, 12), 0),  # left_hip -> right_hip
            ((5, 11), 0),  # left_shoulder -> left_hip
            ((6, 12), 0),  # right_shoulder -> right_hip
            ((5, 6), 0),  # left_shoulder -> right_shoulder
            ((5, 7), -1),  # left_shoulder -> left_elbow
            ((6, 8), 1),  # right_shoulder -> right_elbow
            ((7, 9), -1),  # left_elbow -> left_wrist
            ((8, 10), 1),  # right_elbow -> right_wrist
            # ((1, 2), 0),  # left_eye -> right_eye
            ((0, 1), -1),  # nose -> left_eye
            ((0, 2), 1),  # nose -> right_eye
            ((1, 3), -1),  # left_eye -> left_ear
            ((2, 4), 1),  # right_eye -> right_ear
            ((3, 5), -1),  # left_ear -> left_shoulder
            ((4, 6), 1),  # right_ear -> right_shoulder
        ]
    else:  # 26 keypoints
        config = [
            ((15, 13), -1),  # left_ankle -> left_knee
            ((13, 11), -1),  # left_knee -> left_hip
            ((11, 19), 0),  # left_hip -> hip
            ((16, 14), 1),  # right_ankle -> right_knee
            ((14, 12), 1),  # right_knee -> right_hip
            ((12, 19), 0),  # right_hip -> hip
            ((17, 18), 0),  # head -> neck
            ((18, 19), 0),  # neck -> hip
            ((18, 5), -1),  # neck -> left_shoulder
            ((5, 7), -1),  # left_shoulder -> left_elbow
            ((7, 9), -1),  # left_elbow -> left_wrist
            ((18, 6), 1),  # neck -> right_shoulder
            ((6, 8), 1),  # right_shoulder -> right_elbow
            ((8, 10), 1),  # right_elbow -> right_wrist
            # ((1, 2), 0),  # left_eye -> right_eye
            ((0, 1), -1),  # nose -> left_eye
            ((0, 2), 1),  # nose -> right_eye
            ((1, 3), -1),  # left_eye -> left_ear
            ((2, 4), 1),  # right_eye -> right_ear
            ((3, 5), -1),  # left_ear -> left_shoulder
            ((4, 6), 1),  # right_ear -> right_shoulder
            ((15, 20), -1),  # left_ankle -> left_big_toe
            ((15, 22), -1),  # left_ankle -> left_small_toe
            ((15, 24), -1),  # left_ankle -> left_heel
            ((16, 21), 1),  # right_ankle -> right_big_toe
            ((16, 23), 1),  # right_ankle -> right_small_toe
            ((16, 25), 1),  # right_ankle -> right_heel
        ]

    return key_points, config

def to_key_points(
    query_locations, **kwargs
) -> tuple[list[tuple[float, float]], list[tuple[tuple[int, int], int]]]:
    """
    Convert query locations to key points.
    """
    if len(query_locations["keypoints"].shape) == 2:
        return _to_key_points(query_locations, **kwargs)
    if len(query_locations["keypoints"].shape) == 3:
        out_keypoints = []

        for i in range(query_locations["keypoints"].shape[0]):
            tmp_keypoints = {}
            tmp_keypoints["keypoints"] = query_locations["keypoints"][i]
            tmp_keypoints["keypoint_scores"] = query_locations["keypoint_scores"][i]

            key_points, config = _to_key_points(tmp_keypoints, **kwargs)
            out_keypoints.append(key_points)

        return out_keypoints, config
    raise ValueError("Invalid query locations shape.")

def to_pckh(query_locations, **kwargs):
    """
    Convert query locations to PCKh format.
    """
    if (threshold := kwargs.get("threshold"), None) is not None:
        threshold = THRESHOLD

    keypoints = query_locations["keypoints"][:17]  # works for both 17 and 26 keypoints
    scores = query_locations["keypoint_scores"][
        :17
    ]  # works for both 17 and 26 keypoints

    out = [
        [x, y, 1 if score >= threshold else 0]
        for (x, y), score in zip(keypoints, scores)
    ]

    return list(chain.from_iterable(out))


if __name__ == "__main__":
    import pickle

    with open("../../out/rtm_coco_det-m_pose-l/scores.pkl", "rb") as f:
        x = pickle.load(f)

    print(x["559160_546425"].keys())
    print(x["559160_546425"]["keypoints"].__len__())
    print(x["559160_546425"]["keypoints"])
    print(x["559160_546425"]["keypoint_scores"])
    print(f"--" * 10)
    print(to_pckh(x["559160_546425"]))
    print(len(to_pckh(x["559160_546425"])))
