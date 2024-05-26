from itertools import chain


def to_key_points(
    query_locations,
) -> (list[tuple[float, float]], list[tuple[tuple[int, int], int]]):
    """
    Convert the model output query_locations to key points and edges
    :param query_locations: ndarray: 17x2 array of the coordinates of the query locations
    :return: (list, list): the list of key points and the list of edges
    """
    key_points = [(loc[0], loc[1]) for loc in query_locations]
    config = [
        ((15, 13), -1),  # l_ankle -> l_knee
        ((13, 11), -1),  # l_knee -> l_hip
        ((11, 5), -1),  # l_hip -> l_shoulder
        ((12, 14), 1),  # r_hip -> r_knee
        ((14, 16), 1),  # r_knee -> r_ankle
        ((12, 6), 1),  # r_hip  -> r_shoulder
        ((3, 1), -1),  # l_ear -> l_eye
        ((1, 2), 0),  # l_eye -> r_eye
        ((1, 0), -1),  # l_eye -> nose
        ((0, 2), 1),  # nose -> r_eye
        ((2, 4), 1),  # r_eye -> r_ear
        ((9, 7), -1),  # l_wrist -> l_elbow
        ((7, 5), -1),  # l_elbow -> l_shoulder
        ((5, 6), 0),  # l_shoulder -> r_shoulder
        ((6, 8), 1),  # r_shoulder -> r_elbow
        ((8, 10), 1),  # r_elbow -> r_wrist
    ]
    return key_points, config


def to_pckh(query_locations, **kwargs):
    """
    Convert query locations to PCKh format.
    """

    out = [[x, y, 1] for x, y in query_locations]

    return list(chain.from_iterable(out))
