def to_key_points(query_locations) -> list[tuple[float, float, float, float, int]]:
    """
    Convert the 2D key points to edges
    :param query_locations: ndarray: 17x2 array of the coordinates of the key points
    :return: list: the list of edges - 2D coordinates of the key points in the format of (x0, y0, x1, y1, side) where side is -1 for left and 1 for right and 0 for center
    """
    key_points: list[tuple[float, float, float, float, int]] = []
    config = [
            ([15, 13], [255, 0, 0], -1),  # l_ankle -> l_knee
            ([13, 11], [155, 85, 0], -1),  # l_knee -> l_hip
            ([11, 5], [155, 85, 0], -1),  # l_hip -> l_shoulder
            ([12, 14], [0, 0, 255], 1),  # r_hip -> r_knee
            ([14, 16], [17, 25, 10], 1),  # r_knee -> r_ankle
            ([12, 6], [0, 0, 255], 1),  # r_hip  -> r_shoulder
            ([3, 1], [0, 255, 0], -1),  # l_ear -> l_eye
            ([1, 2], [0, 255, 5], 0),  # l_eye -> r_eye
            ([1, 0], [0, 255, 170], -1),  # l_eye -> nose
            ([0, 2], [0, 255, 25], 1),  # nose -> r_eye
            ([2, 4], [0, 17, 255], 1),  # r_eye -> r_ear
            ([9, 7], [0, 220, 0], -1),  # l_wrist -> l_elbow
            ([7, 5], [0, 220, 0], -1),  # l_elbow -> l_shoulder
            ([5, 6], [125, 125, 155], 0),  # l_shoulder -> r_shoulder
            ([6, 8], [25, 0, 55], 1),  # r_shoulder -> r_elbow
            ([8, 10], [25, 0, 255], 1),  # r_elbow -> r_wrist
        ]
    for i in range(len(config)):
        edge = config[i][0]
        key_points.append((query_locations[edge[0]][0],
                           query_locations[edge[0]][1],
                           query_locations[edge[1]][0],
                           query_locations[edge[1]][1],
                           config[i][2]))
    return key_points
