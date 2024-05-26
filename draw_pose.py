import math

import cv2
from matplotlib import pyplot as plt


def get_pose(
    filename: str,
    key_points: list[tuple[float | None, float | None]],
    edges: list[tuple[tuple[int, int], int]],
    line_width=5,
):
    """
    Draw the pose on the image
    :param filename: str, the path to the image
    :param key_points: list, the 2D coordinates of the key points in the format of (x, y)
    :param edges: list, the edges in the format of ((point1, point2), side), where side is -1 for left and 1 for right and 0 for center
    :param line_width: int, the width of the lines
    """

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for key_point in key_points:
        if key_point[0] is None or key_point[1] is None:
            continue
        cv2.circle(
            image,
            (int(key_point[0]), int(key_point[1])),
            line_width * 3,
            (255, 255, 255),
            thickness=-1,
        )
        cv2.circle(
            image,
            (int(key_point[0]), int(key_point[1])),
            line_width * 2,
            (0, 0, 0),
            thickness=-1,
        )
    for edge in edges:
        x0, y0 = key_points[edge[0][0]]
        x1, y1 = key_points[edge[0][1]]
        if x0 is None or x1 is None or y0 is None or y1 is None:
            continue

        side = edge[1]
        if side == 0:
            color = (0, 255, 0)
        elif side == 1:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        x_mean = (x0 + x1) / 2
        y_mean = (y0 + y1) / 2
        length = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y0 - y1, x0 - x1))
        polygon = cv2.ellipse2Poly((int(x_mean), int(y_mean)), (int(length / 2), int(line_width)), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(image, polygon, color)
        
    return image


def draw_pose(image_pose):
    plt.imshow(image_pose)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.show()


if __name__ == '__main__':
    draw_pose(get_pose(r"dogs-7369533.jpg",
                       [(310.2978706359863, 346.7560577392578), (338.48949432373047, 327.1318817138672),
                        (269.00251388549805, 328.46385955810547), (375.2032470703125, 342.64801025390625),
                        (208.0987548828125, 347.0870590209961), (379.90299224853516, 457.5137710571289),
                        (153.32262992858887, 467.59265899658203), (458.22208404541016, 588.6084365844727),
                        (103.1588077545166, 626.0102462768555), (490.57716369628906, 630.1102447509766),
                        (131.92526817321777, 798.0825042724609), (436.7672348022461, 741.0279846191406),
                        (276.54972076416016, 765.418701171875), (951.5969085693359, 755.1994323730469),
                        (476.4210510253906, 753.2360076904297), (1312.8138732910156, 793.314208984375),
                        (368.16417694091797, 910.5940246582031)],
                       [
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
                       ))
