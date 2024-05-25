import math
import cv2
from matplotlib import pyplot as plt


def get_pose(filename: str, key_points: list[tuple[float, float, float, float, int]], line_width=5):
    """
    Draw the pose on the image
    :param filename: str, the path to the image
    :param key_points: list, the list of edges - 2D coordinates of the key points in the format of (x0, y0, x1, y1, side) where side is -1 for left and 1 for right and 0 for center
    :param line_width: int, the width of the lines
    """

    image = cv2.imread(filename)
    for key_point in key_points:
        x0, y0, x1, y1, side = key_point
        if side == 0:
            color = (0, 255, 0)
        elif side == 1:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        x_mean = (x0 + x1) / 2
        y_mean = (y0 + y1) / 2
        length = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(x0 - x1, y0 - y1))
        polygon = cv2.ellipse2Poly((int(x_mean), int(y_mean)), (int(length / 2), int(line_width)), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(image, polygon, color)

    return image


def draw_pose(image_pose):
    plt.imshow(image_pose)
    plt.show()


if __name__ == '__main__':
    # przykład zwalony bo to TransPose
    draw_pose(get_pose(r"dogs-7369533.jpg", [(1312.8138732910156, 793.314208984375, 951.5969085693359, 755.1994323730469, -1), (951.5969085693359, 755.1994323730469, 436.7672348022461, 741.0279846191406, -1), (436.7672348022461, 741.0279846191406, 379.90299224853516, 457.5137710571289, -1), (276.54972076416016, 765.418701171875, 476.4210510253906, 753.2360076904297, 1), (476.4210510253906, 753.2360076904297, 368.16417694091797, 910.5940246582031, 1), (276.54972076416016, 765.418701171875, 153.32262992858887, 467.59265899658203, 1), (375.2032470703125, 342.64801025390625, 338.48949432373047, 327.1318817138672, -1), (338.48949432373047, 327.1318817138672, 269.00251388549805, 328.46385955810547, 0), (338.48949432373047, 327.1318817138672, 310.2978706359863, 346.7560577392578, -1), (310.2978706359863, 346.7560577392578, 269.00251388549805, 328.46385955810547, 1), (269.00251388549805, 328.46385955810547, 208.0987548828125, 347.0870590209961, 1), (490.57716369628906, 630.1102447509766, 458.22208404541016, 588.6084365844727, -1), (458.22208404541016, 588.6084365844727, 379.90299224853516, 457.5137710571289, -1), (379.90299224853516, 457.5137710571289, 153.32262992858887, 467.59265899658203, 0), (153.32262992858887, 467.59265899658203, 103.1588077545166, 626.0102462768555, 1), (103.1588077545166, 626.0102462768555, 131.92526817321777, 798.0825042724609, 1)]))
