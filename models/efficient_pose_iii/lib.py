from track import analyze_image, get_model
from typing import List, Tuple

model, resolution = get_model(framework='tf', model_variant='iii')

def interference(filename: str) -> List[Tuple[str, int, int]]:
    """
    Interference function to run the model on the input image
    :param filename: str: file path of the image
    :return: list: coordinates of the keypoints (name of the keypoint, x, y coordinates)
    x and y coordinates are percentages of the image width and height where the keypoint is located
    """
    return analyze_image(model=model, file_path=filename, framework='tf', resolution=resolution, lite=False)

if __name__ == "__main__":
    keypoint_coords = interference("data/sampled_images/17905_2157397.jpg")
    print("Keypoint Coords: ")
    print(keypoint_coords)
