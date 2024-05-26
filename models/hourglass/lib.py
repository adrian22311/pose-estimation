import torch
import numpy as np

from soft_gated.config import config
import soft_gated.utils.model as model_utils
import soft_gated.utils.img as img_utils
from imageio.v2 import imread
from soft_gated.utils.keypoints import get_keyppoints, post_process_keypoints

net = model_utils.load_model(config)
net = model_utils.load_model_weights(config, net)
net.eval()

input_res = 256

def prepare_image(image):
    height, width = image.shape[0:2]
    c = np.array((width/2, height/2))
    s = max(height, width)/200

    cropped = img_utils.crop(image, c, s, (input_res, input_res))
    cropped = cropped / 255
    inp = torch.from_numpy(cropped.copy()) # returns shape  [256, 256, 3]

    inp = inp.permute(2, 0, 1) # change shape (h, w, c) to (c, h, w )
    # inp shape: [3, 256, 256]
    inp = inp.type(torch.FloatTensor).unsqueeze(dim=0) # add batch dimension
    return inp, c, s


def inference(filename: str) -> np.ndarray:
    """
    Args:
        filename: str: path to image file
        
    Returns:
        np.ndarray: 2D array of keypoint coordinates (16 keypoints, 2 coordinates each)   
    """
    img = imread(filename)
    input, c, s = prepare_image(img)
    with torch.no_grad():
        heatmaps = net(input)
    
    pred_keypoints = get_keyppoints(heatmaps[:, -1])
    keypoints = post_process_keypoints(pred_keypoints, input, c, s, input_res)[0]
    return keypoints
    
if __name__ == "__main__":
    keypoint_coords = inference("data/sampled_images/17905_2157397.jpg")
    print("Keypoint Coords: ")
    print(keypoint_coords)
    print("Shape: ", keypoint_coords.shape)
    
