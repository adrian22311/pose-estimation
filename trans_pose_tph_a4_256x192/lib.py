import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from hubconf import tph_a4_256x192
from lib.core.inference import get_final_preds
from lib.config import cfg


def inference(filename: str):
    """
    Interference the image and return the 2D poses
    :param filename: str, the path of the image
    :return: ndarray: 17x2 array of the coordinates of the key points
    """
    model = tph_a4_256x192(pretrained=True)

    # set up the model (download the weights)
    if filename is None:
        return None

    image = Image.open(filename)
    convert_tensor = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = convert_tensor(image)

    with torch.no_grad():
        model.eval()

        inputs = torch.cat([img]).unsqueeze(0)
        outputs = model(inputs.cpu())
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        preds = get_final_preds(cfg, output.clone().cpu().numpy(), None, None, transform_back=False)

    # from heatmap_coord to original_image_coord
    query_locations = np.array([p * 4 + 0.5 for p in preds[0]])

    # resize back to original image size
    query_locations = query_locations * np.array([image.size[0] / 192, image.size[1] / 256])

    return query_locations[0]


if __name__ == '__main__':
    inference(None)
