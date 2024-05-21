import posenet_lib.posenet as posenet
from posenet_lib.posenet.decode_multi import *

model = posenet.load_model(101)
output_stride = model.output_stride


def inference(filename: str) -> (np.ndarray, np.ndarray):
    """
    Interference the model with the input image
    :param filename: the path of the input image
    :return: keypoint_scores: the scores of the keypoints
            keypoint_coords: the coordinates of the keypoints
    """
    input_image, draw_image, output_scale = posenet.read_imgfile(
        filename, scale_factor=1.0, output_stride=output_stride)

    with torch.no_grad():
        input_image = torch.Tensor(input_image)

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
        pose_scores, keypoint_scores, keypoint_coords, pose_offsets = posenet.decode_multi.decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=output_stride,
            max_pose_detections=1,
            min_pose_score=0.25)

    keypoint_coords *= output_scale
    pose_id = np.argmax(pose_scores)
    keypoint_scores = keypoint_scores[pose_id, :]
    keypoint_coords = keypoint_coords[pose_id, :, :]
    return keypoint_scores, keypoint_coords

if __name__ == "__main__":
    keypoint_scores, keypoint_coords = inference("data/sampled_images/17905_2157397.jpg")
    print("Keypoint Scores:")
    print(keypoint_scores)
    print("Keypoint Coords:")
    print(keypoint_coords)
    # print(f"Total keypoints: {(len(poses_2d) - 1) // 3}")
    # print(f"Numbers of keypoints: {len(list(filter(lambda x: x != -1, poses_2d))) // 3}")

