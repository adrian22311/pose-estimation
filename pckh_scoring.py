import numpy as np


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def validate_keypoints_format(keypoints):
    if len(keypoints) % 3 != 0:
        raise ValueError("Invalid number of keypoints. The number of keypoints must be divisible by 3.")
    for i in range(len(keypoints) // 3):
        visibility = keypoints[i * 3 + 2]
        if visibility not in [0, 1, 2]:
            raise ValueError("Invalid visibility value. Visibility must be 0, 1, or 2.")

def calculate_pckh(gt_keypoints, pred_keypoints, threshold=0.5, bbox: list[float]=None, strict=False):
    # Validate keypoints format
    validate_keypoints_format(gt_keypoints)
    validate_keypoints_format(pred_keypoints)

    head_segment_indices=(3, 4)  # left ear and right ear

    num_gt_keypoints = len(gt_keypoints) // 3
    num_pred_keypoints = len(pred_keypoints) // 3

    if num_gt_keypoints != num_pred_keypoints:
        raise ValueError("The number of keypoints in ground truth and predictions must be the same")

    num_correct_keypoints = 0
    num_valid_keypoints = 0
    head_segment_length = 0

    # Calculate head segment length using the specified keypoints
    if gt_keypoints[head_segment_indices[0]*3+2] != 0 and gt_keypoints[head_segment_indices[1]*3+2] != 0:
        head_segment_length = euclidean_distance(
            gt_keypoints[head_segment_indices[0]*3:head_segment_indices[0]*3+2],
            gt_keypoints[head_segment_indices[1]*3:head_segment_indices[1]*3+2]
        )
    if head_segment_length <= 1e-6:
        head_segment_indices=(1, 2)  # left eye and right eye
        if gt_keypoints[head_segment_indices[0]*3+2] != 0 and gt_keypoints[head_segment_indices[1]*3+2] != 0:
            head_segment_length = euclidean_distance(
                gt_keypoints[head_segment_indices[0]*3:head_segment_indices[0]*3+2],
                gt_keypoints[head_segment_indices[1]*3:head_segment_indices[1]*3+2]
            ) * 1.25
    if head_segment_length <= 1e-6: # in case ground truth doesn't contain information about ears/eyes
        head_segment_length = bbox[2]*0.1

    for i in range(num_gt_keypoints):
        gt_x, gt_y, gt_vis = gt_keypoints[i*3:i*3+3]
        pred_x, pred_y, pred_vis = pred_keypoints[i*3:i*3+3]

        if gt_vis > (0 + int(strict)):
            if pred_vis != 2:
                num_valid_keypoints += 1
            if pred_vis == 1:
                distance = euclidean_distance((gt_x, gt_y), (pred_x, pred_y))
                normalized_distance = distance / head_segment_length
                if normalized_distance <= threshold:
                    num_correct_keypoints += 1

    # pckh = num_correct_keypoints / num_valid_keypoints if num_valid_keypoints > 0 else 0
    return num_correct_keypoints, num_valid_keypoints

if __name__ == "__main__":
    # Example usage
    gt_keypoints_0 = [44, 32, 2, 49, 28, 2, 39, 29, 2, 0, 0, 0, 28, 21, 2, 57, 30, 2, 17, 32, 2, 0, 0, 0, 13, 70, 1, 70, 60, 2, 43, 64, 1, 55, 84, 1, 26, 85, 1, 80, 105, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pred_keypoints_0 = [45, 33, 2, 50, 29, 2, 40, 30, 2, 0, 0, 0, 29, 22, 2, 58, 31, 2, 18, 33, 2, 0, 0, 0, 14, 71, 1, 71, 61, 2, 44, 65, 1, 56, 85, 1, 27, 86, 1, 81, 106, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pckh_0 = calculate_pckh(gt_keypoints_0, pred_keypoints_0)
    print(f'PCKh@0.5: {pckh_0:.2f}')

    gt_keypoints_1 = [44, 32, 2, 39, 18, 2, 49, 19, 2, 4, 4, 2, 22, 45, 2, 57, 30, 2, 17, 32, 2, 0, 0, 0, 13, 70, 1, 70, 60, 2, 43, 64, 1, 55, 84, 1, 26, 85, 1, 80, 105, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pred_keypoints_1 = [45, 33, 2, 50, 29, 2, 40, 30, 2, 0, 0, 0, 29, 22, 2, 58, 31, 2, 18, 33, 2, 0, 0, 0, 14, 71, 1, 71, 61, 2, 44, 65, 1, 56, 85, 1, 27, 86, 1, 81, 106, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pckh_1 = calculate_pckh(gt_keypoints_1, pred_keypoints_1)
    print(f'PCKh@0.5: {pckh_1:.2f}')


    gt_keypoints_2 = [44, 32, 2, 49, 28, 2, 39, 29, 2, 0, 0, 0, 28, 21, 2, 57, 30, 2, 17, 32, 2, 0, 0, 0, 13, 70, 1, 70, 60, 2, 43, 64, 1, 55, 84, 1, 26, 85, 1, 80, 105, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Predicted keypoints have incorrect format (missing visibility for one keypoint)
    pred_keypoints_2 = [44, 32, 2, 49, 28, 2, 39, 29, 2, 0, 0, 0, 28, 21, 2, 58, 31, 2, 18, 33, 2, 0, 0, 0, 14, 71, 71, 61, 2, 44, 65, 1, 56, 85, 1, 27, 86, 1, 81, 106, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    try:
        pckh_2 = calculate_pckh(gt_keypoints_2, pred_keypoints_2)
        print(f'PCKh@0.5: {pckh_2:.2f}')
    except ValueError as e:
        print(f'Error: {e}')