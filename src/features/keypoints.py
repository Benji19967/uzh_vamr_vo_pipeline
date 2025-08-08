import numpy as np

from features.features import Descriptors
from image import Image
from src.features.features_cv2 import good_features_to_track


def get_keypoint_correspondences(
    I_0: Image,
    I_1: Image,
    max_num_keypoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
     - I_0, I_1: images to extract corresponding keypoints from
     - max_num_keypoints: max number of keypoints to extract

    Returns:
        tuple[np.ndarray, np.ndarray]:
            (
                (2, N) matched_keypoints1 in pixel coordinates (x, y),
                (2, N) matched_keypoints2 in pixel coordinates (x, y)
            )
    """
    keypoints = []
    descriptors = []
    for image in [I_0, I_1]:
        p_P_corners = good_features_to_track(
            img=image.img, max_features=max_num_keypoints
        )
        keypoints.append(p_P_corners)

        desc = Descriptors(image=image, keypoints=p_P_corners)
        descriptors.append(desc.descriptors)

    matches = Descriptors.match(
        query_descriptors=descriptors[1], db_descriptors=descriptors[0]
    )

    I_0_keypoints = keypoints[0]
    I_1_keypoints = keypoints[1]
    I_1_indices = np.nonzero(matches >= 0)[0]
    I_0_indices = matches[I_1_indices]

    I_0_matched_keypoints = np.zeros((2, len(I_1_indices)))
    I_1_matched_keypoints = np.zeros((2, len(I_1_indices)))

    I_0_matched_keypoints[0:] = I_0_keypoints[0, I_0_indices]
    I_0_matched_keypoints[1:] = I_0_keypoints[1, I_0_indices]
    I_1_matched_keypoints[0:] = I_1_keypoints[0, I_1_indices]
    I_1_matched_keypoints[1:] = I_1_keypoints[1, I_1_indices]

    return I_0_matched_keypoints, I_1_matched_keypoints
