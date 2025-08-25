import numpy as np

from src.features.features import Descriptors
from src.features.features_cv2 import good_features_grid, good_features_to_track
from src.image import Image


def find_keypoints(
    img: np.ndarray,
    max_keypoints: int,
    exclude: list[np.ndarray] | None = None,
    use_grid: bool = True,
):
    """
    Args:
     - img
     - max_keypoints
     - exclude list[np.ndarray(2, N)]: keypoints already found in image

    Returns:
     - p_I_new_keypoints np.ndarray(2, N)
     - num_new_candidate_keypoints
    """
    if use_grid:
        p_I_new_keypoints = good_features_grid(img=img, max_features=max_keypoints)
    else:
        p_I_new_keypoints = good_features_to_track(img=img, max_features=max_keypoints)

    # Remove keypoints that are already found
    if exclude:
        for keypoints in exclude:
            p_I_new_keypoints = keep_unique(
                p_I=p_I_new_keypoints,
                p_I_existing=keypoints.astype(np.int16),
            )

    num_new_candidate_keypoints = p_I_new_keypoints.shape[1]

    return p_I_new_keypoints, num_new_candidate_keypoints


def get_keypoint_correspondences(
    image_0: Image,
    image_1: Image,
    max_num_keypoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
     - image_0, image_1: images to extract corresponding keypoints from
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
    for image in [image_0, image_1]:
        p_I_corners = good_features_to_track(
            img=image.img, max_features=max_num_keypoints
        )
        keypoints.append(p_I_corners)

        desc = Descriptors(image=image, keypoints=p_I_corners)
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


def keep_unique(p_I: np.ndarray, p_I_existing: np.ndarray):
    """
    Remove existing points from p_I

    Args:
        - p_I           np.ndarray(2,N) | (x,y)
        - p_I_existing  np.ndarray(2,N) | (x,y)
    """
    _assert_dtype_int(p_I)
    _assert_dtype_int(p_I_existing)

    # Example arrays

    # Transpose to shape (N, 2) for easy comparison
    p_I = p_I.T  # shape (4, 2)
    p_I_existing = p_I_existing.T  # shape (2, 2)

    # Create a mask of which rows in p_I are NOT in p_I_existing
    mask = ~np.any(np.all(p_I[:, None] == p_I_existing[None, :], axis=2), axis=1)

    # Filter p_I with the mask, then transpose back to shape (2, K)
    p_I_filtered = p_I[mask].T

    return p_I_filtered


def _assert_dtype_int(arr: np.ndarray):
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("arr is not of type int")
