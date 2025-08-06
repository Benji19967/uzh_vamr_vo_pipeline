import numpy as np

DESCRIPTOR_RADIUS = 9


def compute_descriptors(
    img: np.ndarray,
    p_P_keypoints: np.ndarray,
    descriptor_radius: int = DESCRIPTOR_RADIUS,
) -> np.ndarray:
    """Compute descriptors for keypoints

    Args:
     - img           np.ndarray
     - p_P_keypoints np.ndarray(2,N): (x,y) keypoints to compute descriptors for

    Returns:
     - descriptors   np.ndarray(R,N): R=(2 * r + 1) ** 2
    """
    r = descriptor_radius
    N = p_P_keypoints.shape[1]

    # `(2 * r + 1) ** 2` is the number of pixels in a patch/descriptor
    descriptors = np.zeros([(2 * r + 1) ** 2, N])
    padded = np.pad(img, [(r, r), (r, r)], mode="constant", constant_values=0)

    for i in range(N):
        kp = p_P_keypoints[:, i].astype(int) + r  # `+r` to account for padding

        # store the the pixel intensities of the descriptors in a flattened way
        descriptors[:, i] = padded[
            (kp[1] - r) : (kp[1] + r + 1), (kp[0] - r) : (kp[0] + r + 1)
        ].flatten()

    return descriptors
