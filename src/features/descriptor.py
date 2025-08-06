import numpy as np
from scipy.spatial.distance import cdist

DESCRIPTOR_RADIUS = 9
MATCH_LAMBDA = 4


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


def match_descriptors(
    query_descriptors: np.ndarray,
    db_descriptors: np.ndarray,
    match_lambda: int = MATCH_LAMBDA,
) -> np.ndarray:
    """
    For each query_descriptor find the closest db_descriptor.
    Use each db_descriptor only once.

    Args:
        query_descriptors np.ndarray(R, N_Q): descriptors at time t2
        db_descriptors np.ndarray(R, N_D): descriptors at time t1
        match_lambda (int):

    Returns:
        np.ndarray: (N_Q,)
    """
    # shape: (N_Q, N_D)
    # distance from each query descriptor to each database descriptor
    dists = cdist(query_descriptors.T, db_descriptors.T, "euclidean")

    # shape: (N_Q, 1)
    # for each query_descriptor, which db_descriptor (index) is closest (argmin)
    matches = np.argmin(dists, axis=1)

    # shape: (N_Q, 1)
    # keep only distances that matched in `matches`
    dists = dists[np.arange(matches.shape[0]), matches]

    # scalar
    # min distance between any two descriptors across both sets
    min_non_zero_dist = dists.min()

    # keep only descriptors with small distance
    # adaptive threshold (because there should be at least one match)
    matches[dists >= match_lambda * min_non_zero_dist] = -1

    # remove double matches:
    # if a db_descriptor was assigned to several query_descriptors, keep only 1 match
    unique_matches = np.ones_like(matches) * -1
    _, unique_match_idxs = np.unique(matches, return_index=True)
    unique_matches[unique_match_idxs] = matches[unique_match_idxs]

    return unique_matches
