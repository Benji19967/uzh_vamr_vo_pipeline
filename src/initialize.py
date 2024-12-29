import numpy as np

from src.features import Descriptors, HarrisScores, Keypoints
from src.image import Image
from src.triangulate.decompose_essential_matrix import decomposeEssentialMatrix
from src.triangulate.disambiguate_relative_pose import disambiguateRelativePose
from src.triangulate.estimate_essential_matrix import estimateEssentialMatrix
from src.triangulate.triangulate import linear_triangulation


def get_keypoint_correspondences(
    I_0: Image, I_1: Image
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        tuple[np.ndarray, np.ndarray]:
            (
                (2, N) matched_keypoints1 in pixel coordinates (x, y),
                (2, N) matched_keypoints2 in pixel coordinates (x, y)
            )
    """
    keypoints = []
    descriptors = []
    kps = []
    for image in [I_0, I_1]:
        hs = HarrisScores(image=image)
        kp = Keypoints(image=image, scores=hs.scores)
        keypoints.append(kp.keypoints)
        kps.append(kp)
        kp.plot()
        # print(kp.keypoints)
        desc = Descriptors(image=image, keypoints=kp.keypoints)
        desc.plot()
        descriptors.append(desc.descriptors)

    matches = Descriptors.match(
        query_descriptors=descriptors[1], db_descriptors=descriptors[0]
    )
    Descriptors.plot_matches(
        matches=matches, query_keypoints=keypoints[1], database_keypoints=keypoints[0]
    )

    I_0_keypoints = keypoints[0]
    I_1_keypoints = keypoints[1]
    I_1_indices = np.nonzero(matches >= 0)[0]
    I_0_indices = matches[I_1_indices]

    I_0_matched_keypoints = np.zeros((2, len(I_1_indices)))
    I_1_matched_keypoints = np.zeros((2, len(I_1_indices)))

    # Switch pixel coordinates from (y, x) to (x, y)
    I_0_matched_keypoints[0:] = I_0_keypoints[1, I_0_indices]
    I_0_matched_keypoints[1:] = I_0_keypoints[0, I_0_indices]
    I_1_matched_keypoints[0:] = I_1_keypoints[1, I_1_indices]
    I_1_matched_keypoints[1:] = I_1_keypoints[0, I_1_indices]

    kps[0].plot(I_0_matched_keypoints)
    kps[1].plot(I_1_matched_keypoints)

    return I_0_matched_keypoints, I_1_matched_keypoints


def initialize(
    I_0: Image, I_1: Image, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        a (int): _description_
        b (np.ndarray): _description_

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (
                keypoints of image 1 in pixel coordinates (x, y),
                keypoints of image 2 in pixel coordinates (x, y),
                landmarks P_W in 3D coordinates (x, y, x)
            )
    """
    keypoints_1, keypoints_2 = get_keypoint_correspondences(I_0=I_0, I_1=I_1)

    # Triangulate
    num_keypoints = keypoints_1.shape[1]
    keypoints_1_homogeneous = np.r_[keypoints_1, np.ones((1, num_keypoints))]
    keypoints_2_homogeneous = np.r_[keypoints_2, np.ones((1, num_keypoints))]

    p1 = keypoints_1_homogeneous
    p2 = keypoints_2_homogeneous

    # Estimate the essential matrix E using the 8-point algorithm
    E = estimateEssentialMatrix(p1, p2, K, K)

    # Extract the relative camera positions (R,T) from the essential matrix
    # Obtain extrinsic parameters (R,t) from E
    Rots, u3 = decomposeEssentialMatrix(E)

    # Disambiguate among the four possible configurations
    R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, p1, p2, K, K)

    # Triangulate a point cloud using the final transformation (R,T)
    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R_C2_W, T_C2_W]

    points_3d_homogenous = linear_triangulation(
        p1=keypoints_1_homogeneous, p2=keypoints_2_homogeneous, M1=M1, M2=M2
    )
    points_3d = points_3d_homogenous[:3, :]

    return keypoints_1, keypoints_2, points_3d
