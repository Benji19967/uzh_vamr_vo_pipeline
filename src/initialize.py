import sys

import cv2
import numpy as np

from src.features.keypoints import get_keypoint_correspondences
from src.localization.pnp_ransac_localization import pnp_ransac_localization_cv2
from src.mapping.structure_from_motion import sfm
from src.structures.keypoints2D import Keypoints2D
from src.structures.landmarks3D import Landmarks3D
from src.utils.image import Image

MAX_NUM_KEYPOINTS = 1000


def initialize_cv2(img1, img2, K):
    # --- 3. Detect ORB features ---
    img1 = cv2.imread(str(img1.filepath), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2.filepath), cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # --- 4. Match features ---
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # --- 5. Extract matched points ---
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # --- 6. Compute Essential matrix ---
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    # --- 7. Recover relative camera pose ---
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    print(mask_pose)

    # --- 8. Triangulate points ---
    inliers = mask_pose.ravel() != 0
    pts1_inliers = pts1[inliers].reshape(-1, 1, 2).astype(np.float32)
    pts2_inliers = pts2[inliers].reshape(-1, 1, 2).astype(np.float32)
    distCoeffs = np.zeros((4, 1), dtype=np.float32)
    pts1_hom = cv2.undistortPoints(pts1_inliers, K, distCoeffs)
    pts2_hom = cv2.undistortPoints(pts2_inliers, K, distCoeffs)

    # Projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_hom, pts2_hom)
    points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T  # Nx3 array

    # --- 9. Filter points in front of cameras ---
    points_3d = points_3d[points_3d[:, 2] > 0]
    print(points_3d.shape)

    print(f"Generated {points_3d.shape[0]} initial 3D points")

    # print(pts1_inliers.T.shape)
    # print(pts2_inliers.T.shape)
    # print(points_3d.T.shape)
    # sys.exit(0)

    return (
        Keypoints2D(pts1_inliers.reshape(-1, 2).T),
        Keypoints2D(pts2_inliers.reshape(-1, 2).T),
        Landmarks3D(points_3d.T),
    )


def initialize(
    image_0: Image,
    image_1: Image,
    K: np.ndarray,
) -> tuple[Keypoints2D, Keypoints2D, Landmarks3D]:
    """
    From two images find a set of corresponding 2D keypoints and compute the
    associated 3D landmarks.

    Run RANSAC to remove outliers.

    Args:
     - image_0, image_1: images to extract corresponding keypoints from
     - K: camera matrix
    """
    p1_I_keypoints, p2_I_keypoints = get_keypoint_correspondences(
        image_0=image_0, image_1=image_1, max_num_keypoints=MAX_NUM_KEYPOINTS
    )

    p_W, _, _ = sfm.run_sfm(p1_I=p1_I_keypoints, p2_I=p2_I_keypoints, K=K)
    _, best_inlier_mask, _ = pnp_ransac_localization_cv2(
        p_I_keypoints=p1_I_keypoints,
        p_W_landmarks=p_W,
        K=K,
    )

    return (
        Keypoints2D(p1_I_keypoints[:, best_inlier_mask]),
        Keypoints2D(p2_I_keypoints[:, best_inlier_mask]),
        Landmarks3D(p_W[:, best_inlier_mask]),
    )
