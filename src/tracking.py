import sys
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

import plot
from features.features_cv2 import good_features_to_track
from image import Image
from localization import ransacLocalization
from structure_from_motion import sfm
from utils.utils_cv2 import from_cv2, to_cv2

# TODO: add a note about notation / documentation regarding (x,y) vs (y,x)


def assert_dtype_int(arr: np.ndarray):
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("arr is not of type int")


def keep_unique(p_P: np.ndarray, p_P_existing: np.ndarray):
    """
    Remove existing points from p_P

    Args:
        - p_P           np.ndarray(2,N) | (x,y)
        - p_P_existing  np.ndarray(2,N) | (x,y)
    """
    assert_dtype_int(p_P)
    assert_dtype_int(p_P_existing)

    # Example arrays

    # Transpose to shape (N, 2) for easy comparison
    p_P = p_P.T  # shape (4, 2)
    p_P_existing = p_P_existing.T  # shape (2, 2)

    # Create a mask of which rows in p_P are NOT in p_P_existing
    mask = ~np.any(np.all(p_P[:, None] == p_P_existing[None, :], axis=2), axis=1)

    # Filter p_P with the mask, then transpose back to shape (2, K)
    p_P_filtered = p_P[mask].T

    return p_P_filtered


"""
Si = (Pi,Xi,Ci,Fi,Ti)
Pi: 2xK
Xi: 2xK
Ci: 2xM
Fi: 2xM
Ti: 12xM

"""


def initialize_state(
    p_P_keypoints_initial: np.ndarray,
    p_W_landmarks_initial: np.ndarray,
):
    P0 = p_P_keypoints_initial
    X0 = p_W_landmarks_initial
    C0 = np.zeros((2, 0), dtype=np.int32)
    C1 = np.zeros((2, 0), dtype=np.int32)
    F0 = np.zeros((2, 0), dtype=np.int32)
    F1 = np.zeros((2, 0), dtype=np.int32)
    T0 = np.zeros((12, 0), dtype=np.int32)
    T1 = np.zeros((12, 0), dtype=np.int32)

    return P0, X0, C0, C1, F0, F1, T0, T1


def run_klt(image_0: Image, image_1: Image, p0_P_keypoints: np.ndarray):
    """
    Run KLT on the images: track keypoints from image_0 to image_1

    Args:
        - image_0 Image
        - image_1 Image
        - p0_P_keypoints np.ndarray(2,N) | (x,y)
    """
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # calculate optical flow
    p0_P_keypoints_cv2 = to_cv2(p0_P_keypoints)
    p1_P_keypoints_cv2, status, err = cv2.calcOpticalFlowPyrLK(
        image_0.img, image_1.img, p0_P_keypoints_cv2, None, **lk_params
    )

    # Select good points
    # p0_P_keypoints_cv2 = p0_P_keypoints_cv2[status == 1]
    # p1_P_keypoints_cv2 = p1_P_keypoints_cv2[status == 1]

    def from_cv2_status(st):
        """
        Returns:
            status: np.ndarray(N,): 1 if point tracked else 0
        """
        return st.T[0]

    if p1_P_keypoints_cv2 is None:
        return (
            np.zeros((2, 0), dtype=np.int32),
            np.zeros((2, 0), dtype=np.int32),
            np.full((0), False),
        )
    return (
        from_cv2(p0_P_keypoints_cv2),
        from_cv2(p1_P_keypoints_cv2),
        from_cv2_status(st=status).astype(np.bool8),
    )


def get_T_C_W(R_C_W, t_C_W):
    # TODO: is this correct?
    return np.c_[R_C_W, t_C_W]


def run_vo(
    images: Sequence[Image],
    p_P_keypoints_initial: np.ndarray,
    p_W_landmarks_initial: np.ndarray,
    K: np.ndarray,
):
    """
    Run a visual odometry pipeline on the images

    Args:
        - images list[np.ndarray]
        - p_P_keypoints_initial np.ndarray(2,N)   | (x,y)
        - p_W_landmarks_initial: np.ndarray(3, N) | (x,y,z)
        - K np.ndarray(3, 3): camera matrix
    """
    P0, X0, C0, C1, F0, F1, T0, T1 = initialize_state(
        p_P_keypoints_initial, p_W_landmarks_initial
    )

    for image_0, image_1 in zip(images, images[1:]):

        P0, P1, status_mask = run_klt(
            image_0=image_0, image_1=image_1, p0_P_keypoints=P0
        )
        P0, P1 = P0[:, status_mask], P1[:, status_mask]
        X0 = X0[:, status_mask]

        # if C0:
        C0, C1, status_mask_candiate_kps = run_klt(
            image_0=image_0, image_1=image_1, p0_P_keypoints=C0
        )
        C0 = C0[:, status_mask_candiate_kps]
        C1 = C1[:, status_mask_candiate_kps]
        # else:
        #     status_mask_candiate_kps = np.full((0), True)

        print(f"P1: {P1.shape}")
        print(f"X0: {X0.shape}")
        R_C_W_1, t_C_W_1, best_inlier_mask, _, _ = ransacLocalization(
            p_P_keypoints=P1,
            p_W_landmarks=X0,
            K=K,
        )
        X0 = X0[:, best_inlier_mask]
        P1 = P1[:, best_inlier_mask]

        if R_C_W_1 is not None:
            T_C_W_1 = get_T_C_W(R_C_W_1, t_C_W_1).flatten()
            camera_position = -R_C_W_1 @ t_C_W_1
            # print(camera_position)

        plot.plot_keypoints(img=image_1.img, p_P_keypoints=P1)

        # region Plotting
        # print(i_0.img.shape)
        # plot.plot_tracking(
        #     I0_keypoints=from_cv2(p0_P_keypoints_cv2)[:, best_inlier_mask],
        #     I1_keypoints=from_cv2(p1_P_keypoints_cv2)[:, best_inlier_mask],
        #     figsize_pixels_x=i_0.img.shape[1],
        #     figsize_pixels_y=i_0.img.shape[0],
        # )
        # endregion

        # region --- START: add new candidate keypoints ---
        C1_new = good_features_to_track(img=image_1.img, max_features=200)

        # Remove keypoints for which landmark is already computed
        C1_new = keep_unique(
            p_P=C1_new,
            p_P_existing=P1.astype(np.int16),
        )
        # Remove keypoints that are already tracked
        C1_new = keep_unique(
            p_P=C1_new,
            p_P_existing=C1,
        )
        num_new_candidate_keypoints = C1_new.shape[1]
        # endregion --- END: add new candidate keypoints ---

        C0, C1 = C1, np.c_[C1, C1_new]
        F0, F1 = F1, np.c_[F0[:, status_mask_candiate_kps == 1], C1_new]
        T0, T1 = (
            T1,
            np.c_[
                T0[:, status_mask_candiate_kps == 1],
                np.tile(T_C_W_1, (num_new_candidate_keypoints, 1)).T,
            ],
        )
        P0 = P1

        print("C1")
        print(C1.shape)
        print("F1")
        print(F1.shape)
        print("T1")
        print(T1.shape)

        # plot.plot_keypoints(
        #     img=i_1.img, p_P_keypoints=p_P_candidate_keypoints_new, fmt="gx"
        # )
