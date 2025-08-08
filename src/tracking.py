import sys
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

import plot
from features import keypoints
from features.features_cv2 import good_features_to_track
from image import Image
from klt import run_klt
from localization import ransacLocalization
from src.angle import compute_bearing_angles_with_translation, plot_angle
from structure_from_motion import sfm

# TODO: add a note about notation / documentation regarding (x,y) vs (y,x)


def initialize_state(
    p_I_keypoints_initial: np.ndarray,
    p_W_landmarks_initial: np.ndarray,
):
    """
    Si = (Pi,Xi,Ci,Fi,Ti)
    Pi: 2xK
    Xi: 2xK
    Ci: 2xM
    Fi: 2xM
    Ti: 12xM
    """
    P0 = p_I_keypoints_initial
    X0 = p_W_landmarks_initial
    # C0 = np.zeros((2, 0), dtype=np.int32)
    C1 = np.zeros((2, 0), dtype=np.int32)
    # F0 = np.zeros((2, 0), dtype=np.int32)
    F1 = np.zeros((2, 0), dtype=np.int32)
    # T0 = np.zeros((12, 0), dtype=np.int32)
    T1 = np.zeros((12, 0), dtype=np.int32)

    return P0, X0, C1, F1, T1


# TODO: do I need T_W_C rather than T_C_W?
def get_T_C_W_flat(R_C_W, t_C_W):
    """
    From (3, 4):

    r11 r12 r13 tx
    r21 r22 r23 ty
    r31 r32 r33 tz

    to (1, 12):

    r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
    """
    return np.c_[R_C_W, t_C_W].flatten()


def filter(p_Points: np.ndarray, mask: np.ndarray):
    """
    Keep only points of mask

    Args:
        - p_Points  np.ndarray(2, N)
        - mask      np.ndarray(N,)
    """
    if p_Points.any():
        return p_Points[:, mask]
    return p_Points


def run_vo(
    images: Sequence[Image],
    p_I_keypoints_initial: np.ndarray,
    p_W_landmarks_initial: np.ndarray,
    K: np.ndarray,
):
    """
    Run a visual odometry pipeline on the images

    Args:
        - images list[np.ndarray]
        - p_I_keypoints_initial np.ndarray(2,N)   | (x,y)
        - p_W_landmarks_initial: np.ndarray(3, N) | (x,y,z)
        - K np.ndarray(3, 3): camera matrix
    """
    P1, X1, C1, F1, T1 = initialize_state(p_I_keypoints_initial, p_W_landmarks_initial)

    for image_0, image_1 in zip(images, images[1:]):
        P1, status_mask = run_klt(image_0, image_1, P1)
        P1 = filter(P1, status_mask)
        X1 = filter(X1, status_mask)
        print(f"P1: {P1.shape}")
        print(f"X1: {X1.shape}")

        C1, status_mask_candiate_kps = run_klt(image_0, image_1, C1)
        C1 = filter(C1, status_mask_candiate_kps)
        F1 = filter(F1, status_mask_candiate_kps)
        T1 = filter(T1, status_mask_candiate_kps)

        R_C_W_1, t_C_W_1, best_inlier_mask, _, _ = ransacLocalization(
            p_I_keypoints=P1,
            p_W_landmarks=X1,
            K=K,
        )
        X1 = filter(X1, best_inlier_mask)
        P1 = filter(P1, best_inlier_mask)

        if R_C_W_1 is not None:
            T_C_W_1 = get_T_C_W_flat(R_C_W_1, t_C_W_1)
            camera_position = -R_C_W_1 @ t_C_W_1
            # print(camera_position)

        C1_new, num_new_candidate_keypoints = keypoints.find_keypoints(
            img=image_1.img,
            max_keypoints=200,
            exclude=[C1, P1],
        )

        print("C1")
        print(C1.shape)
        print("F1")
        print(F1.shape)
        print("T1")
        print(R_C_W_1, t_C_W_1)

        if C1.shape[1] < 200:
            # keep new candidate keypoints
            C1 = np.c_[C1, C1_new]
            F1 = np.c_[F1, C1_new]
            T1 = np.c_[
                T1,
                np.tile(T_C_W_1, (num_new_candidate_keypoints, 1)).T,
            ]

        plot.plot_keypoints(img=image_1.img, p_I_keypoints=[P1, C1], fmt=["rx", "gx"])

        if F1.any():
            _, angles_deg, mask = compute_bearing_angles_with_translation(
                p_I_1=F1,
                p_I_2=C1,
                poses_A=T1,
                T_C_W=T_C_W_1,
                K=K,
            )
            print(angles_deg)
            print(mask)
            # TODO: points where mask True --> triangulate

            # point_idx = 0
            # angle_deg = plot_angle(
            #     x1=F1[:, point_idx].T,
            #     x2=C1[:, point_idx].T,
            #     K=K,
            #     R1=T1[:, point_idx].reshape((3, 4))[:, :3],
            #     t1=T1[:, point_idx].reshape((3, 4))[:, 3],
            #     R2=R_C_W_1,
            #     t2=t_C_W_1,
            # )
            # print(status_mask_candiate_kps)
            # print(angle_deg)
            print(num_new_candidate_keypoints)

        # region Plotting
        # print(i_0.img.shape)
        # plot.plot_tracking(
        #     I0_keypoints=from_cv2(p0_I_keypoints_cv2)[:, best_inlier_mask],
        #     I1_keypoints=from_cv2(p1_I_keypoints_cv2)[:, best_inlier_mask],
        #     figsize_pixels_x=i_0.img.shape[1],
        #     figsize_pixels_y=i_0.img.shape[0],
        # )
        # endregion
