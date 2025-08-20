import sys
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

import src.utils.plot as plot
from src.features import keypoints
from src.features.features_cv2 import good_features_to_track
from src.image import Image
from src.klt import run_klt
from src.localization.localization import ransacLocalization, ransacLocalizationCV2
from src.structure_from_motion import sfm
from src.structure_from_motion.linear_triangulation import linear_triangulation
from src.structure_from_motion.reprojection_error import reprojection_error
from src.triangulate_landmarks import triangulate_landmarks
from src.utils import points
from src.utils.points import compute_bearing_angles_with_translation

np.set_printoptions(suppress=True)

# TODO: add a note about notation / documentation regarding (x,y) vs (y,x)

MAX_NUM_NEW_CANDIDATE_KEYPOINTS = 1000


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
    C1 = np.zeros((2, 0), dtype=np.int32)
    F1 = np.zeros((2, 0), dtype=np.int32)
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
        P1, status_mask = run_klt(image_0.img, image_1.img, P1)
        P1, X1 = points.apply_mask_many([P1, X1], status_mask)

        C0 = C1
        C1, status_mask_candiate_kps = run_klt(image_0.img, image_1.img, C1)
        C0, C1, F1, T1 = points.apply_mask_many(
            [C0, C1, F1, T1], status_mask_candiate_kps
        )
        print("After KLT")
        print(f"P1: {P1.shape}, X1: {X1.shape}, C1: {C1.shape}")

        R_C_W, t_C_W, best_inlier_mask = ransacLocalizationCV2(
            p_I_keypoints=P1, p_W_landmarks=X1, K=K
        )
        P1, X1 = points.apply_mask_many([P1, X1], best_inlier_mask)

        # region Plotting
        # plot.plot_keypoints(
        #     image_1.img,
        #     [P1[:, best_inlier_mask], P1[:, ~best_inlier_mask]],
        #     fmt=["gx", "rx"],
        # )
        # endregion

        print("After RANSAC")
        print(f"P1: {P1.shape}, X1: {X1.shape}, C1: {C1.shape}")

        if R_C_W is not None:
            T_C_W_flat = get_T_C_W_flat(R_C_W, t_C_W)
            T_C_W = T_C_W_flat.reshape((3, 4))
            print("POSE")
            print(T_C_W)
            camera_position = -R_C_W @ t_C_W
            print("CAMERA POSITION")
            print(camera_position)
        else:
            # TODO: Make sure this is correct behaviour
            continue

        print("REPROJECTION ERROR INIT")
        # reproj_error = reprojection_error(
        #     p_W_hom=np.r_[X1, np.ones((1, X1.shape[1]))],
        #     p_I=P1,
        #     T_C_W=T_C_W,
        #     K=K,
        # )
        # print(reproj_error)
        # if reproj_error > 100:
        #     print(X1[:, 0])

        C1_new, num_new_candidate_keypoints = keypoints.find_keypoints(
            img=image_1.img,
            max_keypoints=MAX_NUM_NEW_CANDIDATE_KEYPOINTS,
            exclude=[C1, P1],
        )

        if C1.shape[1] < 200:
            # keep new candidate keypoints
            C1 = np.c_[C1, C1_new]
            F1 = np.c_[F1, C1_new]
            T1 = np.c_[
                T1,
                np.tile(T_C_W_flat, (num_new_candidate_keypoints, 1)).T,
            ]

        # region Plotting
        # plot.plot_keypoints(img=image_1.img, p_I_keypoints=[P1, C1], fmt=["rx", "gx"])
        # plot.plot_keypoints(img=image_1.img, p_I_keypoints=P1, fmt="rx")
        # endregion

        if F1.any():
            _, _, mask_to_triangulate = compute_bearing_angles_with_translation(
                p_I_1=F1, p_I_2=C1, poses_A=T1, T_C_W=T_C_W_flat, K=K, min_angle=5.0
            )

            # plot.plot_landmarks_top_view(p_W=X1)

            p_W_hom_new_landmarks, mask_successful_triangulation = (
                triangulate_landmarks(
                    F1=F1,
                    C1=C1,
                    T1=T1,
                    T_C_W=T_C_W,
                    K=K,
                    mask_to_triangulate=mask_to_triangulate,
                )
            )
            C1_triangulated, F1_triangulated, T1_triangulated = points.apply_mask_many(
                [C1, F1, T1],
                mask_successful_triangulation,
            )

            # print("XXXXXXXXXXXXXXXX")
            # print(X1)
            # print(p_W_hom_landmark)
            # plot.plot_landmarks_top_view(p_W_hom_new_landmarks, "yx")

            best_inlier_mask_candidates = np.full(sum(status_landmarks), False)
            if C1.any() and mask_successful_triangulation.sum() >= 4:
                _, _, best_inlier_mask_candidates = ransacLocalizationCV2(
                    p_I_keypoints=C1_triangulated,
                    p_W_landmarks=p_W_hom_new_landmarks[:3, :],
                    K=K,
                )
                C1_triangulated_inliers, p_W_hom_new_landmarks_inliers = (
                    points.apply_mask_many(
                        [C1_triangulated, p_W_hom_new_landmarks],
                        best_inlier_mask_candidates,
                    )
                )

            P1 = np.c_[P1, C1_triangulated_inliers]
            X1 = np.c_[X1, p_W_hom_new_landmarks_inliers[:3, :]]
            reproj_error = reprojection_error(
                p_W_hom=p_W_hom_new_landmarks_inliers,
                p_I=C1_triangulated_inliers,
                T_C_W=T_C_W,
                K=K,
            )

            C1 = C1[:, ~mask_successful_triangulation]
            F1 = F1[:, ~mask_successful_triangulation]
            T1 = T1[:, ~mask_successful_triangulation]

            print("After adding new landmarks")
            print("Num new candidate keypoints: ", num_new_candidate_keypoints)
            # print("Num new landmarks added: ", best_inlier_mask_candidates.sum())
            print(f"P1: {P1.shape}")
            print(f"X1: {X1.shape}")
            print(f"C1: {C1.shape}")

        # region Plotting
        # plot.plot_tracking(
        #     I0_keypoints=C0,
        #     I1_keypoints=C1,
        #     figsize_pixels_x=image_0.img.shape[1],
        #     figsize_pixels_y=image_0.img.shape[0],
        # )
        # endregion
