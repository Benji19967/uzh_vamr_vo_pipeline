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
from localization import ransacLocalization, ransacLocalizationCV2
from src.angle import compute_bearing_angles_with_translation, plot_angle
from src.structure_from_motion.linear_triangulation import (
    linear_triangulation,
    reprojection_error,
)
from structure_from_motion import sfm

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

        C0 = C1
        C1, status_mask_candiate_kps = run_klt(image_0, image_1, C1)
        C0 = filter(C0, status_mask_candiate_kps)
        C1 = filter(C1, status_mask_candiate_kps)
        F1 = filter(F1, status_mask_candiate_kps)
        T1 = filter(T1, status_mask_candiate_kps)

        print("After KLT")
        print(f"P1: {P1.shape}")
        print(f"X1: {X1.shape}")
        print(f"C1: {C1.shape}")

        R_C_W, t_C_W, best_inlier_mask = ransacLocalizationCV2(
            p_I_keypoints=P1,
            p_W_landmarks=X1,
            K=K,
        )
        print(R_C_W, t_C_W, best_inlier_mask)
        plot.plot_keypoints(
            image_1.img,
            [P1[:, best_inlier_mask], P1[:, ~best_inlier_mask]],
            fmt=["gx", "rx"],
        )
        X1 = filter(X1, best_inlier_mask)
        P1 = filter(P1, best_inlier_mask)

        print("After RANSAC")
        print(f"P1: {P1.shape}")
        print(f"X1: {X1.shape}")
        print(f"C1: {C1.shape}")

        if R_C_W is not None:
            T_C_W_flat = get_T_C_W_flat(R_C_W, t_C_W)
            T_C_W = T_C_W_flat.reshape((3, 4))
            print("POSE")
            print(T_C_W)
            camera_position = -R_C_W @ t_C_W
            # print(camera_position)

        print("REPROJECTION ERROR INIT")
        reproj_error = reprojection_error(
            p_W_hom=np.r_[X1, np.ones((1, X1.shape[1]))],
            p_I=P1,
            T_C_W=T_C_W,
            K=K,
        )
        print(reproj_error)
        if reproj_error > 100:
            print(X1[:, 0])

        C1_new, num_new_candidate_keypoints = keypoints.find_keypoints(
            img=image_1.img,
            max_keypoints=MAX_NUM_NEW_CANDIDATE_KEYPOINTS,
            exclude=[C1, P1],
        )

        # if C1.shape[1] < 200:
        # keep new candidate keypoints
        C1 = np.c_[C1, C1_new]
        F1 = np.c_[F1, C1_new]
        T1 = np.c_[
            T1,
            np.tile(T_C_W_flat, (num_new_candidate_keypoints, 1)).T,
        ]

        # plot.plot_keypoints(img=image_1.img, p_I_keypoints=[P1, C1], fmt=["rx", "gx"])
        # plot.plot_keypoints(img=image_1.img, p_I_keypoints=P1, fmt="rx")

        if F1.any():
            _, angles_deg, mask_angle = compute_bearing_angles_with_translation(
                p_I_1=F1,
                p_I_2=C1,
                poses_A=T1,
                T_C_W=T_C_W_flat,
                K=K,
            )
            # print(angles_deg)
            # print(mask_angle)

            # TODO: points where mask True --> triangulate
            # C1_to_triangulate = C1[:, mask_angle]
            # F1_to_triangulate = F1[:, mask_angle]
            # T1_to_triangulate = T1[:, mask_angle]

            # print("PPPPPPPPPPPP")
            # print(P1)
            # print(C1_to_triangulate)
            # print(F1_to_triangulate)

            plot.plot_landmarks_top_view(p_W=X1)

            status_landmarks = []
            p_W_hom_new_landmarks = np.empty((4, C1.shape[1]))
            for i, mask_angle_status in enumerate(mask_angle):
                if mask_angle_status == True:
                    M1 = K @ T1[:, i : i + 1].reshape((3, 4))
                    M2 = K @ T_C_W
                    p1_I_hom = np.r_[F1[:, i : i + 1], [[1]]]
                    p2_I_hom = np.r_[C1[:, i : i + 1], [[1]]]
                    # print("TRIANGULATION")
                    # print(p1_I_hom)
                    # print(p2_I_hom)
                    # print(M1)
                    # print(M2)
                    p_W_hom_landmark = linear_triangulation(
                        p1_I_hom=p1_I_hom,
                        p2_I_hom=p2_I_hom,
                        M1=M1,
                        M2=M2,
                    )
                    # print(p_W_hom_landmark)
                    # print("REPROJECTION ERROR")
                    # print(
                    #     reprojection_error(
                    #         p_W_hom=p_W_hom_landmark[:, 0],
                    #         p_I=C1[:, i : i + 1],
                    #         T_C_W=T_C_W_flat.reshape((3, 4)),
                    #         K=K,
                    #     )
                    # )
                    p_W_hom_new_landmarks[:, i] = p_W_hom_landmark[:, 0]
                    if p_W_hom_landmark[2, 0] > 0:  # z-value
                        status_landmarks.append(True)
                    else:
                        status_landmarks.append(False)
                else:
                    status_landmarks.append(False)

                # print("XXXXXXXXXXXXXXXX")
                # print(X1)
                # print(p_W_hom_landmark)
                # plot.plot_landmarks_top_view(p_W_hom_new_landmarks, "yx")

            status_mask_candidate_landmarks = np.array(status_landmarks)
            best_inlier_mask_candidates = np.full(len(status_landmarks), False)
            # if C1.any() and status_mask_candidate_landmarks.sum() > 0:
            #     _, _, best_inlier_mask_candidates, _, _ = ransacLocalization(
            #         p_I_keypoints=C1[:, status_mask_candidate_landmarks],
            #         p_W_landmarks=p_W_hom_new_landmarks[
            #             :3, status_mask_candidate_landmarks
            #         ],
            #         K=K,
            #     )

            # P1 = np.c_[
            #     P1,
            #     filter(
            #         C1[:, status_mask_candidate_landmarks], best_inlier_mask_candidates
            #     ),
            # ]
            # X1 = np.c_[
            #     X1,
            #     filter(
            #         p_W_hom_new_landmarks[:3, status_mask_candidate_landmarks],
            #         best_inlier_mask_candidates,
            #     ),
            # ]
            #
            # C1 = C1[:, ~status_mask_candidate_landmarks]
            # F1 = F1[:, ~status_mask_candidate_landmarks]
            # T1 = T1[:, ~status_mask_candidate_landmarks]
            #
            print("After adding new landmarks")
            print("Num new candidate keypoints: ", num_new_candidate_keypoints)
            print("Num new landmarks added: ", best_inlier_mask_candidates.sum())
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
