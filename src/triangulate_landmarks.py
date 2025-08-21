import numpy as np

from src.structure_from_motion.linear_triangulation import linear_triangulation
from src.structure_from_motion.reprojection_error import reprojection_error
from src.utils import points


def triangulate_landmarks(
    F1: np.ndarray,
    C1: np.ndarray,
    T1: np.ndarray,
    T_C_W: np.ndarray,
    K: np.ndarray,
    mask_to_triangulate: np.ndarray,
    max_reprojection_error: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate landmarks from two sets of keypoints and camera poses.

    Only keep landmarks where the reprojection error is lower than the
    maximum allowed reprojection error and the points are in the positive z-direction.

    Args:
        F1 (np.ndarray) (2, N): First track of keypoints.
        C1 (np.ndarray) (2, N): Keypoints.
        T1 (np.ndarray) (12, N): Camera poses for the first image.
        T_C_W (np.ndarray) (3, 4): Camera pose for the second image in world coordinates.
        K (np.ndarray) (3, 3): Camera intrinsic matrix.
        mask_to_triangulate (np.ndarray) (N,): Boolean mask indicating which keypoints to triangulate.
        max_reprojection_error (int): Maximum allowed reprojection error for a successful triangulation.

    Returns:
        p_W_new_landmarks (np.ndarray) (3, T): Homogeneous coordinates of the triangulated landmarks.
        mask_successful_triangulation (np.ndarray) (N,): Boolean mask indicating which triangulations were successful.

    """
    # TODO: instead of triangulating each point individually, group points with same T1

    mask_successful_triangulation = []
    p_W_hom_new_landmarks = np.empty((4, mask_to_triangulate.shape[0]))
    for i, to_triangulate in enumerate(mask_to_triangulate):
        if to_triangulate == True:
            M1 = K @ T1[:, i : i + 1].reshape((3, 4))
            M2 = K @ T_C_W
            p1_I_hom = np.r_[F1[:, i : i + 1], [[1]]]
            p2_I_hom = np.r_[C1[:, i : i + 1], [[1]]]
            p_W_hom_landmark = linear_triangulation(
                p1_I_hom=p1_I_hom,
                p2_I_hom=p2_I_hom,
                M1=M1,
                M2=M2,
            )
            error = reprojection_error(
                p_W_hom=p_W_hom_landmark[:, :1],
                p_I=C1[:, i : i + 1],
                T_C_W=T_C_W,
                K=K,
            )
            if (
                error <= max_reprojection_error and p_W_hom_landmark[2, 0] > 0
            ):  # z-value:
                mask_successful_triangulation.append(True)
                p_W_hom_new_landmarks[:, i] = p_W_hom_landmark[:, 0]
            else:
                mask_successful_triangulation.append(False)
        else:
            mask_successful_triangulation.append(False)

    mask_successful_triangulation = np.array(mask_successful_triangulation)
    return (
        points.apply_mask(p_W_hom_new_landmarks[:3, :], mask_successful_triangulation),
        mask_successful_triangulation,
    )
