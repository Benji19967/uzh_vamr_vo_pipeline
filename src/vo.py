import logging

import numpy as np

from src.exceptions import FailedLocalizationError
from src.features import keypoints as kp
from src.io.ba_exporter import BAExporter
from src.localization.pnp_ransac_localization import pnp_ransac_localization_cv2
from src.mapping.reprojection_error import reprojection_error
from src.mapping.triangulate_landmarks import triangulate_landmarks
from src.plotting.visualizer import Visualizer
from src.structures.keypoints2D import CandidateKeypoints2D, Keypoints2D
from src.structures.landmarks3D import Landmarks3D
from src.tracking.klt import run_klt
from src.utils import points
from src.utils.masks import compose_masks
from src.utils.points import compute_bearing_angles_with_translation

np.set_printoptions(suppress=True)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MAX_NUM_CANDIDATE_KEYPOINTS = 1000
MAX_NUM_NEW_CANDIDATE_KEYPOINTS = 1000
MAX_REPROJECTION_ERROR = 5  # pixels
MIN_ANGLE_TO_TRIANGULATE = 5.0  # degrees
MIN_NUM_LANDMARKS_FOR_LOCALIZATION = 4
KEYFRAME_INTERVAL = 5  # Process every ith image as a keyframe


class VOPipeline:
    def __init__(self, visualizer: Visualizer, ba_exporter: BAExporter) -> None:
        self.visualizer = visualizer
        self.ba_exporter = ba_exporter

    def run(
        self,
        images: list[np.ndarray],
        p_I_keypoints_initial: np.ndarray,
        p_W_landmarks_initial: np.ndarray,
        K: np.ndarray,
        camera_positions_ground_truth: list[np.ndarray] | None = None,
    ):
        """
        Run a visual odometry pipeline on the images

        Args:
            - images list[np.ndarray]
            - p_I_keypoints_initial np.ndarray(2,N)   | (x,y)
            - p_W_landmarks_initial: np.ndarray(3, N) | (x,y,z)
            - K np.ndarray(3, 3): camera matrix
        """
        keypoints, landmarks, candidate_keypoints, F1, T1 = self.initialize_state(
            p_I_keypoints_initial, p_W_landmarks_initial
        )

        camera_positions, reprojection_errors = self.process_frames(
            images, K, keypoints, landmarks, candidate_keypoints, F1, T1
        )
        self.visualizer.trajectory(camera_positions, camera_positions_ground_truth)
        self.visualizer.reprojection_errors(reprojection_errors)
        if self.visualizer._plot_scale_drift:
            assert camera_positions_ground_truth
            self.visualizer.scale_drift(camera_positions, camera_positions_ground_truth)

    def process_frames(
        self,
        images,
        K,
        keypoints: Keypoints2D,
        landmarks: Landmarks3D,
        candidate_keypoints: CandidateKeypoints2D,
        F1,
        T1,
    ):
        camera_positions = []
        reprojection_errors = []
        for i, (img_0, img_1) in enumerate(zip(images, images[1:])):
            logger.debug(f"Iteration: {i}")

            # Track keypoints from img_0 to img_1
            keypoints, tracked_mask = run_klt(img_0, img_1, keypoints)
            keypoints.keep(tracked_mask)
            landmarks.keep(tracked_mask)

            # Track candidate keypoints from img_0 to img_1
            candidate_keypoints_0 = candidate_keypoints
            candidate_keypoints, tracked_mask = run_klt(
                img_0, img_1, candidate_keypoints
            )
            F1, T1 = points.apply_mask_many([F1, T1], tracked_mask)
            candidate_keypoints_0.keep(tracked_mask)
            candidate_keypoints.keep(tracked_mask)
            self.visualizer.tracking(
                candidate_keypoints_0.array, candidate_keypoints.array, img_0
            )
            logger.debug(
                f"After klt: keypoints: {keypoints.shape}, landmarks: {landmarks.shape}, candidate_keypoints: {candidate_keypoints.shape}"
            )

            # Localize: compute camera pose
            if landmarks.count < MIN_NUM_LANDMARKS_FOR_LOCALIZATION:
                raise ValueError(f"Not enough keypoints/landmarks for localization")
            try:
                T_C_W, best_inlier_mask, camera_position = pnp_ransac_localization_cv2(
                    keypoints.array, landmarks.array, K
                )
                keypoints.keep(best_inlier_mask)
                landmarks.keep(best_inlier_mask)
                camera_positions.append(camera_position)
                logger.debug(
                    f"After ransac: keypoints: {keypoints.shape}, landmarks: {landmarks.shape}, candidate_keypoints: {candidate_keypoints.shape}"
                )
                logger.debug(f"Pose:\n {T_C_W}")
                logger.debug(f"Camera position: {camera_position.flatten()}")
            except FailedLocalizationError:
                logger.debug(f"Failed Ransac localization")
                continue

            # Map: add new landmarks
            if i % KEYFRAME_INTERVAL == 0:
                candidate_keypoints, F1, T1 = self.add_new_candidate_keypoints(
                    img_1, keypoints, candidate_keypoints, F1, T1, T_C_W
                )
                keypoints, landmarks, candidate_keypoints, F1, T1 = (
                    self.add_new_landmarks(
                        keypoints, landmarks, candidate_keypoints, F1, T1, T_C_W, K
                    )
                )
                self.ba_exporter.write(T_C_W, landmarks, keypoints)
            if candidate_keypoints.count > MAX_NUM_CANDIDATE_KEYPOINTS:
                candidate_keypoints.keep_last(n=MAX_NUM_CANDIDATE_KEYPOINTS)
                F1 = F1[:, -MAX_NUM_CANDIDATE_KEYPOINTS:]
                T1 = T1[:, -MAX_NUM_CANDIDATE_KEYPOINTS:]

            # Evaluate results
            reproj_error = reprojection_error(
                landmarks.array_hom, keypoints.array, T_C_W, K
            )
            reprojection_errors.append(reproj_error)
            logger.debug(f"Reprojection error landmarks: {reproj_error}")

            self.visualizer.keypoints_and_landmarks(
                keypoints.array,
                landmarks.array,
                candidate_keypoints.array,
                camera_positions,
                img_1,
            )

        return camera_positions, reprojection_errors

    def add_new_candidate_keypoints(
        self,
        img_1: np.ndarray,
        keypoints: Keypoints2D,
        candidate_keypoints: CandidateKeypoints2D,
        F1,
        T1,
        T_C_W,
    ):
        """
        Add new candidate keypoints to the current set of keypoints.

        Args:
            img_1: np.ndarray: Current image.
            P1: np.ndarray(2, N): Current keypoints.
            C1: np.ndarray(2, M): Current candidate keypoints.
            F1: np.ndarray(2, M): First track of current candidate keypoints.
            T1: np.ndarray(12, M): Camera poses at first track of current candidate keypoints.
            T_C_W: np.ndarray(3, 4): Camera pose for the current image.
        """
        cp_new, num_new_candidate_keypoints = kp.find_keypoints(
            img_1,
            MAX_NUM_NEW_CANDIDATE_KEYPOINTS,
            exclude=[candidate_keypoints.array, keypoints.array],
        )
        candidate_keypoints.add(cp_new)
        F1 = np.c_[F1, cp_new]
        T1 = np.c_[
            T1, self.multiply_T(self.get_T_C_W_flat(T_C_W), num_new_candidate_keypoints)
        ]
        return candidate_keypoints, F1, T1

    def add_new_landmarks(
        self,
        keypoints: Keypoints2D,
        landmarks: Landmarks3D,
        candidate_keypoints: CandidateKeypoints2D,
        F1,
        T1,
        T_C_W,
        K,
    ):
        """
        Add new landmarks to the current set of landmarks.
        Remove the corresponding candidate keypoints from the current set.

        Requirements to find a new landmark:

        1. Bearing angle > threshold
        2. Reprojection error < threshold
        3. Is not an outlier when using RANSAC

        Args:
            P1: np.ndarray(2, N): Current keypoints.
            landmarks: Landmarks3D: Current landmarks.
            C1: np.ndarray(2, M): Current candidate keypoints.
            F1: np.ndarray(2, M): First track of current candidate keypoints.
            T1: np.ndarray(12, M): Camera poses at first track of current candidate keypoints.
            T_C_W: np.ndarray(3, 4): Camera pose for the current image.
            K: np.ndarray(3, 3): Camera intrinsic matrix.
        Returns:
            P1: np.ndarray(2, N): Updated keypoints.
            landmarks: Landmarks3D: Updated landmarks.
            C1: np.ndarray(2, M): Updated candidate keypoints.
        """
        assert F1.any()
        _, _, mask_to_triangulate = compute_bearing_angles_with_translation(
            F1, candidate_keypoints.array, T1, T_C_W, K, MIN_ANGLE_TO_TRIANGULATE
        )

        p_W_new_landmarks, mask_successful_triangulation = triangulate_landmarks(
            F1,
            candidate_keypoints.array,
            T1,
            T_C_W,
            K,
            mask_to_triangulate,
            MAX_REPROJECTION_ERROR,
        )
        ckp_triangulated = candidate_keypoints.filtered(mask_successful_triangulation)
        logger.debug(f"Successful triangulation: {mask_successful_triangulation.sum()}")

        best_inlier_mask_ransac = np.full(mask_successful_triangulation.sum(), False)
        if candidate_keypoints.array.any() and mask_successful_triangulation.sum() >= 4:
            _, best_inlier_mask_ransac, _ = pnp_ransac_localization_cv2(
                ckp_triangulated.array, p_W_new_landmarks, K
            )
            logger.debug(f"Num ransac inliers: {best_inlier_mask_ransac.sum()}")

            ckp_triangulated.keep(best_inlier_mask_ransac)
            p_W_new_landmarks_inliers = points.apply_mask(
                p_W_new_landmarks, best_inlier_mask_ransac
            )

            mask_new_landmarks = compose_masks(
                mask_successful_triangulation, best_inlier_mask_ransac
            )

            keypoints.add(ckp_triangulated.array)
            landmarks.add(p_W_new_landmarks_inliers)

            F1, T1 = points.apply_mask_many([F1, T1], ~mask_new_landmarks)
            candidate_keypoints.keep(~mask_new_landmarks)

        return keypoints, landmarks, candidate_keypoints, F1, T1

    def multiply_T(self, T_C_W_flat, num_new_candidate_keypoints):
        """
        From (1, 12):

        [x, y, z, ..., v]

        to (12, num_new_candidate_keypoints)

        [
            [x, x, x, ...],
            [y, y, y, ...],
            [z, z, z, ...],
            ...
            [v, v, v, ...]
        ]

        """
        return np.tile(T_C_W_flat, (num_new_candidate_keypoints, 1)).T

    def initialize_state(
        self,
        p_I_keypoints_initial: np.ndarray,
        p_W_landmarks_initial: np.ndarray,
    ):
        """
        S1 = (keypoints,landmarks,C1,F1,T1)

        keypoints: Keypoints2D: Current keypoints.
        landmarks: Landmarks3D: Current landmarks.
        candidate_keypoints: CandidateKeypoints2D: Current candidate keypoints.
        F1: np.ndarray(2, M): First track of current candidate keypoints.
        T1: np.ndarray(12, M): Camera poses at first track of current candidate keypoints.
        """
        keypoints = Keypoints2D(p_I_keypoints_initial)
        landmarks = Landmarks3D(array=p_W_landmarks_initial)
        candidate_keypoints = CandidateKeypoints2D(np.zeros((2, 0), dtype=np.int32))
        F1 = np.zeros((2, 0), dtype=np.int32)
        T1 = np.zeros((12, 0), dtype=np.int32)

        return keypoints, landmarks, candidate_keypoints, F1, T1

    def get_T_C_W_flat(self, T_C_W):
        """
        From (3, 4):

        r11 r12 r13 tx
        r21 r22 r23 ty
        r31 r32 r33 tz

        to (1, 12):

        r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
        """
        return T_C_W.flatten()
