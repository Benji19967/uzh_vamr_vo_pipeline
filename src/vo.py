import logging

import numpy as np

from src.exceptions import FailedLocalizationError
from src.features import keypoints as kp
from src.io.ba_exporter import BAExporter
from src.localization.pnp_ransac_localization import pnp_ransac_localization_cv2
from src.mapping.reprojection_error import reprojection_error
from src.mapping.triangulate_landmarks import triangulate_landmarks
from src.plotting.visualizer import Visualizer
from src.structures.candidate_tracks import CandidateTracks
from src.structures.keypoints2D import CandidateKeypoints2D, Keypoints2D
from src.structures.landmark_tracks import LandmarkTracks
from src.structures.landmarks3D import Landmarks3D
from src.structures.pose import Pose
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
        self.candidate_tracks = CandidateTracks()
        self.landmark_tracks = LandmarkTracks()

        # TODO
        self.frame_id = 1

    def run(
        self,
        images: list[np.ndarray],
        keypoints_initial: Keypoints2D,
        landmarks_initial: Landmarks3D,
        K: np.ndarray,
        camera_positions_ground_truth: list[np.ndarray] | None = None,
    ):
        """
        Run a visual odometry pipeline on the images

        Args:
            - images list[np.ndarray]
            - keypoints_initial Keypoints2D
            - landmarks_initial: Landmarks3D
            - K np.ndarray(3, 3): camera matrix
        """
        self.landmark_tracks.add_landmarks(
            frame_id=0, landmarks=landmarks_initial, observations=keypoints_initial
        )

        camera_positions, reprojection_errors = self.process_frames(images, K)

        self.ba_exporter.write(self.landmark_tracks)

        self.visualizer.trajectory(camera_positions, camera_positions_ground_truth)
        self.visualizer.reprojection_errors(reprojection_errors)
        if self.visualizer._plot_scale_drift:
            assert camera_positions_ground_truth
            self.visualizer.scale_drift(camera_positions, camera_positions_ground_truth)

    def process_frames(self, images, K):
        camera_positions = []
        reprojection_errors = []
        for i, (img_0, img_1) in enumerate(zip(images, images[1:])):
            logger.debug(f"Iteration: {i}, Frame id: {self.frame_id}")

            # Track keypoints of landmarks from img_0 to img_1
            tracked_landmarks_keypoints, tracked_landmarks_mask = run_klt(
                img_0, img_1, self.landmark_tracks.get_active_keypoints()
            )
            self.landmark_tracks.add_frame_observations(
                frame_id=self.frame_id,
                observations=tracked_landmarks_keypoints,
                tracked_mask=tracked_landmarks_mask,
            )

            # Track candidate keypoints from img_0 to img_1
            candidate_keypoints = self.candidate_tracks.get_current_coords()
            candidate_keypoints_0 = candidate_keypoints
            candidate_keypoints, tracked_candidates_mask = run_klt(
                img_0, img_1, candidate_keypoints
            )
            candidate_keypoints_0.keep(tracked_candidates_mask)
            self.visualizer.tracking(
                candidate_keypoints_0.array, candidate_keypoints.array, img_0
            )
            self.candidate_tracks.update_tracks(
                candidate_keypoints, tracked_candidates_mask
            )
            logger.debug(
                f"After klt: landmarks: {self.landmark_tracks.num_landmarks}, candidate_keypoints: {candidate_keypoints.shape}"
            )

            # Localize: compute camera pose
            if (
                self.landmark_tracks.get_active_landmarks().count
                < MIN_NUM_LANDMARKS_FOR_LOCALIZATION
            ):
                raise ValueError("Not enough keypoints/landmarks for localization")
            try:
                pose, best_inlier_mask, camera_position = pnp_ransac_localization_cv2(
                    self.landmark_tracks.get_active_keypoints().array,
                    self.landmark_tracks.get_active_landmarks().array,
                    K,
                )
                self.landmark_tracks.keep(best_inlier_mask)
                self.landmark_tracks.add_frame_pose(frame_id=self.frame_id, pose=pose)
                camera_positions.append(camera_position)
                logger.debug(
                    f"After ransac: landmarks: {self.landmark_tracks.num_landmarks}, candidate_keypoints: {candidate_keypoints.shape}"
                )
                logger.debug(f"Pose:\n {pose.T_C_W}")
                logger.debug(f"Camera position: {camera_position.flatten()}")
            except FailedLocalizationError:
                logger.debug("Failed Ransac localization")
                continue

            # Map: add new landmarks
            if i % KEYFRAME_INTERVAL == 0:
                self.add_new_candidate_keypoints(img_1, pose)
                self.add_new_landmarks(pose, K)
            if candidate_keypoints.count > MAX_NUM_CANDIDATE_KEYPOINTS:
                self.candidate_tracks.keep_last(MAX_NUM_CANDIDATE_KEYPOINTS)

            # Evaluate results
            reproj_error = reprojection_error(
                self.landmark_tracks.get_active_landmarks().array_hom,
                self.landmark_tracks.get_active_keypoints().array,
                pose.T_C_W,
                K,
            )
            reprojection_errors.append(reproj_error)
            logger.debug(f"Reprojection error landmarks: {reproj_error}")

            self.visualizer.keypoints_and_landmarks(
                self.landmark_tracks.get_active_keypoints().array,
                self.landmark_tracks.get_active_landmarks().array,
                candidate_keypoints.array,
                camera_positions,
                img_1,
            )
            self.frame_id += 1

        return camera_positions, reprojection_errors

    def add_new_candidate_keypoints(
        self,
        img_1: np.ndarray,
        pose: Pose,
    ) -> None:
        """
        Add new candidate keypoints to the current set of keypoints.

        Args:
            img_1: np.ndarray: Current image.
            pose: Pose: Camera pose for the current image.
        """
        cp_new, _ = kp.find_keypoints(
            img_1,
            MAX_NUM_NEW_CANDIDATE_KEYPOINTS,
            exclude=[
                self.candidate_tracks.get_current_coords().array,
                self.landmark_tracks.get_active_keypoints().array,
            ],
        )
        self.candidate_tracks.add_candidate_keypoints(
            CandidateKeypoints2D(cp_new), pose
        )

    def add_new_landmarks(
        self,
        pose: Pose,
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
            pose: Pose: Camera pose for the current image.
            K: np.ndarray(3, 3): Camera intrinsic matrix.
        """
        assert self.candidate_tracks.get_current_coords().count > 0
        _, _, mask_to_triangulate = compute_bearing_angles_with_translation(
            self.candidate_tracks.get_initial_coords().array,
            self.candidate_tracks.get_current_coords().array,
            self.candidate_tracks.get_TCWs_at_intial_coords(),
            pose.T_C_W,
            K,
            MIN_ANGLE_TO_TRIANGULATE,
        )

        p_W_new_landmarks, mask_successful_triangulation = triangulate_landmarks(
            self.candidate_tracks.get_initial_coords().array,
            self.candidate_tracks.get_current_coords().array,
            self.candidate_tracks.get_TCWs_at_intial_coords(),
            pose.T_C_W,
            K,
            mask_to_triangulate,
            MAX_REPROJECTION_ERROR,
        )
        ckp_triangulated = self.candidate_tracks.get_current_coords().filtered(
            mask_successful_triangulation
        )
        logger.debug(f"Successful triangulation: {mask_successful_triangulation.sum()}")

        best_inlier_mask_ransac = np.full(mask_successful_triangulation.sum(), False)
        if (
            self.candidate_tracks.get_current_coords().count > 0
            and mask_successful_triangulation.sum() >= 4
        ):
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

            self.landmark_tracks.add_landmarks(
                frame_id=self.frame_id,
                landmarks=Landmarks3D(p_W_new_landmarks_inliers),
                observations=Keypoints2D(ckp_triangulated.array),
            )
            self.candidate_tracks.keep(~mask_new_landmarks)
