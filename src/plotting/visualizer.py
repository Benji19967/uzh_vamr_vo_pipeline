import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.plotting.keypoints import draw_keypoints, plot_keypoints
from src.plotting.landmarks import plot_landmarks_top_view
from src.plotting.reprojection_errors import plot_reprojection_errors
from src.plotting.scale_drift import plot_scale_drift
from src.plotting.tracking import plot_tracking_cv2
from src.plotting.trajectory import plot_trajectory


class Visualizer:
    def __init__(
        self,
        plot_keypoints=True,
        plot_landmarks=True,
        plot_tracking=False,
        plot_reprojection_errors=True,
        plot_scale_drift=True,
        plot_trajectory=True,
    ):
        if plot_tracking and (plot_keypoints or plot_landmarks):
            raise ValueError("Cannot plot tracking with other plots")

        self._plot_keypoints = plot_keypoints
        self._plot_landmarks = plot_landmarks
        self._plot_tracking = plot_tracking
        self._plot_reprojection_errors = plot_reprojection_errors
        self._plot_scale_drift = plot_scale_drift
        self._plot_trajectory = plot_trajectory

    def keypoints_and_landmarks(self, P1, X1, C1, camera_positions, image_1):
        """Plot keypoints and/or landmarks

        Args:
            P1: np.ndarray(2, N): Current keypoints.
            X1: np.ndarray(3, N): Current landmarks.
            C1: np.ndarray(2, M): Current candidate keypoints.
            camera_positions List[np.ndarray(3,)]: camera position (x, y, z) at each frame
            image_1 np.ndarray: current frame
        """
        if self._plot_keypoints and self._plot_landmarks:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            img_out = draw_keypoints(
                img=image_1,
                p_I_keypoints=[P1, C1],
                marker_size=[5, 5],
                color=[(0, 0, 255), (0, 255, 0)],
                thickness=[1, 1],
            )
            ax1.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))  # type: ignore
            ax1.axis("off")
            ax1.set_title("Keypoints (red) / Candidate Keypoints (green)", fontsize=10)

            plot_landmarks_top_view(ax=ax2, p_W=X1, camera_positions=camera_positions)
            ax2.set_title(
                "Landmarks (blue) / Camera positions (red). Last 20 frames.", fontsize=8
            )

        elif self._plot_keypoints:
            plot_keypoints(
                img=image_1,
                p_I_keypoints=[P1, C1],
                marker_size=[5, 5],
                color=[(0, 0, 255), (0, 255, 0)],
                thickness=[1, 1],
            )
        elif self._plot_landmarks:
            fig, ax = plt.subplots()
            plot_landmarks_top_view(ax=ax, p_W=X1, camera_positions=camera_positions)
        plt.pause(0.05)
        plt.close()

    def tracking(self, C0, C1, image_0):
        if self._plot_tracking:
            plot_tracking_cv2(C0, C1, image_0.shape)

    def reprojection_errors(self, reprojection_errors: list[float]):
        if self._plot_reprojection_errors:
            plot_reprojection_errors(reprojection_errors)

    def scale_drift(
        self,
        camera_positions_estimated: list[np.ndarray],
        camera_positions_ground_truth: list[np.ndarray],
    ):
        if self._plot_scale_drift:
            plot_scale_drift(camera_positions_estimated, camera_positions_ground_truth)

    def trajectory(
        self,
        camera_positions_estimated: list[np.ndarray],
        camera_positions_ground_truth: list[np.ndarray] | None = None,
    ):
        if self._plot_trajectory:
            plot_trajectory(camera_positions_estimated, camera_positions_ground_truth)
