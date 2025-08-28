import cv2
from matplotlib import pyplot as plt

from src.plotting.keypoints import draw_keypoints, plot_keypoints
from src.plotting.klt_tracking import plot_tracking
from src.plotting.landmarks import plot_landmarks_top_view


class Visualizer:
    def __init__(
        self,
        plot_keypoints=True,
        plot_landmarks=True,
        plot_tracking=False,
    ):
        self.plot_keypoints = plot_keypoints
        self.plot_landmarks = plot_landmarks
        self.plot_tracking = plot_tracking

    def plot_visualizations(
        self,
        P1,
        X1,
        C0,
        C1,
        camera_positions,
        image_0,
        image_1,
    ):
        """Plot

        Args:
            P1 (_type_): _description_
            X1 (_type_): _description_
            C0 (_type_): _description_
            C1 (_type_): _description_
            camera_positions (_type_): _description_
            image_0 (_type_): _description_
            image_1 (_type_): _description_
        """
        if self.plot_keypoints and self.plot_landmarks:
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

            plt.pause(0.05)
            plt.close()
        elif self.plot_keypoints:
            plot_keypoints(
                img=image_1,
                p_I_keypoints=[P1, C1],
                marker_size=[5, 5],
                color=[(0, 0, 255), (0, 255, 0)],
                thickness=[1, 1],
            )
        elif self.plot_landmarks:
            fig, ax = plt.subplots()
            plot_landmarks_top_view(ax=ax, p_W=X1, camera_positions=camera_positions)
            plt.pause(0.05)
            plt.close()

        if self.plot_tracking:
            plot_tracking(
                I0_keypoints=C0,
                I1_keypoints=C1,
                figsize_pixels_x=image_0.shape[1],
                figsize_pixels_y=image_0.shape[0],
            )
