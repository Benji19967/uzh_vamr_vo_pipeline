from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

DPI = 227  # Mac M1


def plot_keypoints(
    ax,
    img: np.ndarray,
    p_I_keypoints: np.ndarray | list[np.ndarray],
    fmt: str | list[str] = "rx",
) -> None:
    # ax.clf()
    # ax.close()
    ax.imshow(img, cmap="gray")
    if isinstance(p_I_keypoints, list):
        for points, f in zip(p_I_keypoints, fmt):
            ax.plot(points[0, :], points[1, :], f, linewidth=2)
    else:
        ax.plot(p_I_keypoints[0, :], p_I_keypoints[1, :], fmt, linewidth=2)
    ax.axis("off")
    # plt.show()
    # plt.pause(0.05)


def plot_keypoints_cv2(
    img: np.ndarray,
    p_I_keypoints: np.ndarray | list[np.ndarray],
    size: int | list[int] = 5,
    color: tuple[int, int, int] | list[tuple[int, int, int]] = (255, 0, 0),
    thickness: int | list[int] = 1,
) -> None:
    """
    Significantly faster than plot_keypoints
    """
    img_out = img.copy()
    img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)  # type: ignore

    def draw_marker(p_I_keypoints, s, c, t):
        for x, y in p_I_keypoints.T:
            cv2.drawMarker(  # type: ignore
                img_out,
                (int(x), int(y)),
                c,
                markerType=cv2.MARKER_TILTED_CROSS,  # type: ignore
                markerSize=s,
                thickness=t,
                line_type=cv2.LINE_AA,  # type: ignore
            )

    if isinstance(p_I_keypoints, list):
        assert isinstance(size, list)
        assert isinstance(color, list)
        assert isinstance(thickness, list)
        assert len(p_I_keypoints) == len(size) == len(color) == len(thickness)

        for p_I_kps, s, c, t in zip(p_I_keypoints, size, color, thickness):
            draw_marker(p_I_kps, s, c, t)
    else:
        draw_marker(p_I_keypoints, size, color, thickness)

    cv2.imshow("Keypoints", img_out)  # type: ignore
    cv2.waitKey(10)  # type: ignore
    cv2.destroyAllWindows()  # type: ignore


def plot_landmarks_top_view(
    ax, p_W: np.ndarray, fmt="bx", camera_positions: list[np.ndarray] | None = None
) -> None:
    # ax.clf()
    # ax.close()
    ax.plot(p_W[0, :], p_W[2, :], fmt)
    if camera_positions is not None:
        for camera_position in camera_positions[-20:]:
            ax.plot(camera_position[0], camera_position[2], "rx")
    # plt.show()
    # plt.pause(0.05)


def plot_tracking(
    I0_keypoints: np.ndarray,
    I1_keypoints: np.ndarray,
    figsize_pixels_x: int | None = None,
    figsize_pixels_y: int | None = None,
):
    """
    Args:
        I0_keypoints np.ndarray(2,N): p=(x,y)
        I1_keypoints np.ndarray(2,N): p=(x,y)
        figsize_pixels_x: x size of figure in pixels
        figsize_pixels_y: y size of figure in pixels
    """
    x_from = I0_keypoints[0]
    x_to = I1_keypoints[0]
    y_from = I0_keypoints[1]
    y_to = I1_keypoints[1]

    if figsize_pixels_x and figsize_pixels_y:
        plt.figure(figsize=(figsize_pixels_x / DPI, figsize_pixels_y / DPI), dpi=DPI)
        ax = plt.gca()
        ax.set_xlim([0, figsize_pixels_x + 1])  # type: ignore
        ax.set_ylim([0, figsize_pixels_y + 1])  # type: ignore
    plt.gca().invert_yaxis()  # because p=(x, y) of keypoints are given for origin at top left corner  # type: ignore
    for i in range(x_from.shape[0]):
        plt.plot([x_from[i], x_to[i]], [y_from[i], y_to[i]], "g-", linewidth=1)
    # plt.show()
    plt.pause(0.05)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)  # type: ignore
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)  # type: ignore
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def drawCamera(
    ax,
    position,
    direction,
    length_scale=1,
    head_size=10,
    equal_axis=True,
    set_ax_limits=True,
):
    # Draws a camera consisting of arrows into a 3d Plot
    # ax            axes object, creates as follows
    #                   fig = plt.figure()
    #                   ax = fig.add_subplot(projection='3d')
    # position      np.array(3,) containing the camera position
    # direction     np.array(3,3) where each column corresponds to the [x, y, z]
    #               axis direction
    # length_scale  length scale: the arrows are drawn with length
    #               length_scale * direction
    # head_size     controls the size of the head of the arrows
    # equal_axis    boolean, if set to True (default) the axis are set to an
    #               equal aspect ratio
    # set_ax_limits if set to false, the plot box is not touched by the function

    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle="-|>", color="r")
    a = Arrow3D(
        [position[0], position[0] + length_scale * direction[0, 0]],
        [position[1], position[1] + length_scale * direction[1, 0]],
        [position[2], position[2] + length_scale * direction[2, 0]],
        **arrow_prop_dict,
    )
    ax.add_artist(a)
    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle="-|>", color="g")
    a = Arrow3D(
        [position[0], position[0] + length_scale * direction[0, 1]],
        [position[1], position[1] + length_scale * direction[1, 1]],
        [position[2], position[2] + length_scale * direction[2, 1]],
        **arrow_prop_dict,
    )
    ax.add_artist(a)
    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle="-|>", color="b")
    a = Arrow3D(
        [position[0], position[0] + length_scale * direction[0, 2]],
        [position[1], position[1] + length_scale * direction[1, 2]],
        [position[2], position[2] + length_scale * direction[2, 2]],
        **arrow_prop_dict,
    )
    ax.add_artist(a)

    if not set_ax_limits:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.set_xlim([min(xlim[0], position[0]), max(xlim[1], position[0])])
    ax.set_ylim([min(ylim[0], position[1]), max(ylim[1], position[1])])
    ax.set_zlim([min(zlim[0], position[2]), max(zlim[1], position[2])])

    # This sets the aspect ratio to 'equal'
    if equal_axis:
        ax.set_box_aspect(
            (np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim()))
        )


def normalize(v):
    return v / np.linalg.norm(v)


def plot_angle(x1, x2, K, R1, t1, R2, t2):
    # Camera intrinsics (identity matrix for visualization simplicity)
    # K = np.eye(3)

    # Matched keypoints in image 1 and image 2
    # x1 = np.array([100, 120])
    # x2 = np.array([130, 115])

    # Camera poses
    # R1 = np.eye(3)
    # t1 = np.array([0, 0, 0])  # Camera 1 at origin

    # R2 = np.eye(3)
    # t2 = np.array([1, 0, 0])  # Camera 2 translated 1 unit to the right

    # Convert to homogeneous coordinates
    x1_h = np.append(x1, 1)
    x2_h = np.append(x2, 1)

    # Step 1: Get bearing vectors in camera frame
    b1_c = normalize(np.linalg.inv(K) @ x1_h)
    b2_c = normalize(np.linalg.inv(K) @ x2_h)

    # Step 2: Rotate bearing vectors to world frame
    b1_w = normalize(R1.T @ b1_c)
    b2_w = normalize(R2.T @ b2_c)

    # Step 3: Compute camera centers
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2

    # Step 4: Plot
    fig = plt.figure(figsize=(10, 7))
    ax: Axes3D = cast(Axes3D, fig.add_subplot(111, projection="3d"))

    # Camera centers
    ax.scatter(*C1, color="blue", label="Camera 1")
    ax.scatter(*C2, color="green", label="Camera 2")  # type: ignore

    # Bearing vectors (rays)
    scale = 3
    ax.quiver(*C1, *(b1_w * scale), color="blue", arrow_length_ratio=0.1)
    ax.quiver(*C2, *(b2_w * scale), color="green", arrow_length_ratio=0.1)

    # Labels
    ax.text(*C1, "Cam1", color="blue")  # type: ignore
    ax.text(*C2, "Cam2", color="green")  # type: ignore

    # Compute angle between the two world-frame rays
    dot = np.dot(b1_w, b2_w)
    angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    # Plot formatting
    ax.set_title(f"Angle Between Bearing Vectors: {angle_deg:.2f}°")
    ax.set_xlim3d([-1, 5])
    ax.set_ylim3d([-2, 2])
    ax.set_zlim3d([-1, 5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return angle_deg
