import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

DPI = 227  # Mac M1


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
        ax.set_xlim([0, figsize_pixels_x + 1])
        ax.set_ylim([0, figsize_pixels_y + 1])
    plt.gca().invert_yaxis()  # because p=(x, y) of keypoints are given for origin at top left corner
    for i in range(x_from.shape[0]):
        plt.plot([x_from[i], x_to[i]], [y_from[i], y_to[i]], "g-", linewidth=1)
    plt.show()


def plot_landmarks_top_view(p_W: np.ndarray, fmt="bx") -> None:
    plt.clf()
    plt.close()
    plt.plot(p_W[0, :], p_W[2, :], fmt)
    plt.show()


def plot_keypoints(
    img: np.ndarray,
    p_I_keypoints: np.ndarray | list[np.ndarray],
    fmt: str | list[str] = "rx",
) -> None:
    plt.clf()
    plt.close()
    plt.imshow(img, cmap="gray")
    if isinstance(p_I_keypoints, list):
        for points, f in zip(p_I_keypoints, fmt):
            plt.plot(points[0, :], points[1, :], f, linewidth=2)
    else:
        plt.plot(p_I_keypoints[0, :], p_I_keypoints[1, :], fmt, linewidth=2)
    plt.axis("off")
    plt.show()


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
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
