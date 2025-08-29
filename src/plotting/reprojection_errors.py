import matplotlib.pyplot as plt


def plot_reprojection_errors(reprojection_error: list[float]) -> None:
    """
    Plot the reprojection error over time/frames
    """
    plt.plot(reprojection_error, "b-", linewidth=2, label="Reprojection error")

    plt.xlabel("Frame id")
    plt.ylabel("Reprojection error in pixels")
    plt.title("Reprojection error of landmarks at each frame")
    plt.legend()
    plt.grid(True)
    plt.show()
