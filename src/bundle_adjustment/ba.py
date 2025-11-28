from pathlib import Path

HERE = Path(__file__).parent
BA_DATA_FILENAME = HERE / ".." / "ba_data" / "ba_data.txt"


def fun(x):
    """
    Compute residuals.

    x contains the camera parameters and 3D point positions to optimize.

    typical x:
        x = [
            camera_0_params,
            camera_1_params,
            ...,
            camera_N_params,
            point_0_params,
            point_1_params,
            ...,
            point_M_params
        ]
    where:
        camera_i_params = [
            R1,
            R2,
            R3,
            t1,
            t2,
            t3,
        ]
        R: rotations params (axis-angle)
        t: translation params

        point_j_params = [
            X,
            Y,
            Z
        ]



    Args:
        - x np.ndarray(num_cameras * num_3d_points,)

    Returns:
        - residuals np.ndarray(m,)
    """
    pass


def read_data():
    with open(BA_DATA_FILENAME, "r") as f:
        num_keyframes = int(f.readline())
        for _ in range(num_keyframes):
            num_points = int(f.readline())
            pass


def main():
    pass


if __name__ == "__main__":
    main()
