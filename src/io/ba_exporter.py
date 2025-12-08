from pathlib import Path

import numpy as np

from src.structures.landmark_tracks import LandmarkTracks

HERE = Path(__file__).parent
BA_DATA_FILENAME = HERE / ".." / ".." / "ba_data" / "ba_data.txt"


class BAExporter:
    """
    Export camera poses, landmarks and landmark observations
    according to the bundle adjustment in the large (BAL) format,
    but each camera is only 6 params: rvec (Rodrigues) (3) and tvec (3).

    <num_cameras> <num_points> <num_observations>
    <camera_index_1> <point_index_1> <x_1> <y_1>
    ...
    <camera_index_num_observations> <point_index_num_observations> <x_num_observations> <y_num_observations>
    <camera_1>
    ...
    <camera_num_cameras>
    <point_1>
    ...
    <point_num_points>

    https://grail.cs.washington.edu/projects/bal/
    """

    def write(self, landmark_tracks: LandmarkTracks) -> None:
        poses = landmark_tracks._poses
        points = landmark_tracks._landmarks
        observations = landmark_tracks.get_observations()

        with open(BA_DATA_FILENAME, "w") as f:
            f.write(f"{len(poses)} {points.count} {len(observations)}\n")
            for o in observations:
                f.write(f"{o[0]} {o[1]}     {o[2]} {o[3]}\n")
            for _, pose in poses.items():
                np.savetxt(f, pose.rvec)
                np.savetxt(f, pose.tvec)
            points_flat = points.array.reshape(-1)
            np.savetxt(f, points_flat)
