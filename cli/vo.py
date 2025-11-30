from typing import Annotated, List, Optional

import typer

from src.config import settings
from src.enums import Dataset, Plot
from src.initialize import initialize
from src.io.ba_exporter import BAExporter
from src.plotting.visualizer import Visualizer
from src.utils.data_reader import KittiDataReader, MalagaDataReader, ParkingDataReader
from src.vo import VOPipeline


def run(
    dataset: Annotated[Dataset, typer.Option()],
    plot: Annotated[List[Plot], typer.Option()] = [
        Plot.KEYPOINTS,
        Plot.LANDMARKS,
        Plot.REPROJECTION_ERRORS,
        Plot.SCALE_DRIFT,
        Plot.TRAJECTORY,
    ],
    num_images: Annotated[Optional[int], typer.Option()] = None,
) -> None:
    match dataset:
        case Dataset.PARKING:
            DataReader = ParkingDataReader
            dataset_settings = settings.dataset.parking
        case Dataset.MALAGA:
            DataReader = MalagaDataReader
            dataset_settings = settings.dataset.malaga
        case Dataset.KITTI:
            DataReader = KittiDataReader
            dataset_settings = settings.dataset.kitti

    NUM_IMAGES = num_images or dataset_settings.num_images

    image_0 = DataReader.read_image(id=0)
    image_1 = DataReader.read_image(id=dataset_settings.initialization_second_image_id)
    p1_I_keypoints, _, p_W_landmarks = initialize(
        image_0, image_1, K=dataset_settings.k
    )

    plot_trajectory = Plot.TRAJECTORY in plot
    plot_scale_drift = Plot.SCALE_DRIFT in plot
    camera_positions_ground_truth = (
        DataReader.read_trajectory() if (plot_trajectory or plot_scale_drift) else None
    )
    images = DataReader.read_imgs(end_id=NUM_IMAGES)
    visualizer = Visualizer(
        plot_keypoints=Plot.KEYPOINTS in plot,
        plot_landmarks=Plot.LANDMARKS in plot,
        plot_tracking=Plot.TRACKING in plot,
        plot_reprojection_errors=Plot.REPROJECTION_ERRORS in plot,
        plot_scale_drift=Plot.SCALE_DRIFT in plot,
        plot_trajectory=Plot.TRAJECTORY in plot,
    )
    VOPipeline(visualizer=visualizer, ba_exporter=BAExporter()).run(
        images=images,
        p_I_keypoints_initial=p1_I_keypoints,
        p_W_landmarks_initial=p_W_landmarks,
        K=dataset_settings.k,
        camera_positions_ground_truth=camera_positions_ground_truth,
    )
