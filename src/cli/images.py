from pathlib import Path

import typer

from src.features.features import HarrisScores, Keypoints
from src.image import Image
from src.utils import utils

images = typer.Typer(no_args_is_help=True)


@images.command(no_args_is_help=True)
def view(
    filepaths: list[Path] = typer.Option(
        ...,
        "--path",
    ),
) -> None:
    for filepath in filepaths:
        img = utils.read_img(filepath=filepath)
        utils.show_img(img=img)


@images.command(no_args_is_help=True)
def shape(
    filepaths: list[Path] = typer.Option(
        ...,
        "--path",
    ),
) -> None:
    for filepath in filepaths:
        img = utils.read_img(filepath=filepath)
        print(img.shape)


@images.command(no_args_is_help=True)
def keypoints(
    filepaths: list[Path] = typer.Option(
        ...,
        "--path",
    ),
    downsample: bool = typer.Option(
        True,
        "--downsample",
    ),
) -> None:
    for filepath in filepaths:
        img = utils.read_img(filepath=filepath)
        if downsample:
            img = utils.downsample_img(img, height=img.shape[0], width=img.shape[1])
        image = Image(img=img, filepath=filepath)
        scores = HarrisScores(image=image).scores
        keypoints = Keypoints(image=image, scores=scores)
        keypoints.plot()
