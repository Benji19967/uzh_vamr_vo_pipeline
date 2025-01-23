from pathlib import Path

import typer

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
