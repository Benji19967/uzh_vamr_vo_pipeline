import typer

from src.cli import images

# from housing.config import settings


# CSV_FILEPATH = settings.HOMEGATE_CSV_FILEPATH

vo = typer.Typer(no_args_is_help=True, add_completion=False)

vo.add_typer(images.images, name="images")


if __name__ == "__main__":
    vo()
