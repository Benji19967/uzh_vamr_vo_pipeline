import typer

from cli import images
from cli.vo import vo

cli = typer.Typer(no_args_is_help=True, add_completion=False)

cli.add_typer(images.images, name="images")
cli.add_typer(vo, name="vo")


if __name__ == "__main__":
    cli()
