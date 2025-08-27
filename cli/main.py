import typer

from cli import images, vo

cli = typer.Typer(no_args_is_help=True, add_completion=False)


cli.command(no_args_is_help=True)(vo.run)
cli.add_typer(images.images, name="images")


if __name__ == "__main__":
    cli()
