import subprocess
from pathlib import Path

import click


@click.command()
@click.argument("directory_to_watch", type=click.Path(exists=True, file_okay=False))
@click.argument("file_extension")
def main(directory_to_watch, file_extension):
    directory = Path(directory_to_watch)
    for file_path in directory.glob(f"**/*{file_extension}"):
        if file_path.is_file():
            dvc_file_path = file_path.with_suffix(f"{file_extension}.dvc")
            if not dvc_file_path.exists():
                subprocess.run(["dvc", "add", str(file_path)])


if __name__ == "__main__":
    main()
