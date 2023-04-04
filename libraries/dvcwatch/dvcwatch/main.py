import os
import subprocess
import sys


def main(directory_to_watch, file_extension):
    for root, _, files in os.walk(directory_to_watch):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                subprocess.run(["dvc", "add", file_path])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: dvc_auto_add.py [directory_to_watch] [file_extension]")
        sys.exit(1)

    directory_to_watch = sys.argv[1]
    file_extension = sys.argv[2]

    if not os.path.isdir(directory_to_watch):
        print(f"Directory '{directory_to_watch}' does not exist")
        sys.exit(1)

    main(directory_to_watch, file_extension)
