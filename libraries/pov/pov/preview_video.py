import click
from pov import Camera

@click.command()
def preview():
    cam = Camera()
    cam.video_preview()

if __name__ == '__main__':
    preview()