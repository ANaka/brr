import ffmpeg
import click
from pathlib import Path
import fn

ROOT_DIRECTORY = '/home/naka/Videos/pov'
    
@click.command()
@click.option('--input_dir', '-i', default=None)
@click.option('--suffix', '-s', default='.jpg')
@click.option('--output', '-o', default=None)
@click.option('--framerate', '-r', default=35)
@click.option('--crf', '-c', default=25)
def make_timelapse(
    input_dir=None,
    output=None,
    suffix='.jpg',
    framerate=35,
    crf=25,
    ):
    if input_dir == None:
        plot_id = fn.get_current_plot_id()
        input_dir = Path(ROOT_DIRECTORY).joinpath(plot_id)
    else:
        input_dir = Path(input_dir)
    if output == None:
        output = input_dir.joinpath('movie.mp4')
    try:
        (
            ffmpeg
            .input(
                f'{input_dir}/*{suffix}', 
                pattern_type='glob', 
                framerate=framerate,
                pix_fmt='yuv420p',
                
            )
            .output(output.as_posix(), crf=crf)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    
if __name__ == '__main__':
    make_timelapse()