import click
import logging

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug


@cli.command()
@click.pass_context
@click.option('--interval', type=int, default=1)
@click.option('--number_of_images', type=int, default=12)
def take_calibration_images(ctx, interval, number_of_images):
    from calibration.functions import take_calibration_images as take_calibration_images_func
    take_calibration_images_func(interval, number_of_images)


@cli.command()
@click.pass_context
def calibrate(ctx):
    from calibration.functions import calibrate_camera as calibrate_camera_func
    calibrate_camera_func()


@cli.command()
@click.pass_context
def take_image(ctx):
    from calibration.functions import take_camera_image as take_camera_image_func
    take_camera_image_func()


if __name__ == "__main__":
    cli()
