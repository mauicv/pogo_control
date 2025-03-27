import os
import click
import logging

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option('--host', type=str, default=None)
@click.option('--port', type=int, default=8000)
@click.pass_context
def camera(ctx, debug, host, port):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    ctx.obj['HOST'] = host if host else os.getenv("HOST")
    ctx.obj['POST'] = port if port else int(os.getenv("POST"))


@camera.command()
@click.pass_context
@click.option('--camera-matrix-file', type=str, default='camera_calibration_files/picamera-module-3.json')
def start(ctx, camera_matrix_file):
    from peripherals.camera.setup import setup_camera_sensor
    setup_camera_sensor(
        host=ctx.obj['HOST'],
        port=ctx.obj['POST'],
        camera_matrix_file=camera_matrix_file,
    )


@camera.command()
@click.option('--interval', type=int, default=1)
@click.option('--number_of_images', type=int, default=12)
def take_calibration_images(interval, number_of_images):
    from peripherals.camera.functions import take_calibration_images as take_calibration_images_func
    take_calibration_images_func(interval, number_of_images)


@camera.command()
def generate_calibration_file():
    from peripherals.camera.functions import calibrate_camera as calibrate_camera_func
    calibrate_camera_func()


@camera.command()
def capture():
    from peripherals.camera.functions import take_camera_image as take_camera_image_func
    take_camera_image_func()
