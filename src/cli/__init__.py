import click
import logging
import dotenv
import os
import time
import numpy as np
import json
import pprint

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

c_params = {
    "camera_matrix": np.array([
        [2085.4159146805323, 0.0, 1148.4264238731403],
        [0.0, 2081.546003047629, 790.0377744644074],
        [0.0, 0.0, 1.0]
    ]),
    "dist_coeff": np.array([[
        0.020513211878008017,
        -0.38588468516917407,
        0.005600115703970824,
        0.012111562135710205,
        2.66038039807393
    ]])
}


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    pass


@cli.command()
@click.option('--name', type=str, default='pogo_control')
def clean(name):
    from storage import GCS_Interface
    gcs = GCS_Interface(
        experiment_name=name,
        credentials='world-model-rl-01a513052a8a.json',
        bucket='pogo_wmrl',
        model_limits=4,
    )
    gcs.rollout.remove_all_rollouts()
    gcs.model.remove_old_models()


@cli.command()
@click.option('--port', type=int, default=8000)
@click.option('--camera-matrix-file', type=str, default='camera_calibration_files/picamera-module-3.json')
def camera_sensor_server(port, camera_matrix_file):
    from server.channel import Channel
    from server.pose_sensor import PoseSensor
    from server.camera import Picamera2Camera as Camera

    with open(camera_matrix_file, 'r') as f:
        c_params = json.load(f)
        pprint.pprint(c_params)

    camera_matrix = np.array(c_params['camera_matrix'])
    dist_coeff = np.array(c_params['dist_coeff'])

    camera = Camera(
        input_source="main",
        height=1536,
        width=2048,
        camera_matrix=camera_matrix,
        dist_coeff=dist_coeff,
    )
    pose_sensor = PoseSensor(camera=camera, update_interval=0.01)
    HOST = os.getenv("HOST")
    POST = port if port else int(os.getenv("POST"))

    def _handle_message(message):
        # time.sleep(0.08)
        data = pose_sensor.get_data()
        return data

    channel = Channel(host=HOST, port=POST)
    channel.serve(_handle_message)


@cli.command()
@click.option('--port', type=int, default=8000)
def pogo_server(port):
    from server.channel import Channel
    from server.pogo import Pogo
    import pigpio
    from server.mpu6050 import mpu6050

    gpio = pigpio.pi()
    mpu = mpu6050(0x68)
    pogo = Pogo(
        gpio=gpio,
        mpu=mpu,
        update_interval=0.01,
    )
    HOST = os.getenv("HOST")
    POST = port if port else int(os.getenv("POST"))

    def _handle_message(message):
        pogo.update_angle(message)
        time.sleep(0.08)
        return pogo.get_data()

    channel = Channel(host=HOST, port=POST)
    channel.serve(_handle_message)


@cli.command()
@click.option('--port', type=int, default=8000)
def pogo_sensor_server(port):
    from server.channel import Channel
    from server.mpu6050 import mpu6050
    from server.pogo import SensorPogo

    mpu = mpu6050(0x68)
    sensor_pogo = SensorPogo(mpu=mpu, update_interval=0.01)
    HOST = os.getenv("HOST")
    POST = port if port else int(os.getenv("POST"))

    def _handle_message(message):
        # time.sleep(0.08)
        data = sensor_pogo.get_data()
        return data

    channel = Channel(host=HOST, port=POST)
    channel.serve(_handle_message)


@cli.command()
@click.option('--num-steps', type=int, default=250)
@click.option('--interval', type=float, default=0.1)
@click.option('--noise', type=float, default=0.3)
@click.option('--weight-perturbation', type=float, default=0.0)
@click.option('--consecutive-error-limit', type=int, default=10)
@click.option('--name', type=str, default='pogo_control')
@click.option('--random-model', is_flag=True)
@click.option('--test', is_flag=True)
def client(
        num_steps,
        interval,
        noise,
        weight_perturbation,
        consecutive_error_limit,
        name,
        random_model,
        test
    ): 
    # from client.multi_client import MultiClientInterface
    from client.client import Client
    from filters.butterworth import ButterworthFilter
    from filters.identity import IdentityFilter
    from storage import GCS_Interface
    from client.run import run_client
    import torch
    torch.set_grad_enabled(False)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s'
    )
    gcs = GCS_Interface(
        experiment_name=name,
        credentials='world-model-rl-01a513052a8a.json',
        bucket='pogo_wmrl',
        model_limits=4
    )
    # camera_host = os.getenv("CAMERA_HOST")
    # camera_port = int(os.getenv("CAMERA_POST"))
    # client = MultiClientInterface(
    #     pogo_host=pogo_host,
    #     pogo_port=pogo_port,
    #     camera_host=camera_host,
    #     camera_port=camera_port
    # )
    pogo_host = os.getenv("POGO_HOST")
    pogo_port = int(os.getenv("POGO_POST"))
    client = Client(
        host=pogo_host,
        port=pogo_port
    )
    client.connect()

    filter = ButterworthFilter(
        order=5,
        cutoff=12.0,
        fs=50.0,
        num_components=8 # 8 servo motors
    )
    # filter = IdentityFilter()
    run_client(
        gcs,
        client,
        filter,
        num_steps=num_steps,
        interval=interval,
        noise=noise,
        consecutive_error_limit=consecutive_error_limit,
        random_model=random_model,
        test=test,
        weight_perturbation=weight_perturbation
    )


@cli.command()
@click.option('--front-left-bottom', type=float, default=0.0)
@click.option('--front-left-top', type=float, default=0.0)
@click.option('--front-right-bottom', type=float, default=0.0)
@click.option('--front-right-top', type=float, default=0.0)
@click.option('--back-left-bottom', type=float, default=0.0)
@click.option('--back-left-top', type=float, default=0.0)
@click.option('--back-right-bottom', type=float, default=0.0)
@click.option('--back-right-top', type=float, default=0.0)
def move_robot(front_left_bottom, front_left_top, front_right_bottom, front_right_top, back_left_bottom, back_left_top, back_right_bottom, back_right_top):
    """
    Move the robot to the given angles.

    example:
        pogo move-robot --front-left-bottom=0.4 --front-right-bottom=0.4 --back-right-bottom=0.4 --back-left-bottom=0.4 --front-left-top=-0.3 --front-right-top=-0.3 --back-right-top=-0.3 --back-left-top=-0.3
    """
    from server.pogo import Pogo
    import pigpio
    from server.mpu6050 import mpu6050

    gpio = pigpio.pi()
    mpu = mpu6050(0x68)
    pogo = Pogo(
        gpio=gpio,
        mpu=mpu,
        update_interval=0.01,
    )

    pogo.update_angle([
        front_right_top,
        front_right_bottom,
        front_left_top,
        front_left_bottom,
        back_right_top,
        back_right_bottom,
        back_left_top,
        back_left_bottom
    ])

    # pogo move-robot --back-right-top=0.2 --front-right-top=-0.3
    time.sleep(3)
    pogo.deinit()


@cli.command()
def sense():
    from server.pogo import Pogo
    import pigpio
    from server.mpu6050 import mpu6050

    gpio = pigpio.pi()
    mpu = mpu6050(0x68)
    pogo = Pogo(
        gpio=gpio,
        mpu=mpu,
        update_interval=0.01,
    )
    try:
        while True:
            data = pogo.get_data()
            print(data[0:3])
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        pogo.deinit()


if __name__ == "__main__":
    cli()