import click
import logging
import dotenv
import os
import time
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    pass


@cli.command()
@click.option('--name', type=str, default='pogo_control')
def clean(name):
    from client.gcs_interface import GCS_Interface
    gcs = GCS_Interface(
        experiment_name=name,
        credentials='world-model-rl-01a513052a8a.json',
        bucket='pogo_wmrl',
        model_limits=4,
    )
    gcs.rollout.remove_all_rollouts()
    gcs.model.remove_old_models()


@cli.command()
def server():
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
    POST = int(os.getenv("POST"))

    def _handle_message(message):
        pogo.update_angle(message)
        time.sleep(0.08)
        return pogo.get_data()

    channel = Channel(host=HOST, port=POST)
    channel.serve(_handle_message)
    click.echo(f"Server running on {HOST}:{POST}")


@cli.command()
@click.option('--name', type=str, default='pogo_control')
def create(name):
    from client.gcs_interface import GCS_Interface
    from client.model import Actor
    gcs = GCS_Interface(
        experiment_name=name,
        credentials='world-model-rl-01a513052a8a.json',
        bucket='pogo_wmrl',
        model_limits=4
    )
    model = Actor(
        input_dim=6 + 8, # 6 mpu sensor + 8 servo motors
        output_dim=8,
        bound=1,
        num_layers=3
    )
    gcs.model.upload_model(model)


@cli.command()
@click.option('--num-steps', type=int, default=250)
@click.option('--interval', type=float, default=0.1)
@click.option('--noise', type=float, default=0.4)
@click.option('--consecutive-error-limit', type=int, default=10)
@click.option('--name', type=str, default='pogo_control')
@click.option('--random-model', is_flag=False)
def client(
        num_steps,
        interval,
        noise,
        consecutive_error_limit,
        name,
        random_model
    ): 
    from client.client import Client
    from filters.butterworth import ButterworthFilter
    from client.gcs_interface import GCS_Interface
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
    host = os.getenv("HOST")
    port = int(os.getenv("POST"))

    client = Client(
        host=host,
        port=port
    )
    client.connect()
    butterworth_filter = ButterworthFilter(
        order=4,
        cutoff=2.0,
        fs=50.0,
        num_components=8 # 8 servo motors
    )
    run_client(
        gcs,
        client,
        butterworth_filter,
        num_steps=num_steps,
        interval=interval,
        noise=noise,
        consecutive_error_limit=consecutive_error_limit,
        random_model=random_model
    )


@cli.command()
def reset():
    from client.client import Client
    from client.run import set_init_state

    host = os.getenv("HOST")
    port = int(os.getenv("POST"))

    client = Client(
        host=host,
        port=port
    )
    client.connect()
    INITIAL_POSITION = (-0.4, -0.4, 0.4, 0.4, -0.4, -0.4, 0.4, 0.4)
    set_init_state(client, INITIAL_POSITION)


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
    from server.servo_controller import PIGPIO_AsyncServoInterface
    import pigpio

    # pogo move-robot --back-right-top=0.2 --front-right-top=-0.3

    SERVO_PINMAP = {0:4, 1:18, 2:27, 3:10, 4:20, 5:19, 6:13, 7:6}
    INITIAL_POSITION = {
        0: front_right_top, # front right top
        1: front_right_bottom, # front right bottom
        2: -front_left_top, # front left top (reversed)
        3: -front_left_bottom, # front left bottom (reversed)
        4: back_right_top, # back right top
        5: back_right_bottom, # back right bottom
        6: -back_left_top, # back left top (reversed)
        7: -back_left_bottom # back left bottom (reversed)
    }

    pigpio = pigpio.pi()
    servo = PIGPIO_AsyncServoInterface(
        SERVO_PINMAP,
        INITIAL_POSITION,
        pigpio=pigpio,
        update_interval=0.01,
        pid_kp=0.1,
        pid_ki=0.01,
        pid_kd=0.001,
    )

    time.sleep(3)
    # servo.update_angle()


if __name__ == "__main__":
    cli()