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
    from server.mpu6050_interface import MPU6050Interface
    # from server.piggpio_servo_interface import PIGPIO_ServoInterface
    from server.piggpio_async_servo_interface import PIGPIO_AsyncServoInterface
    import pigpio

    SERVO_PINMAP = {0:4, 1:18, 2:27, 3:10, 4:20, 5:19, 6:13, 7:6}
    INITIAL_POSITION = {0:-0.4, 1:-0.4, 2:0.4, 3:0.4, 4:-0.4, 5:-0.4, 6:0.4, 7:0.4}

    HOST = os.getenv("HOST")
    POST = int(os.getenv("POST"))

    pigpio = pigpio.pi()
    mpu = MPU6050Interface()
    servo = PIGPIO_AsyncServoInterface(
        SERVO_PINMAP,
        INITIAL_POSITION,
        pigpio=pigpio,
        update_interval=0.01,
        pid_kp=0.1,
        pid_ki=0.01,
        pid_kd=0.001,
    )

    def _handle_message(message):
        servo.update_angle(message)
        time.sleep(0.05)
        mpu_data = mpu.get_data()
        servo_data = servo.get_data()
        return mpu_data + servo_data

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
def client(num_steps, interval, noise, consecutive_error_limit, name):
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
        consecutive_error_limit=consecutive_error_limit
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


if __name__ == "__main__":
    cli()