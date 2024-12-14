import click
import logging
import dotenv
import os
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    pass


@cli.command()
def clean():
    from client.gcs_interface import GCS_Interface
    gcs = GCS_Interface(
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
    from server.piggpio_servo_interface import PIGPIO_ServoInterface

    SERVO_PINMAP = {1:4, 2:18, 3:27, 4:10, 5:20, 6:19, 7:13, 8:6}
    HOST = os.getenv("HOST")
    POST = int(os.getenv("POST"))

    mpu = MPU6050Interface()
    servo = PIGPIO_ServoInterface(SERVO_PINMAP)
    servo.update_angle([0.0] * 8)

    def _handle_message(message):
        servo.update_angle(message)
        data = mpu.get_data()
        return data

    channel = Channel(host=HOST, port=POST)
    channel.serve(_handle_message)
    click.echo(f"Server running on {HOST}:{POST}")


@cli.command()
@click.option('--num-steps', type=int, default=100)
@click.option('--interval', type=float, default=0.1)
@click.option('--consecutive-error-limit', type=int, default=10)
def client(num_steps, interval, consecutive_error_limit):
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
        order=2,
        cutoff=5.0,
        fs=50.0,
        num_components=6
    )
    run_client(
        gcs,
        client,
        butterworth_filter,
        num_steps=num_steps,
        interval=interval,
        consecutive_error_limit=consecutive_error_limit
    )


if __name__ == "__main__":
    cli()