import click
import logging

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")

@cli.command()
def server():
    from server.channel import Channel
    from server.mpu6050_interface import MPU6050Interface
    from server.piggpio_servo_interface import PIGPIO_ServoInterface

    SERVO_PINMAP = {1:4, 2:18, 3:27, 4:10, 5:20, 6:19, 7:13, 8:6}
    HOST = "192.168.0.27"
    POST = 8000

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
def client():
    from client.client import Client
    from filters.low_pass import LowPassFilter
    from filters.butterworth import ButterworthFilter
    from client.gcs_interface import GCS_Interface
    from client.sample import sample
    import torch
    torch.set_grad_enabled(False)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s'
    )
    gcs = GCS_Interface(
        credentials='world-model-rl-01a513052a8a.json',
        bucket='pogo_wmrl',
        model_limits=4,
    )
    gcs.model.load_model()
    client = Client(
        host='192.168.0.27',
        port=8000
    )
    client.connect()
    # low_pass_filter = LowPassFilter(
    #     alpha=0.85,
    #     num_components=6
    # )
    butterworth_filter = ButterworthFilter(
        order=2,
        cutoff=5.0,
        fs=50.0,
        num_components=6
    )
    rollout = sample(gcs.model.model, butterworth_filter, client)
    print(rollout)

if __name__ == "__main__":
    cli()