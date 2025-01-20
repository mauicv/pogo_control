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
    from client.model import Actor, EncoderActor, DenseModel
        
    gcs = GCS_Interface(
        experiment_name=name,
        credentials='world-model-rl-01a513052a8a.json',
        bucket='pogo_wmrl',
        model_limits=4
    )

    state_dim = 6 + 8
    action_dim = 8

    encoder = DenseModel(
        depth=1,
        input_dim=state_dim,
        hidden_dim=256,
        output_dim=256 * 32,
    )

    actor = Actor(
        input_dim=256 * 32,
        output_dim=action_dim,
        bound=1,
    )

    model = EncoderActor(
        encoder=encoder,
        actor=actor,
        num_latent=256,
        num_cat=32
    )

    gcs.model.upload_model(model)


@cli.command()
@click.option('--num-steps', type=int, default=250)
@click.option('--interval', type=float, default=0.1)
@click.option('--noise', type=float, default=0.0)
@click.option('--weight-perturbation', type=float, default=0.01)
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
        random_model=random_model,
        test=test,
        weight_perturbation=weight_perturbation
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