import dotenv
dotenv.load_dotenv()

import os
import click
import logging

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option('--port', type=int, default=8000)
@click.pass_context
def cli(ctx, debug, port):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug

    if debug:
        from calibration.mock_client import MockClient as Client
    else:
        from client.client import Client

    host = os.getenv("HOST")
    port = port if port else int(os.getenv("POST"))
    client = Client(
        host=host,
        port=port
    )
    ctx.obj['client'] = client


@cli.command()
@click.pass_context
def plot_base_readings(ctx):
    from calibration.plot_sense_readings import plot_base_sense_readings as plot_base_sense_readings_func
    client = ctx.obj['client']
    client.connect()
    plot_base_sense_readings_func(client)
    client.close()


@cli.command()
@click.pass_context
def plot_pitch_roll_readings(ctx):
    from calibration.plot_sense_readings import plot_pitch_roll_readings as plot_pitch_roll_readings_func
    client = ctx.obj['client']
    client.connect()
    plot_pitch_roll_readings_func(client)
    client.close()


# @cli.command()
# @click.pass_context
# def plot_v_readings(ctx):
#     from calibration.plot_sense_readings import plot_v_readings as plot_v_readings_func
#     client = ctx.obj['client']
#     client.connect()
#     plot_v_readings_func(client)
#     client.close()


@cli.command()
@click.pass_context
def plot_v_readings(ctx):
    from calibration.plot_sense_readings import plot_v_readings as plot_v_readings_func
    client = ctx.obj['client']
    client.connect()
    plot_v_readings_func(client)
    client.close()


@cli.command()
@click.pass_context
def measure_mpu_offsets(ctx):
    from calibration.measurement import measure_mpu_offsets as measure_mpu_offsets_func
    client = ctx.obj['client']
    client.connect()
    measure_mpu_offsets_func(client)
    client.close()


@cli.command()
@click.pass_context
def plot_readings(ctx):
    from calibration.plot_sense_readings import plot_readings as plot_readings_func
    client = ctx.obj['client']
    client.connect()
    plot_readings_func(client)
    client.close()


if __name__ == "__main__":
    cli()
