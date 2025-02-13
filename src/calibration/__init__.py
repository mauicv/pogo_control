import dotenv
dotenv.load_dotenv()

import os
import click
import logging

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option('--host', type=str, default='192.168.0.23')
@click.option('--port', type=int, default=8000)
@click.pass_context
def cli(ctx, debug, host, port):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug

    if debug:
        from calibration.mock_client import MockClient as Client
    else:
        from client.client import Client

    client = Client(
        host=host,
        port=port
    )
    ctx.obj['client'] = client


@cli.command()
@click.pass_context
def plot_pose_readings(ctx):
    from calibration.plot_pose_readings import plot_pose_readings as plot_pose_readings_func
    client = ctx.obj['client']
    client.connect()
    data = client.send_data({})
    print(data)
    plot_pose_readings_func(client)
    client.close()



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
