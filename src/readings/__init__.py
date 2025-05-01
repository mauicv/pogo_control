import dotenv
dotenv.load_dotenv()

import uuid
import click
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option('--camera-host', type=str, default='192.168.0.27')
@click.option('--camera-port', type=int, default=8000)
@click.option('--pogo-host', type=str, default='192.168.0.29')
@click.option('--pogo-port', type=int, default=8000)
@click.pass_context
def readings(ctx, debug, camera_host, camera_port, pogo_host, pogo_port):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug

    if debug:
        from readings.mock_client import MockClient as Client
    else:
        from networking_utils.client import Client

    client = Client(
        host=camera_host,
        port=camera_port
    )

    pogo_client = Client(
        host=pogo_host,
        port=pogo_port
    )

    ctx.obj['client'] = client
    ctx.obj['pogo_client'] = pogo_client


@readings.command()
@click.pass_context
def position(ctx):
    from readings.plot_pose_readings import plot_pose_readings as plot_pose_readings_func
    client = ctx.obj['client']
    client.connect()
    plot_pose_readings_func(client)
    client.close()


@readings.command()
@click.pass_context
@click.option('--num', type=int, default=5)
@click.option('--interval', type=float, default=0.2)
def capture(ctx, num, interval):
    client = ctx.obj['client']
    client.connect()
    start_time = time.time()
    for _ in tqdm(range(num)):
        last_time = time.time()
        data = client.send_data({'command': 'capture'})
        if time.time() - last_time < interval:
            time.sleep(interval - (time.time() - last_time))
    print(f'Time taken: {time.time() - start_time:.2f} seconds')
    print(f'Average time per capture: {(time.time() - start_time) / num:.2f} seconds')
    client.close()


@readings.command()
@click.pass_context
def pose(ctx):
    from readings.plot_pose_readings import plot_pose_readings as plot_readings_func
    client = ctx.obj['client']
    client.connect()
    pogo_client = ctx.obj['pogo_client']
    pogo_client.connect()
    plot_readings_func(client, pogo_client)
    client.close()
    pogo_client.close()

@readings.command()
@click.pass_context
def store(ctx):
    client = ctx.obj['client']
    client.connect()
    data = client.send_data({
        'command': 'store',
        'args': {
            'name': str(uuid.uuid4())
        }
    })
    client.close()


@readings.command()
@click.pass_context
def process(ctx):
    client = ctx.obj['client']
    client.connect()
    data = client.send_data({'command': 'process'})
    client.close()


@readings.command()
@click.pass_context
def reset(ctx):
    client = ctx.obj['client']
    client.connect()
    data = client.send_data({'command': 'reset'})
    print(data)
    client.close()


@readings.command()
@click.pass_context
def raw(ctx):
    from readings.plot_sense_readings import plot_base_sense_readings as plot_base_sense_readings_func
    client = ctx.obj['client']
    client.connect()
    plot_base_sense_readings_func(client)
    client.close()


@readings.command()
@click.pass_context
def mpu_offsets(ctx):
    from readings.measurement import measure_mpu_offsets as measure_mpu_offsets_func
    client = ctx.obj['client']
    client.connect()
    measure_mpu_offsets_func(client)
    client.close()
