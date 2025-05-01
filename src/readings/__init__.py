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
@click.option('--host', type=str, default='192.168.0.27')
@click.option('--port', type=int, default=8000)
@click.pass_context
def readings(ctx, debug, host, port):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug

    if debug:
        from readings.mock_client import MockClient as Client
    else:
        from networking_utils.client import Client

    client = Client(
        host=host,
        port=port
    )
    ctx.obj['client'] = client


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
