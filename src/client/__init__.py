import click
import logging
import dotenv
import os

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option('--host', type=str, default=None)
@click.option('--port', type=int, default=8000)
@click.pass_context
def client(ctx, debug, host, port):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    ctx.obj['HOST'] = host if host else os.getenv("HOST")
    ctx.obj['POST'] = port if port else int(os.getenv("POST"))


@client.command()
@click.option('--num-steps', type=int, default=250)
@click.option('--interval', type=float, default=0.1)
@click.option('--noise-range', nargs=2, type=float, default=(0.01, 0.5))
@click.option('--weight-range', nargs=2, type=float, default=(0.01, 0.02))
@click.option('--consecutive-error-limit', type=int, default=10)
@click.option('--name', type=str, default='pogo_control')
@click.option('--random-model', is_flag=True)
@click.option('--test', is_flag=True)
def client(
        num_steps,
        interval,
        noise_range,
        weight_range,
        consecutive_error_limit,
        name,
        random_model,
        test
    ): 
    from client.setup import client as client_setup
    client_setup(
        num_steps,
        interval,
        noise_range,
        weight_range,
        consecutive_error_limit,
        name,
        random_model,
        test
    )

