import logging
import dotenv
import click
from peripherals.pogo import pogo
from peripherals.camera import camera
from client import client
from readings import readings

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass

cli.add_command(pogo)
cli.add_command(camera)
cli.add_command(client)
cli.add_command(readings)

if __name__ == "__main__":
    cli()