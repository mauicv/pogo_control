import logging
import dotenv
import click


dotenv.load_dotenv()

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


try:
    from peripherals.pogo import pogo
    cli.add_command(pogo)
except ImportError:
    pass

try:
    from peripherals.camera import camera
    cli.add_command(camera)
except ImportError:
    pass

try:
    from client import client
    cli.add_command(client)
except ImportError:
    pass

try:
    from readings import readings
    cli.add_command(readings)
except ImportError:
    pass

if __name__ == "__main__":
    cli()