import os
import sys
import torch
import logging

logger = logging.getLogger(__name__)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage import GCS_Interface
from client.model import Actor


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s'
    )

    gcs = GCS_Interface(
        credentials='world-model-rl-01a513052a8a.json',
        bucket='pogo_wmrl',
        model_limits=4,
    )
    model = Actor(
        input_dim=6,
        output_dim=8,
        bound=1,
        num_layers=2
    )
