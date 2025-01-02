import os
import sys
import torch
import logging

logger = logging.getLogger(__name__)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.gcs_interface import GCS_Interface
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
        experiment_name='test'
    )
    model = Actor(
        input_dim=6 + 8,
        output_dim=8,
        bound=1,
        num_layers=2
    )

    model = gcs.model.load_model()
    gcs.model.remove_old_models()

    t = torch.tensor([1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., ])
    print(t)
    print(model(t, deterministic=True))

    model = gcs.model.upload_model(model)
    gcs.model.remove_old_models()