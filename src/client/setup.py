import logging
import os

logger = logging.getLogger(__name__)


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
    # from client.multi_client import MultiClientInterface
    from networking_utils.client import Client
    from filters.butterworth import ButterworthFilter
    from filters.identity import IdentityFilter
    from storage import GCS_Interface
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
    # camera_host = os.getenv("CAMERA_HOST")
    # camera_port = int(os.getenv("CAMERA_POST"))
    # client = MultiClientInterface(
    #     pogo_host=pogo_host,
    #     pogo_port=pogo_port,
    #     camera_host=camera_host,
    #     camera_port=camera_port
    # )
    pogo_host = os.getenv("POGO_HOST")
    pogo_port = int(os.getenv("POGO_POST"))
    client = Client(
        host=pogo_host,
        port=pogo_port
    )
    client.connect()

    filter = ButterworthFilter(
        order=5,
        cutoff=12.0,
        fs=50.0,
        num_components=8 # 8 servo motors
    )
    # filter = IdentityFilter()
    run_client(
        gcs,
        client,
        filter,
        num_steps=num_steps,
        interval=interval,
        noise_perturbation_range=noise_range,
        weight_perturbation_range=weight_range,
        consecutive_error_limit=consecutive_error_limit,
        random_model=random_model,
        test=test
    )