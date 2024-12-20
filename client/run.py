from client.client import Client
from filters.butterworth import ButterworthFilter
from client.gcs_interface import GCS_Interface
from client.sample import sample
import torch
from time import sleep
torch.set_grad_enabled(False)


def run_client(
        gcs: GCS_Interface,
        client: Client,
        butterworth_filter: ButterworthFilter,
        num_steps: int = 100,
        interval: float = 0.1,
        consecutive_error_limit: int = 10,
        noise: float = 0.3
    ):

    model = wait_for_model(gcs)
    perform_rollouts(
        model,
        butterworth_filter,
        client,
        gcs,
        num_steps,
        interval,
        consecutive_error_limit,
        noise
    )


def perform_rollouts(
        model: torch.nn.Module,
        butterworth_filter: ButterworthFilter,
        client: Client,
        gcs: GCS_Interface,
        num_steps: int = 100,
        interval: float = 0.1,
        consecutive_error_limit: int = 10,
        noise: float = 0.3
    ):
    consecutive_errors = 0
    while True:
        try:
            rollout = sample(
                model,
                butterworth_filter,
                client,
                num_steps,
                interval,
                noise
            )
            gcs.rollout.upload_rollout(
                rollout.to_dict(),
                gcs.model.version
            )
            model = gcs.model.load_model()
            consecutive_errors = 0
        except Exception as e:
            print(e)
            consecutive_errors += 1
            if consecutive_errors > consecutive_error_limit:
                print("Too many consecutive errors, exiting")
                break
            sleep(1)
            continue


def wait_for_model(gcs: GCS_Interface):
    while True:
        try:
            return gcs.model.load_model()
        except Exception as e:
            print(e)
            sleep(1)
            continue
