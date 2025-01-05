from client.client import Client
from filters.butterworth import ButterworthFilter
from client.gcs_interface import GCS_Interface
from client.sample import sample
from client.model import Actor
import torch
from time import sleep, time
torch.set_grad_enabled(False)


def wait_for_model(gcs: GCS_Interface):
    while True:
        try:
            return gcs.model.load_model()
        except Exception as e:
            print(e)
            sleep(1)
            continue


def set_init_state(
        client: Client,
        target_position: list[float]=(-0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4)
    ):
    client.send_data(target_position)
    sleep(4)


def run_client(
        gcs: GCS_Interface,
        client: Client,
        butterworth_filter: ButterworthFilter,
        num_steps: int = 100,
        interval: float = 0.1,
        consecutive_error_limit: int = 3,
        noise: float = 0.3,
        random_model: bool = False
    ):
    if not random_model:
        model = wait_for_model(gcs)
    consecutive_errors = 0
    count = 0
    time_start = time()
    while True:
        count += 1
        try:
            print('==========================================')
            print(f'Count: {count}')
            print(f'Time: {(time() - time_start)/60:.2f} minutes')

            print('Sampling rollout')
            if random_model:
                print('Randomizing model')
                model = Actor(
                    input_dim=6 + 8, # 6 mpu sensor + 8 servo motors
                    output_dim=8,
                    bound=1,
                    num_layers=3
                )
            sample_start = time()
            rollout = sample(
                model,
                butterworth_filter,
                client,
                num_steps,
                interval,
                noise
            )
            print(f'Sampling time: {time() - sample_start:.2f} seconds')
            print('Re-initalizing')
            set_init_state(client)
            print('Uploading rollout')
            gcs.rollout.upload_rollout(
                rollout.to_dict(),
                gcs.model.version
            )
            if not random_model:
                model = gcs.model.load_model()
            consecutive_errors = 0
        except Exception as e:
            # print(e)
            raise e
            consecutive_errors += 1
            if consecutive_errors > consecutive_error_limit:
                print("Too many consecutive errors, exiting")
                raise e
            sleep(1)
            continue
