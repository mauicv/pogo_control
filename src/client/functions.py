from typing import Optional
from filters.butterworth import ButterworthFilter
from storage import GCS_Interface
from client.sample import sample
from client.model import Actor, EncoderActor, DenseModel
from client.client_interface import ClientInterface
import torch
from time import sleep, time
import random
import numpy as np
from client.sample import Rollout
from client.sample import deploy_model
torch.set_grad_enabled(False)
from config import INITIAL_POSITION


def wait_for_model(gcs: GCS_Interface):
    while True:
        try:
            return gcs.model.load_model()
        except Exception as e:
            print(e)
            sleep(1)
            continue

def load_local_model(solution: str):
    local_path = f"solutions/{solution}.pt"
    model = torch.load(
        local_path,
        map_location=torch.device('cpu')
    )
    return model


def set_init_state(
        client: ClientInterface,
        filter: Optional[ButterworthFilter]=None,
        soln_model: Optional[torch.nn.Module]=None,
        target_position: Optional[list[float]]=INITIAL_POSITION,
    ):
    if soln_model is not None:
        state, action = deploy_model(soln_model, filter, client)
        return state, action
    else:
        client.set_servo_states(target_position)
        return None, None

def create_random_model(
        state_dim: int = 8 + 6 + 2 + 2 + 2,
        action_dim: int = 8
    ):
    print('Randomizing model')
    encoder = DenseModel(
        depth=1,
        input_dim=state_dim,
        hidden_dim=256,
        output_dim=256 * 32,
    )

    actor = Actor(
        input_dim=256 * 32,
        output_dim=action_dim,
        bound=1,
    )

    return EncoderActor(
        encoder=encoder,
        actor=actor
    )

def get_random_perturbation(perturbation_range: tuple[float, float]):
    assert perturbation_range[0] <= perturbation_range[1]
    if perturbation_range[0] == perturbation_range[1]:
        return perturbation_range[0]
    else:
        return random.uniform(perturbation_range[0], perturbation_range[1])
    

def plot_rollout(rollout: Rollout):
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data from rollout
    actions = np.array([action[0] * 0.05 for action in rollout.actions])
    filtered_actions = np.array([action[0] for action in rollout.filtered_actions])
    action_noise = np.array([noise[0] * 0.05 for noise in rollout.noise])
    # Create subplots
    fig, axs = plt.subplots(1, 1)

    # Plot actions
    axs.plot(actions)
    axs.plot(filtered_actions)
    axs.plot(action_noise)
    axs.set_title('Actions')
    axs.set_xlabel('Time')
    axs.legend(['True', 'Filtered', 'Noise'])
    # Show plot
    plt.show()


def show_rollout_details(rollout: Rollout):
    conditions = np.array(rollout.conditions)
    rolled = conditions[:, 0]
    last_detection_ts = conditions[:, -1]
    print(f'Rollout rolled: {"True" if rolled.any() else "False"}')
    num_failed_detections = 0
    for i in range(1, len(last_detection_ts)):
        if last_detection_ts[i] == last_detection_ts[i-1]:
            num_failed_detections += 1
    print(f'Rollout num failed detections: {num_failed_detections}/{len(last_detection_ts)}')


def deploy_solution(
        gcs: GCS_Interface,
        client: ClientInterface,
        butterworth_filter: ButterworthFilter,
        solution: str
    ):
    set_init_state(client)
    soln_model = None
    if solution is not None:
        soln_model = load_local_model(solution)
    set_init_state(client=client, filter=butterworth_filter, soln_model=soln_model)
    set_init_state(client)


def run_training(
        gcs: GCS_Interface,
        client: ClientInterface,
        butterworth_filter: ButterworthFilter,
        num_steps: int = 100,
        interval: float = 0.05,
        noise_perturbation_range: tuple[float, float] = (0.00, 0.00),
        weight_perturbation_range: tuple[float, float] = (0.00, 0.00),
        random_model: bool = False,
        test: bool = False,
        pre_solution: Optional[str] = None
    ):
    set_init_state(client)
    soln_model = None
    if pre_solution is not None:
        soln_model = load_local_model(pre_solution)
    
    if not random_model:
        model = wait_for_model(gcs)
        # print(model)

    initial_state, initial_action = set_init_state(
        client=client,
        filter=butterworth_filter,
        soln_model=soln_model
    )
    
    count = 0
    time_start = time()
    while True:
        count += 1
        weight_perturbation = get_random_perturbation(weight_perturbation_range)
        noise = get_random_perturbation(noise_perturbation_range)

        print('==========================================')
        print(f'Count: {count}')
        print(f'Training Running Time: {(time() - time_start)/60:.2f} minutes')
        print(f'Weight perturbation: {weight_perturbation}')
        print(f'Noise: {noise}')

        print('Sampling rollout')
        if random_model:
            model = create_random_model(
                state_dim=2 * 8 + 6 + 2,
                action_dim=8
            )

        sample_start = time()
        try:
            rollout = sample(
                model,
                butterworth_filter,
                client,
                num_steps,
                interval,
                noise,
                weight_perturbation,
            )
        except KeyboardInterrupt as e:
            print(f'Interrupted sampling rollout: {e}')
            set_init_state(client)
            butterworth_filter.reset()
            raise e
        

        print(f'Rollout Sampling time: {time() - sample_start:.2f} seconds')
        # plot_rollout(rollout)
        show_rollout_details(rollout)
        set_init_state(client, filter=butterworth_filter)

        accept = input('(A)ccept, (Q)uit, (S)ave-images')


        for opt in ['S', 's']:
            if opt in accept:
                name = client.save_images()
                print(f'Saved images to {name}')
                break

        if not test:
            for opt in ['A', 'a']:
                if opt in accept:
                    print('Uploading rollout')
                    gcs.rollout.upload_rollout(
                        rollout.to_dict(),
                        gcs.model.version
                    )
                    break
        else:
            print('Skipping upload')
            
        quit_training = False
        for opt in ['Q', 'q']:
            if opt in accept:
                quit_training = True
                break

        if quit_training:
            break

        if not random_model:
            model = gcs.model.load_model()

        butterworth_filter.reset()

        initial_state, initial_action = set_init_state(
            client,
            filter=butterworth_filter,
            soln_model=soln_model
        )

        
