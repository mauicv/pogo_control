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
import uuid
from client.sample import Rollout, deploy_model, test
from storage.reward import default_standing_reward, default_velocity_reward
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
    print(f'Loading model from {local_path}')
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

    # fig, axs = plt.subplots(2, 4)
    fig, axs = plt.subplot_mosaic(
        [
            ['action-1', 'action-2', 'action-3', 'action-4', 'reward', 'reward'],
            ['action-5', 'action-6', 'action-7', 'action-8', 'reward', 'reward'],
        ],
        layout='constrained'
    )

    action_labels = [
        "front_right_top",
        "front_right_bottom",
        "front_left_top",
        "front_left_bottom",
        "back_right_top",
        "back_right_bottom",
        "back_left_top",
        "back_left_bottom"
    ]

    conditions = rollout.conditions
    states = torch.tensor(rollout.states)
    posture_rewards = default_standing_reward(states, conditions)
    if len(posture_rewards) > 15:
        velocity_rewards = default_velocity_reward(states, conditions)
    else:
        velocity_rewards = torch.zeros(len(posture_rewards))
    for i, label in enumerate(action_labels):
        actions = np.array([action[i] * 0.1 for action in rollout.actions])
        filtered_actions = np.array([action[i] for action in rollout.filtered_actions])
        action_noise = np.array([noise[i] * 0.1 for noise in rollout.noise])

        axs[f'action-{i+1}'].set_title(label)
        axs[f'action-{i+1}'].plot(actions)
        axs[f'action-{i+1}'].plot(filtered_actions)
        axs[f'action-{i+1}'].plot(action_noise)
        axs[f'action-{i+1}'].set_ylim(-0.12, 0.12)
    axs[f'reward'].plot(posture_rewards, label='posture')
    axs[f'reward'].plot(velocity_rewards, label='velocity')
    axs[f'reward'].legend()
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
    client.set_servo_states(INITIAL_POSITION)
    soln_model = None
    if solution is not None:
        soln_model = load_local_model(solution)
    deploy_model(soln_model, butterworth_filter, client)
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
        kp: float = 0.0,
    ):
    client.set_servo_states(INITIAL_POSITION)

    if not random_model:
        model = wait_for_model(gcs)
    
    count = 0
    time_start = time()
    while True:
        count += 1
        weight_perturbation = get_random_perturbation(weight_perturbation_range)
        noise = get_random_perturbation(noise_perturbation_range)
        rollout_name = str(uuid.uuid4())

        print('==========================================')
        print(f'Rollout name: {rollout_name}')
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
                kp,
            )
        except KeyboardInterrupt as e:
            print(f'Interrupted sampling rollout: {e}')
            client.set_servo_states(INITIAL_POSITION)
            butterworth_filter.reset()
            client.reset()
            raise e

        print(f'Rollout Sampling time: {time() - sample_start:.2f} seconds')
        plot_rollout(rollout)
        show_rollout_details(rollout)
        set_init_state(client, filter=butterworth_filter)

        accept = input('(A)ccept, (Q)uit, (S)ave-images')


        for opt in ['S', 's']:
            if opt in accept:
                client.save_images(
                    name=rollout_name
                )
                print(f'Saved rollout recording')
                break
                
        client.reset()

        if not test:
            for opt in ['A', 'a']:
                if opt in accept:
                    print('Uploading rollout')
                    gcs.rollout.upload_rollout(
                        rollout.to_dict(),
                        name=rollout_name
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
        client.set_servo_states(INITIAL_POSITION)


def run_test(
        client: ClientInterface,
        butterworth_filter: ButterworthFilter,
        num_steps: int = 100,
        interval: float = 0.05,
    ):
    client.set_servo_states(INITIAL_POSITION)
    rollout = test(
        butterworth_filter,
        client,
        num_steps,
        interval,
    )
    show_rollout_details(rollout)
    plot_rollout(rollout)
    client.set_servo_states(INITIAL_POSITION)
    return rollout