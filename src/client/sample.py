import torch
import time
from tqdm import tqdm
from client.rollout import Rollout
from client.client_interface import ClientInterface
from filters.butterworth import ButterworthFilter
import numpy as np
from config import PRECOMPUTED_MEANS, PRECOMPUTED_STDS, INITIAL_ACTION, INITIAL_POSITION, ALT_INITIAL_POSITION
from typing import Optional
from client.noise import LinearSegmentNoiseND, SquareWaveND


INITIAL_POSITION = ALT_INITIAL_POSITION

    
def check_overturned(conditions: list[float]) -> bool:
    overturned, *_ = conditions
    return overturned
    

def compute_actions(
        model: torch.nn.Module,
        state: torch.Tensor,
        filter: ButterworthFilter,
        noise_generator: LinearSegmentNoiseND,
        mean: torch.Tensor = PRECOMPUTED_MEANS,
        std: torch.Tensor = PRECOMPUTED_STDS,
        kp: float = 0.0,
) -> list[float]:
    norm_state = (state - mean) / std
    true_action = model(norm_state).numpy()[0, 0]
    action_noise = noise_generator()
    true_action = true_action + action_noise

    current_joint_posistions = state[0:8].numpy()
    neutral_joint_posistions = np.array(INITIAL_POSITION)
    neutral_error = current_joint_posistions - neutral_joint_posistions
    neutral_action = -kp * neutral_error * np.abs(neutral_error)
    true_action = true_action + neutral_action

    true_action = np.clip(true_action, -1, 1)
    filtered_action = filter(true_action * 0.1)
    return true_action, filtered_action, action_noise


def sample(
        model: torch.nn.Module,
        filter: ButterworthFilter,
        client: ClientInterface,
        num_steps: int = 100,
        interval: float = 0.05,
        noise: float = 0.3,
        weight_perturbation: float = 0.0,
        kp: float = 0.0,
    ) -> Rollout:
    client.reset()
    torch.set_grad_enabled(False)
    model.perturb_actor(
        weight_perturbation_size=weight_perturbation
    )
    noise_generator = LinearSegmentNoiseND(
        dim=8,
        steps=num_steps,
        noise_size=noise,
        num_interp_points=40
    )
    rollout = Rollout(
        states=[],
        actions=[],
        times=[],
        conditions=[],
        filtered_actions=[],
        noise=[]
    )
    current_time = time.time()
    for i in tqdm(range(num_steps - 1)):
        current_time = time.time()
        state, conditions = client.read_state()
        true_action, filtered_action, action_noise = compute_actions(
            model=model,
            state=state,
            filter=filter,
            noise_generator=noise_generator,
            kp=kp,
        )
        # NOTE: the state, actions stored here are related as the
        # action resulting from the state (not the state resulting
        # from the action)
        rollout.append(
            state.numpy(),
            true_action,
            filtered_action,
            action_noise,
            current_time,
            conditions
        )
        client.take_action(filtered_action)
        if check_overturned(conditions):
            break

        elapsed_time = time.time() - current_time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)
    
    rollout = client.post_process(rollout)
    return rollout


def deploy_model(
        model: torch.nn.Module,
        filter: ButterworthFilter,
        client: ClientInterface,
        num_steps: int = 15,
        interval: float = 0.05,
    ) -> Rollout:
    raise NotImplementedError
    # filter.reset()
    # client.reset()
    # torch.set_grad_enabled(False)
    # model.perturb_actor(
    #     weight_perturbation_size=0.0
    # )
    # true_action = torch.tensor(INITIAL_ACTION)
    # filtered_action = filter(true_action)
    # state, conditions = client.send_data(filtered_action)
    # current_time = time.time()
    # last_conditions = conditions
    # for i in tqdm(range(num_steps - 1)):
    #     current_time = time.time()
    #     true_action, filtered_action = compute_actions(
    #         model=model,
    #         state=state,
    #         filter=filter,
    #         noise=0.0,
    #     )
    #     state, conditions = client.send_data(filtered_action)
    #     if check_overturned(last_conditions):
    #         break

    #     last_conditions = conditions
        
    #     elapsed_time = time.time() - current_time
    #     if elapsed_time < interval:
    #         time.sleep(interval - elapsed_time)
    
    # print(f'Final state: {state}')
    # print(f'Final action: {true_action}')
    # return state, true_action


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


def test(
        filter: ButterworthFilter,
        client: ClientInterface,
        num_steps: int = 15,
        interval: float = 0.05,
    ) -> Rollout:
    filter.reset()
    client.reset()

    model = TestModel()

    noise_generator = SquareWaveND(
        freq=25.0,
        amplitude=1,
        dim=8
    )
    rollout = Rollout(
        states=[],
        actions=[],
        times=[],
        conditions=[],
        filtered_actions=[],
        noise=[]
    )
    current_time = time.time()
    for i in tqdm(range(num_steps - 1)):
        current_time = time.time()
        state, conditions = client.read_state()
        true_action, filtered_action, action_noise = compute_actions(
            model=model,
            state=state,
            filter=filter,
            noise_generator=noise_generator,
        )
        # NOTE: the state, actions stored here are related as the
        # action resulting from the state (not the state resulting
        # from the action)
        rollout.append(
            state.numpy(),
            true_action,
            filtered_action,
            action_noise,
            current_time,
            conditions
        )
        for i, (a, m) in enumerate(zip(filtered_action, [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
            filtered_action[i] = a * m
        # filtered_action = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        client.take_action(filtered_action)
        if check_overturned(conditions):
            break

        elapsed_time = time.time() - current_time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)
    
    rollout = client.post_process(rollout)
    return rollout
