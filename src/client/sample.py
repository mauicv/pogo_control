import torch
import time
from tqdm import tqdm
from client.rollout import Rollout
from client.client_interface import ClientInterface
from filters.butterworth import ButterworthFilter
import numpy as np
from config import PRECOMPUTED_MEANS, PRECOMPUTED_STDS, INITIAL_POSITION
from typing import Optional
from client.noise import LinearSegmentNoiseND
    
def check_overturned(conditions: list[float]) -> bool:
    overturned, *_ = conditions
    return overturned
    

def compute_actions(
        model: torch.nn.Module,
        state: torch.Tensor,
        filter: ButterworthFilter,
        noise_generator: Optional[LinearSegmentNoiseND] = None,
        mean: torch.Tensor = PRECOMPUTED_MEANS,
        std: torch.Tensor = PRECOMPUTED_STDS,
        kp: float = 0.0,
) -> list[float]:
    norm_state = (state - mean) / std
    true_action = model(norm_state).numpy()[0, 0]
    action_noise = noise_generator() if noise_generator is not None else 0.0
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
    model.reset()
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
        num_steps: int = 500,
        interval: float = 0.1,
    ) -> Rollout:
        client.reset()
        torch.set_grad_enabled(False)
        model.reset_to_original_actor()
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
            )
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
        client.reset()
        return rollout
