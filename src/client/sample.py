import torch
import time
from tqdm import tqdm
from client.rollout import Rollout
from client.client_interface import ClientInterface
from filters.butterworth import ButterworthFilter
import numpy as np
from config import PRECOMPUTED_MEANS, PRECOMPUTED_STDS, INITIAL_ACTION
from typing import Optional

    
def check_overturned(conditions: list[float]) -> bool:
    overturned, *_ = conditions
    return overturned
    

def compute_actions(
        model: torch.nn.Module,
        state: torch.Tensor,
        filter: ButterworthFilter,
        noise: float = 0.3,
        mean: torch.Tensor = PRECOMPUTED_MEANS,
        std: torch.Tensor = PRECOMPUTED_STDS
) -> list[float]:
    # norm_state = (state - mean) / std
    norm_state = state
    true_action = model(norm_state).numpy()[0, 0]
    action_noise = np.random.normal(0, noise, size=true_action.shape)
    true_action = true_action + action_noise
    true_action = np.clip(true_action, -1, 1) * 0.05
    filtered_action = filter(true_action)
    return true_action, filtered_action


def sample(
        model: torch.nn.Module,
        filter: ButterworthFilter,
        client: ClientInterface,
        num_steps: int = 100,
        interval: float = 0.05,
        noise: float = 0.3,
        weight_perturbation: float = 0.0,
        initial_state: Optional[torch.Tensor] = None,
        initial_action: Optional[torch.Tensor] = INITIAL_ACTION
    ) -> Rollout:
    client.reset()
    torch.set_grad_enabled(False)
    model.perturb_actor(
        weight_perturbation_size=weight_perturbation
    )
    if initial_action is None:
        initial_action = torch.tensor(INITIAL_ACTION)
    true_action = torch.tensor(initial_action)
    filtered_action = filter(true_action)
    state, conditions = client.send_data(filtered_action)
    if initial_state is not None:
        state = initial_state
    rollout = Rollout(
        states=[],
        actions=[],
        times=[],
        conditions=[]
    )
    current_time = time.time()
    rollout.append(state, true_action, current_time, conditions)
    last_conditions = conditions
    for i in tqdm(range(num_steps - 1)):
        current_time = time.time()
        true_action, filtered_action = compute_actions(
            model=model,
            state=state,
            filter=filter,
            noise=noise,
        )
        # NOTE: the state, actions stored here are related as the
        # action resulting from the state (not the state resulting
        # from the action)
        state = state.numpy()
        rollout.append(state, true_action, current_time, last_conditions)
        state, conditions = client.send_data(filtered_action)
        if check_overturned(last_conditions):
            break

        last_conditions = conditions
        
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
    filter.reset()
    client.reset()
    torch.set_grad_enabled(False)
    model.perturb_actor(
        weight_perturbation_size=0.0
    )
    true_action = torch.tensor(INITIAL_ACTION)
    filtered_action = filter(true_action)
    state, conditions = client.send_data(filtered_action)
    current_time = time.time()
    last_conditions = conditions
    for i in tqdm(range(num_steps - 1)):
        current_time = time.time()
        true_action, filtered_action = compute_actions(
            model=model,
            state=state,
            filter=filter,
            noise=0.0,
        )
        state, conditions = client.send_data(filtered_action)
        if check_overturned(last_conditions):
            break

        last_conditions = conditions
        
        elapsed_time = time.time() - current_time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)
    
    print(f'Final state: {state}')
    print(f'Final action: {true_action}')
    return state, true_action
