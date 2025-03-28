import torch
import time
from tqdm import tqdm
from client.rollout import Rollout
from client.client_interface import StandingClientInterface, WalkingClientInterface
from filters.butterworth import ButterworthFilter
import numpy as np
from config import PRECOMPUTED_MEANS, PRECOMPUTED_STDS, INITIAL_POSITION

    
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
    norm_state = (state - mean) / std
    true_action = model(norm_state).numpy()[0, 0]
    action_noise = np.random.normal(0, noise, size=true_action.shape)
    true_action = true_action + action_noise
    true_action = np.clip(true_action, -1, 1)
    filtered_action = filter(true_action)
    return true_action, filtered_action


def sample(
        model: torch.nn.Module,
        filter: ButterworthFilter,
        client: StandingClientInterface | WalkingClientInterface,
        num_steps: int = 100,
        interval: float = 0.1,
        noise: float = 0.3,
        weight_perturbation: float = 0.0,
    ) -> Rollout:
    filter.reset()
    client.reset()
    torch.set_grad_enabled(False)
    model.perturb_actor(
        weight_perturbation_size=weight_perturbation
    )
    true_action = torch.tensor(INITIAL_POSITION)
    filtered_action = filter(true_action)
    state, conditions = client.send_data(filtered_action)
    rollout = Rollout(
        states=[],
        actions=[],
        times=[],
        conditions=[]
    )
    current_time = time.time()
    rollout.append(state, true_action, current_time, conditions)
    for i in tqdm(range(num_steps)):
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
        rollout.append(state, true_action, current_time, conditions)
        if check_overturned(conditions):
            break
        state, conditions = client.send_data(filtered_action)

        elapsed_time = time.time() - current_time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)
    
    rollout = client.post_process(rollout)
    client.reset()

    return rollout
