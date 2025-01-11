import torch
import time
from tqdm import tqdm
from dataclasses import dataclass
from client.client import Client
from filters.butterworth import ButterworthFilter
import numpy as np

INITIAL_POSITION = (-0.4, -0.4, 0.4, 0.4, -0.4, -0.4, 0.4, 0.4)

@dataclass
class Rollout:
    states: list[list[float]]
    actions: list[list[float]]
    times: list[float]

    def to_dict(self):
        return {
            "states": self.states,
            "actions": self.actions,
            "times": self.times
        }

    def append(self, state, action, time):
        if isinstance(state, torch.Tensor):
            state = state.numpy().tolist()
        if isinstance(action, torch.Tensor):
            action = action.numpy().tolist()
        if isinstance(state, np.ndarray):
            state = state.tolist()
        if isinstance(action, np.ndarray):
            action = action.tolist()
        self.states.append(state)
        self.actions.append(action)
        self.times.append(time)


def linear_noise_warmup(j, end_i=25):
    if j < end_i:
        return j / end_i
    return 1


def sample(
        model: torch.nn.Module,
        filter: ButterworthFilter,
        client: Client,
        num_steps: int = 100,
        interval: float = 0.1,
        noise: float = 0.3,
        weight_perturbation: float = 0.01,
        noise_warmup: int = 25
    ) -> Rollout:
    filter.reset()
    torch.set_grad_enabled(False)
    model.perturb_actor(
        weight_perturbation_size=weight_perturbation
    )
    action = torch.tensor(INITIAL_POSITION)
    action = filter(action)
    state = client.send_data(action)
    state = torch.tensor(state)
    rollout = Rollout(states=[], actions=[], times=[])
    current_time = time.time()
    for i in tqdm(range(num_steps)):
        current_time = time.time()
        action = model(state).numpy()[0, 0]
        noise_scale = linear_noise_warmup(i, noise_warmup)
        action = action + np.random.normal(0, noise * noise_scale, size=action.shape)
        action = np.clip(action, -1, 1)
        action = filter(action)
        # NOTE: the state, actions stored here are related as the
        # action resulting from the state (not the state resulting
        # from the action)
        state = state.numpy()
        rollout.append(state, action, current_time)
        state = client.send_data(action)
        state = torch.tensor(state)

        elapsed_time = time.time() - current_time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)

    return rollout
