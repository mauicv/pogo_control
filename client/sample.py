import torch
import time
from dataclasses import dataclass
from client.client import Client
from filters.butterworth import ButterworthFilter
import numpy as np


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
        self.states.append(state)
        self.actions.append(action)
        self.times.append(time)


def sample(
        model: torch.nn.Module,
        filter: ButterworthFilter,
        client: Client,
        num_steps: int = 100,
        interval: float = 0.1,
        noise: float = 0.3
    ) -> Rollout:
    torch.set_grad_enabled(False)
    action = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    action = filter(action)
    rollout = Rollout(states=[], actions=[], times=[])
    current_time = time.time()
    for _ in range(num_steps):
        state = client.send_data(action)
        state = torch.tensor(state)
        action = model(state, deterministic=True).numpy()
        action = action + np.random.normal(0, noise, size=action.shape)
        action = np.clip(action, -1, 1)
        action = filter(action)
        current_time = time.time()
        # NOTE: the state, actions stored here are related as the
        # action resulting from the state (not the state resulting
        # from the action)
        rollout.append(state, action, current_time)
        state = client.send_data(action)

        elapsed_time = time.time() - current_time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)

    return rollout
