import torch
import time
from dataclasses import dataclass
from client.client import Client
from filters.butterworth import ButterworthFilter


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
        interval: float = 0.1
    ) -> Rollout:
    torch.set_grad_enabled(False)
    action = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    action = filter.filter(action)
    rollout = Rollout(states=[], actions=[], times=[])
    current_time = time.time()
    for _ in range(num_steps):
        state = client.send_data(action)
        state = torch.tensor(state)
        action = model(state, deterministic=True).numpy()
        action = filter.filter(action)
        state = client.send_data(action)
        rollout.append(state, action, time.time())
        elapsed_time = time.time() - current_time
        if elapsed_time > interval:
            current_time = time.time()
        else:
            time.sleep(interval - elapsed_time)

    return rollout
