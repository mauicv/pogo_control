import torch
from dataclasses import dataclass

@dataclass
class Rollout:
    states: list[list[float]]
    actions: list[list[float]]


def sample(model, filter, client, num_steps: int = 100) -> Rollout:
    torch.set_grad_enabled(False)
    action = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    action = filter.filter(action)
    states = []
    actions = []
    for _ in range(num_steps):
        state = client.send_data(action)
        state = torch.tensor(state)
        action = model(state, deterministic=True).numpy()
        action = filter.filter(action)
        state = client.send_data(action)
        states.append(state)
        actions.append(action)
    return Rollout(states, actions)
