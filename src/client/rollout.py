import torch
from dataclasses import dataclass
import numpy as np


@dataclass
class Rollout:
    states: list[list[float]]
    actions: list[list[float]]
    filtered_actions: list[list[float]]
    noise: list[list[float]]
    times: list[float]
    conditions: list[list[float]]
    end_index: int = None

    def to_dict(self):
        return {
            "states": self.states,
            "actions": self.actions,
            "times": self.times,
            "conditions": self.conditions,
            "end_index": self.end_index
        }

    def append(self, state, action, filtered_action, noise, time, conditions):
        if isinstance(state, torch.Tensor):
            state = state.numpy().tolist()
        if isinstance(action, torch.Tensor):
            action = action.numpy().tolist()
        if isinstance(filtered_action, torch.Tensor):
            filtered_action = filtered_action.numpy().tolist()
        if isinstance(state, np.ndarray):
            state = state.tolist()
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(noise, np.ndarray):
            noise = noise.tolist()
        self.states.append(state)
        self.actions.append(action)
        self.filtered_actions.append(filtered_action)
        self.noise.append(noise)
        self.times.append(time)
        self.conditions.append(conditions)
        self.end_index = len(self.states) - 1
