import torch
import time
from tqdm import tqdm
from dataclasses import dataclass
from client.multi_client import MultiClientInterface
from filters.butterworth import ButterworthFilter
import numpy as np

INITIAL_POSITION = (-0.4, -0.4, 0.4, 0.4, -0.4, -0.4, 0.4, 0.4)

@dataclass
class Rollout:
    states: list[list[float]]
    actions: list[list[float]]
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

    def append(self, state, action, time, conditions):
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
        self.conditions.append(conditions)
        self.end_index = len(self.states) - 1


class ConditionCounter:
    def __init__(self, overturned_iteration_count_limit: int):
        self.overturned_iteration_count_limit = overturned_iteration_count_limit
        self.overturned_iteration_count = 0

    def update_check(self, conditions: list[float]) -> bool:
        (
            overturned,
            last_mpus6050_sample_ts,
            last_servo_set_ts,
            last_detection_ts
        ) = conditions

        if overturned:
            self.overturned_iteration_count += 1
        else:
            self.overturned_iteration_count = 0
        if self.overturned_iteration_count > self.overturned_iteration_count_limit:
            return True

        print(f"last_mpus6050_sample_ts: {last_mpus6050_sample_ts}")
        print(f"last_servo_set_ts: {last_servo_set_ts}")
        print(f"last_detection_ts: {last_detection_ts}")
        return False


def sample(
        model: torch.nn.Module,
        filter: ButterworthFilter,
        client: MultiClientInterface,
        num_steps: int = 100,
        interval: float = 0.1,
        noise: float = 0.3,
        weight_perturbation: float = 0.01,
    ) -> Rollout:
    filter.reset()
    torch.set_grad_enabled(False)
    model.perturb_actor(
        weight_perturbation_size=weight_perturbation
    )
    counter = ConditionCounter(
        overturned_iteration_count_limit=3,
        no_marker_count_limit=20
    )
    action = torch.tensor(INITIAL_POSITION)
    action = filter(action)
    servo_state, world_state, conditions = client.send_data(action)
    state = torch.tensor(servo_state + world_state)
    rollout = Rollout(
        states=[],
        actions=[],
        times=[],
        conditions=[]
    )
    current_time = time.time()
    for i in tqdm(range(num_steps)):
        current_time = time.time()
        true_action = model(state).numpy()[0, 0]
        action_noise = np.random.normal(0, noise, size=true_action.shape)
        true_action = true_action + action_noise
        true_action = np.clip(true_action, -1, 1)
        # NOTE: the state, actions stored here are related as the
        # action resulting from the state (not the state resulting
        # from the action)
        state = state.numpy()
        rollout.append(state, true_action, current_time, conditions)
        if counter.update_check(conditions):
            break
        filtered_action = filter(true_action)
        servo_state, world_state, conditions = client.send_data(filtered_action)
        state = torch.tensor(servo_state + world_state)

        elapsed_time = time.time() - current_time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)

    return rollout
