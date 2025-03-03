from google.cloud import storage
import torch
import json
from tqdm import tqdm

def overturned_penalty(rewards, conditions):
    overturn_index = None
    for i, condition in enumerate(conditions):
        [overturned, *_] = condition
        if overturned:
            overturn_index = i
            break
    
    if overturn_index is None:
        return torch.zeros(len(rewards))

    overturned_reward = -10
    for i in range(overturn_index, 0, -1):
        rewards[i] += overturned_reward
        overturned_reward = overturned_reward * 0.75
    return torch.tensor(rewards)

def default_velocity_reward_function(states, conditions):
    rewards = []
    last_distance = None

    for condition in conditions:
        [*_, distance] = condition
        if last_distance is None:
            last_distance = distance
        distance_delta = distance - last_distance
        last_distance = distance
    rewards.append(-distance_delta)
    velocity_reward = torch.tanh(0.25 * torch.tensor(rewards))
    overturned_reward = overturned_penalty(rewards, conditions)
    return (velocity_reward + overturned_reward)[:, None]

def default_standing_reward(states, conditions):
    rewards = []
    for state in states:
        [
            front_right_top,
            front_right_bottom,
            front_left_top,
            front_left_bottom,
            back_right_top,
            back_right_bottom,
            back_left_top,
            back_left_bottom, 
            *_,
            roll,
            pitch,
        ] = state
        standing_reward = - (
            (front_left_bottom - 0.4)**2 +
            (front_right_bottom - 0.4)**2 +
            (back_right_bottom - 0.4)**2 +
            (back_left_bottom - 0.4)**2 +
            (front_left_top - -0.3)**2 +
            (front_right_top - -0.3)**2 +
            (back_right_top - -0.3)**2 +
            (back_left_top - -0.3)**2 +
            8*roll**2 +
            8*pitch**2
        ) / (8 + 2 * 8)
        rewards.append(standing_reward)
    standing_reward = torch.tensor(rewards)
    overturned_reward = overturned_penalty(rewards, conditions)
    return (standing_reward + overturned_reward)[:, None]

def make_mask(detection_ts):
    # if timesteps are the same set mask to 0 else 1
    mask = torch.ones(len(detection_ts), 1)
    for i in range(1, len(detection_ts)):
        if detection_ts[i] == detection_ts[i-1]:
            mask[i, 0] = 0
    return mask


class DataLoader:
    def __init__(
            self,
            bucket,
            experiment_name,
            rollout_length=100,
            num_runs=0,
            state_dim=14,
            action_dim=8,
            num_time_steps=25,
            reward_function=default_standing_reward
        ) -> None:
        self.num_time_steps = num_time_steps
        self.reward_function = reward_function
        self.bucket = bucket
        self.experiment_name = experiment_name
        self.rollout_ind = 0
        self.rollout_length = rollout_length
        self.num_runs = num_runs
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_buffer = torch.zeros(
            (self.num_runs, self.rollout_length, self.state_dim),
            dtype=torch.float32
        )

        self.action_buffer = torch.zeros(
            (self.num_runs, self.rollout_length, self.action_dim),
            dtype=torch.float32
        )

        self.reward_buffer = torch.zeros(
            (self.num_runs, self.rollout_length, 1),
            dtype=torch.float32
        )

        self.dropout_mask = torch.ones(
            (self.num_runs, self.rollout_length, 1),
            dtype=torch.float32
        )

        self.end_index = torch.zeros(
            self.num_runs,
            dtype=torch.int
        )

        self.fetched_rollouts = set()
        self.indexed_rollouts = set()

    def init_load(self):
        self.index_rollouts()

    def index_rollouts(self):
        blobs = self.bucket.list_blobs(
            prefix=f'{self.experiment_name}/rollouts/',
        )
        for blob in blobs:
            if blob.name.endswith('.json'):
                self.indexed_rollouts.add(blob.name)

    def fetch_rollouts(self):
        missing_rollouts = self.indexed_rollouts - self.fetched_rollouts

        if len(missing_rollouts) > self.num_runs:
            missing_rollouts = list(missing_rollouts)
            missing_rollouts.sort(key=lambda x: int(x.split('/')[-1].split('-')[0]))
            missing_rollouts = missing_rollouts[-self.num_runs:]

        for rollout in tqdm(missing_rollouts):
            run_index = self.rollout_ind % self.num_runs
            with self.bucket.blob(rollout).open('r') as f:
                rollout_data = json.load(f)
            end_index = rollout_data['end_index']
            if end_index < self.num_time_steps:
                # skip rollouts that are too short
                self.fetched_rollouts.add(rollout)
                continue
            conditions = rollout_data['conditions']
            states = torch.tensor(rollout_data['states'])
            self.state_buffer[run_index][:end_index+1] = states
            actions = torch.tensor(rollout_data['actions'])
            self.action_buffer[run_index][:end_index+1] = actions
            rewards = self.reward_function(rollout_data['states'], conditions)
            self.reward_buffer[run_index][:end_index+1] = rewards
            self.dropout_mask[run_index][:end_index+1] = 1
            detection_ts = [condition[-1] for condition in conditions]
            self.dropout_mask[run_index][:end_index+1] = make_mask(detection_ts)
            self.end_index[run_index] = end_index + 1
            self.fetched_rollouts.add(rollout)
            self.rollout_ind += 1

    def compute_rollout_rewards(self, num_rollouts=10):
        rollout_rewards = []
        for i in range(self.rollout_ind - num_rollouts, self.rollout_ind):
            rewards = self.reward_buffer[[i % self.num_runs]]
            rollout_rewards.append(rewards.mean())
        return torch.tensor(rollout_rewards).mean()

    def sample(
            self,
            batch_size=None,
            num_time_steps=None,
            from_start=False
        ):
        """Sample a batch of data from the buffer.

        args:
            batch_size: int, optional
                The number of samples to return.
            num_time_steps: int, optional
                The number of time steps to sample.
            from_start: bool, optional
                If True, sample from the start of the rollout.
        """
        if not batch_size:
            batch_size = self.batch_size
        if not num_time_steps:
            num_time_steps = self.num_time_steps
        else:
            # clamp num_time_steps to the maximum number of time steps in the rollouts
            print(f'Clamping num_time_steps to {self.num_time_steps}')
            num_time_steps = min(num_time_steps, self.num_time_steps)

        max_index = min(self.rollout_ind, self.num_runs)
        b_inds = torch.randint(0, max_index, (batch_size, 1))
        end_inds = self.end_index[b_inds]
        t_inds = []
        if from_start:
            t_inds = torch.zeros(batch_size, dtype=torch.int)
        else:
            for end_ind in end_inds:
                t_ind = torch.randint(0, (end_ind - num_time_steps), (1, ))
                t_inds.append(t_ind)
            t_inds = torch.cat(t_inds, dim=0)
        t_inds = t_inds[:, None] + torch.arange(0, num_time_steps)

        return (
            self.state_buffer[b_inds, t_inds].detach(),
            self.action_buffer[b_inds, t_inds].detach(),
            self.reward_buffer[b_inds, t_inds].detach(),
            self.dropout_mask[b_inds, t_inds].detach()
        )
