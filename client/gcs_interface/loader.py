from google.cloud import storage
import torch
import json
from tqdm import tqdm
# import uuid


class DataLoader:
    def __init__(
            self,
            bucket,
            experiment_name,
            rollout_length=100,
            num_runs=0,
            state_dim=6,
            action_dim=8,
            reward_function=None
        ) -> None:
        if not reward_function:
            # TODO: check this is the correct function for forward velocity!
            reward_function = lambda x: x[:, :, [0]] # Forward velocity reward function
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
            states = torch.tensor(rollout_data['states'])
            self.state_buffer[run_index] = states
            actions = torch.tensor(rollout_data['actions'])
            self.action_buffer[run_index] = actions
            self.fetched_rollouts.add(rollout)
            self.rollout_ind += 1

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

        max_index = min(self.rollout_ind, self.num_runs)
        b_inds = torch.randint(0, max_index, (batch_size, 1))
        t_inds = []
        if from_start:
            t_inds = torch.zeros(batch_size, dtype=torch.int)
        else:
            t_inds.append(torch.randint(0, (self.rollout_length - num_time_steps), (1, )))
            t_inds = torch.cat(t_inds, dim=0)
        t_inds = t_inds[:, None] + torch.arange(0, num_time_steps)
        return (
            self.state_buffer[b_inds, t_inds].detach(),
            self.action_buffer[b_inds, t_inds].detach(),
            self.reward_function(self.state_buffer[b_inds, t_inds].detach())
        )
