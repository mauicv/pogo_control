import torch
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from config import PRECOMPUTED_MEANS, PRECOMPUTED_STDS
from storage.reward import default_standing_reward, default_velocity_reward


def make_mask(conditions):
    # if timesteps are the same and not overturned set mask to 0 else 1
    detection_ts = [condition[-1] for condition in conditions]
    overturned = [condition[0] for condition in conditions]
    mask = torch.ones(len(detection_ts), 1)
    for i in range(1, len(detection_ts)):
        if detection_ts[i] == detection_ts[i-1] and not overturned[i]:
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
            reward_type='walking',
            reward_function=default_velocity_reward,
            means=PRECOMPUTED_MEANS,
            stds=PRECOMPUTED_STDS,
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
        self.means = means
        self.stds = stds

        self.reward_type = reward_type

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

    def _download_rollout(self, rollout_name):
        with self.bucket.blob(rollout_name).open('r') as f:
            rollout_data = json.load(f)
        return rollout_name, rollout_data
    
    def _download_rollouts(self, rollout_names):
        with ThreadPoolExecutor(max_workers=8) as executor:
            rollout_data_list = list(tqdm(
                executor.map(self._download_rollout, rollout_names),
                total=len(rollout_names),
                desc="Downloading rollouts"
            ))
        return rollout_data_list
    
    def _process_rollout(self, rollout_data, rollout_name):
        run_index = self.rollout_ind % self.num_runs
        end_index = rollout_data['end_index']
        conditions = rollout_data['conditions']
        states = torch.tensor(rollout_data['states'])
        # Note that the reward funciton is applied to the unnormalized states!
        rewards = self.reward_function(states, conditions)
        if self.means is not None and self.stds is not None:
            states = (states - self.means) / self.stds
        self.state_buffer[run_index][:end_index+1] = states
        actions = torch.tensor(rollout_data['actions'])
        self.action_buffer[run_index][:end_index+1] = actions
        self.reward_buffer[run_index][:end_index+1] = rewards
        self.dropout_mask[run_index][:end_index+1] = 1
        if self.reward_type == 'walking':
            self.dropout_mask[run_index][:end_index+1] = make_mask(conditions)
        if end_index < self.num_time_steps:
            for i in range(end_index + 1, self.num_time_steps + 1):
                # pad the rollout with the last state
                self.state_buffer[run_index][i] = self.state_buffer[run_index][end_index]
                self.action_buffer[run_index][i] = self.action_buffer[run_index][end_index]
                self.reward_buffer[run_index][i] = self.reward_buffer[run_index][end_index]
                self.dropout_mask[run_index][i] = self.dropout_mask[run_index][end_index]
            end_index = self.num_time_steps
        self.end_index[run_index] = end_index + 1
        self.fetched_rollouts.add(rollout_name)
        self.rollout_ind += 1

    def fetch_all_rollouts(self):
        rollout_data_list = self._download_rollouts(self.indexed_rollouts)
        for rollout_name, rollout_data in tqdm(rollout_data_list):
            self._process_rollout(rollout_data, rollout_name)

    def fetch_new_rollouts(self):
        missing_rollouts = self.indexed_rollouts - self.fetched_rollouts

        if len(missing_rollouts) > self.num_runs:
            missing_rollouts = list(missing_rollouts)
            missing_rollouts.sort(key=lambda x: int(x.split('/')[-1].split('-')[0]))
            missing_rollouts = missing_rollouts[-self.num_runs:]

        for rollout in tqdm(missing_rollouts):
            rollout_name, rollout_data = self._download_rollout(rollout)
            self._process_rollout(rollout_data, rollout_name)

    def compute_rollout_rewards(self, num_rollouts=10):
        rollout_rewards = []
        for i in range(self.rollout_ind - num_rollouts, self.rollout_ind):
            rewards = self.reward_buffer[[i % self.num_runs]]
            rollout_rewards.append(rewards.sum())
        return torch.tensor(rollout_rewards).mean()

    def sample(
            self,
            batch_size=None,
            num_time_steps=None
        ):
        """Sample a batch of data from the buffer.

        args:
            batch_size: int, optional
                The number of samples to return.
            num_time_steps: int, optional
                The number of time steps to sample.
        """
        if not batch_size:
            batch_size = self.batch_size
        if not num_time_steps:
            num_time_steps = self.num_time_steps
        else:
            # clamp num_time_steps to the maximum number of time steps in the rollouts
            print(f'Clamping num_time_steps to {self.num_time_steps}')
            num_time_steps = min(num_time_steps, self.num_time_steps)

        valid_lengths = self.end_index - num_time_steps
        b_inds = torch.multinomial(valid_lengths.float(), batch_size, replacement=True)
        end_inds = self.end_index[b_inds]
        t_inds = []
        for end_ind in end_inds:
            t_ind = torch.randint(0, int(end_ind - num_time_steps), (1, ))
            t_inds.append(t_ind)
        t_inds = torch.cat(t_inds, dim=0)
        t_inds = t_inds[:, None] + torch.arange(0, num_time_steps)

        return (
            self.state_buffer[b_inds, t_inds].detach(),
            self.action_buffer[b_inds, t_inds].detach(),
            self.reward_buffer[b_inds, t_inds].detach(),
            self.dropout_mask[b_inds, t_inds].detach()
        )
