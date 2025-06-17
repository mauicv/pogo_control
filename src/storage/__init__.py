from google.cloud import storage
from storage.model import GCSModel
from storage.rollout import GCSRollout
from storage.loader import DataLoader
from storage.reward import default_velocity_reward


class GCS_Interface:
    def __init__(
            self,
            experiment_name,
            model_name=None,
            credentials='world-model-rl-01a513052a8a.json',
            project_id='world-model-rl',
            bucket='pogo_wmrl',
            model_limits=25,
            num_runs=0,
            rollout_length=100,
            state_dim=14,
            action_dim=8,
            num_time_steps=25,
            reward_function=default_velocity_reward
        ) -> None:
        if credentials:
            client = storage.Client.from_service_account_json(credentials)
        elif project_id:
            client = storage.Client(project=project_id)

        if model_name is None:
            model_name = experiment_name

        self.bucket = client.bucket(bucket)
        self.model = GCSModel(
            self.bucket,
            model_limits=model_limits,
            experiment_name=model_name
        )
        self.rollout = GCSRollout(
            self.bucket,
            experiment_name=experiment_name
        )
        self.loader = DataLoader(
            self.bucket,
            experiment_name=experiment_name,
            num_runs=num_runs,
            rollout_length=rollout_length,
            state_dim=state_dim,
            action_dim=action_dim,
            num_time_steps=num_time_steps,
            reward_function=reward_function
        )
