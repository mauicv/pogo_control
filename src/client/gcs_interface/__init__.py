from google.cloud import storage
from client.gcs_interface.model import GCSModel
from client.gcs_interface.rollout import GCSRollout
from client.gcs_interface.loader import DataLoader
import uuid


class GCS_Interface:
    def __init__(
            self,
            experiment_name,
            credentials='world-model-rl-01a513052a8a.json',
            bucket='pogo_wmrl',
            model_limits=25,
            num_runs=0,
            rollout_length=100,
        ) -> None:
        client = storage.Client.from_service_account_json(credentials)
        self.bucket = client.bucket(bucket)
        self.model = GCSModel(
            self.bucket,
            model_limits=model_limits,
            experiment_name=experiment_name
        )
        self.rollout = GCSRollout(
            self.bucket,
            experiment_name=experiment_name
        )
        self.loader = DataLoader(
            self.bucket,
            experiment_name=experiment_name,
            num_runs=num_runs,
            rollout_length=rollout_length
        )