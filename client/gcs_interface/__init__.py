from google.cloud import storage
from client.gcs_interface.model import GCSModel
from client.gcs_interface.rollout import GCSRollout

class GCS_Interface:
    def __init__(
            self,
            credentials='world-model-rl-01a513052a8a.json',
            bucket='pogo_wmrl',
            model_limits=25,
            rollout_limits=100000,
        ) -> None:
        client = storage.Client.from_service_account_json(credentials)
        self.bucket = client.bucket(bucket)
        self.model = GCSModel(self.bucket, model_limits)
        self.rollout = GCSRollout(self.bucket, rollout_limits)
