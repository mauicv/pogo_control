import json
import uuid


class GCSRollout:
    def __init__(
            self,
            bucket,
            experiment_name,
        ) -> None:
        self.bucket = bucket
        self.experiment_name = experiment_name

    def upload_rollout(self, rollout, model_version):
        rollout_index = str(uuid.uuid4())
        blob_name = f"{self.experiment_name}/rollouts/{model_version}-{rollout_index}.json"
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(rollout))

    def remove_all_rollouts(self):
        blobs = self.bucket.list_blobs(
            prefix=f'{self.experiment_name}/rollouts'
        )
        for blob in blobs:
            blob.delete()