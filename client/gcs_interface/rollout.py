from google.cloud import storage
import json
import uuid


class GCSRollout:
    def __init__(
            self,
            bucket,
        ) -> None:
        self.bucket = bucket
        self.experiment_id = str(uuid.uuid4())

    def upload_rollout(self, rollout, model_version):
        rollout_index = str(uuid.uuid4())
        blob_name = f"rollouts/{model_version}-{self.experiment_id}-{rollout_index}.json"
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(rollout))

    def remove_all_rollouts(self):
        blobs = self.bucket.list_blobs(prefix='rollouts')
        for blob in blobs:
            blob.delete()