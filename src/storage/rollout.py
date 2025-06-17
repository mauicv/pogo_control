import json
import uuid
from typing import Optional

class GCSRollout:
    def __init__(
            self,
            bucket,
            experiment_name,
        ) -> None:
        self.bucket = bucket
        self.experiment_name = experiment_name

    def upload_rollout(self, rollout, name: Optional[str] = None):
        if name is None:
            name = str(uuid.uuid4())
        blob_name = f"{self.experiment_name}/rollouts/{name}.json"
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(rollout))

    def remove_all_rollouts(self):
        blobs = self.bucket.list_blobs(
            prefix=f'{self.experiment_name}/rollouts'
        )
        for blob in blobs:
            blob.delete()