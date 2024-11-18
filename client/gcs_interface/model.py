from google.cloud import storage
import torch


def parse_version(blob_name):
    return int(blob_name.split('-')[1].split('.')[0])


class GCSModel:
    def __init__(
            self,
            bucket,
            model_limits=25,
        ) -> None:
        self.bucket = bucket
        self.version = None
        self.model = None
        self.model_limits = model_limits
        self.load_model()

    def remove_old_models(self):
        blobs = self.bucket.list_blobs(prefix='actor')
        filtered_blobs = [blob for blob in blobs if blob.name.endswith('.pt')]
        self.version = max([parse_version(blob.name) for blob in filtered_blobs])
        version_diff = self.version - self.model_limits
        filtered_blobs = [
            blob for blob in filtered_blobs
            if parse_version(blob.name) < version_diff
        ]
        for blob in filtered_blobs:
            blob.delete()

    def get_latest_model_version(self):
        try:
            blobs = self.bucket.list_blobs(prefix='actor')
            filtered_blobs = [blob for blob in blobs if blob.name.endswith('.pt')]
            versions = [parse_version(blob.name) for blob in filtered_blobs]
            print('[versions]', sorted(versions))
            version = max(versions)
        except Exception as e:
            print(f"Error getting latest model version: {e}")
            version = 0
        return version

    def upload_model(self, model):
        if self.version is None: self.get_latest_model_version()

        try:
            blob_name = f"actor/actor-{self.version + 1}.pt"
            blob = self.bucket.blob(blob_name)
            with blob.open("wb", ignore_flush=True) as f:
                torch.save(model, f)
            print(f'uploaded model version {self.version + 1}')
            self.version += 1
        except Exception as e:
            print(f"Error uploading model: {e}")
            raise e

    def load_model(self):
        remote_version = self.get_latest_model_version()
        if self.version is None or remote_version != self.version:
            blob_name = f"actor/actor-{remote_version}.pt"
            blob = self.bucket.blob(blob_name)
            self.model = torch.load(blob.open("rb"))
            self.version = remote_version
            print(f'loaded model version {self.version}')
        else:
            print(f'model version {self.version} already loaded')
        return self.model
