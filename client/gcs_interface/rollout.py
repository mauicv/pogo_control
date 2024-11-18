from google.cloud import storage
import torch


def parse_version(blob_name):
    return int(blob_name.split('-')[1].split('.')[0])


class GCSRollout:
    def __init__(
            self,
            bucket,
            limits=100000,
        ) -> None:
        self.bucket = bucket
        self.limits = limits
        self.rollout_index = None

    def get_latest_rollout_index(self):
        pass
        # try:
        #     blobs = self.bucket.list_blobs(prefix='rollouts')
        #     filtered_blobs = [blob for blob in blobs if blob.name.endswith('.pt')]
        #     versions = [parse_version(blob.name) for blob in filtered_blobs]
        #     print('[versions]', sorted(versions))
        #     version = max(versions)
        # except Exception as e:
        #     print(f"Error getting latest model version: {e}")
        #     version = 0
        # return version

    def upload_rollout(self, rollout):
        pass

    def get_rollout(self, version):
        pass

    def remove_old_rollouts(self):
        pass

