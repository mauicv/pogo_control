from google.cloud import storage
import torch
import logging
logger = logging.getLogger(__name__)


def parse_version(blob_name):
    return int(blob_name.split('-')[1].split('.')[0])


class GCSModel:
    def __init__(
            self,
            bucket,
            model_limits=25,
        ) -> None:
        self.bucket = bucket
        self.model = None
        self.model_limits = model_limits
        self.version = self.get_latest_model_version()

    def remove_all_models(self):
        blobs = self.bucket.list_blobs(prefix='actor')
        for blob in blobs:
            blob.delete()

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
            logger.info(f'versions: {sorted(versions)}')
            if len(versions) == 0:
                version = None
            else:
                version = max(versions)
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            version = 0
        return version

    def upload_model(self, model):
        if self.version is None: self.version = self.get_latest_model_version()
        logger.info(f"'self.version', {self.version}")

        try:
            if self.version is None:
                logger.info('no version found, setting to 0')
                self.version = 0
            else:
                self.version += 1
            blob_name = f"actor/actor-{self.version}.pt"
            blob = self.bucket.blob(blob_name)
            with blob.open("wb", ignore_flush=True) as f:
                torch.save(model, f)
            logger.info(f'uploaded model version {self.version}')
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            raise e

    def load_model(self):
        remote_version = self.get_latest_model_version()
        if remote_version is None:
            logger.info('no remote model found')
            return None
        if remote_version is None:
            # no model remote or otherwise
            self.version = remote_version
        if (remote_version != self.version) or \
                (self.model is None and remote_version is not None):
            # if the remote version is different from the local version
            # or if there is no local model but there is a remote model
            blob_name = f"actor/actor-{remote_version}.pt"
            blob = self.bucket.blob(blob_name)
            self.model = torch.load(blob.open("rb"))
            self.version = remote_version
            logger.info(f'loaded model version {self.version}')
        else:
            logger.info(f'model version {self.version} already loaded')
        return self.model
