import os
import sys
import logging

logger = logging.getLogger(__name__)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.sample import sample
from tests.mock_client import MockClient
from filters.butterworth import ButterworthFilter
from client.gcs_interface import GCS_Interface


def test_sample():
    gcs = GCS_Interface(
        credentials='world-model-rl-01a513052a8a.json',
        bucket='pogo_wmrl',
        model_limits=4,
    )
    model = gcs.model.load_model()
    client = MockClient(
        host="localhost",
        port=8000
    )
    filter = ButterworthFilter(
        order=2,
        cutoff=5.0,
        fs=50.0,
        num_components=8
    )
    rollout = sample(
        model,
        filter,
        client,
        num_steps=100
    )
    assert len(rollout.states) == 100
    assert len(rollout.actions) == 100

    gcs.rollout.upload_rollout(
        rollout.to_dict(),
        gcs.model.version
    )

test_sample()
