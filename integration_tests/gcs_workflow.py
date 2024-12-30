import os
import sys
import logging

logger = logging.getLogger(__name__)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.sample import sample
from tests.mock_client import MockClient
from filters.butterworth import ButterworthFilter
from client.gcs_interface import GCS_Interface
from client.model import Actor

if __name__ == '__main__':
    gcs = GCS_Interface(
        experiment_name='test',
        credentials='world-model-rl-01a513052a8a.json',
        bucket='pogo_wmrl',
        model_limits=4,
        num_runs=3,
        rollout_length=100,
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

    print('Sampling first set of rollouts')
    for i in range(2):
        rollout = sample(
            model,
            filter,
            client,
            num_steps=100,
            interval=0.001
        )
        assert len(rollout.states) == 100
        assert len(rollout.actions) == 100
        gcs.rollout.upload_rollout(
            rollout.to_dict(),
            gcs.model.version
        )

    print('Indexing rollouts')
    gcs.loader.index_rollouts()
    print('Fetching rollouts')
    gcs.loader.fetch_rollouts()

    assert gcs.loader.fetched_rollouts.issubset(gcs.loader.indexed_rollouts)

    print('Sampling second set of rollouts')
    for i in range(2):
        rollout = sample(
            model,
            filter,
            client,
            num_steps=100,
            interval=0.001
        )
        gcs.rollout.upload_rollout(
            rollout.to_dict(),
            gcs.model.version
        )


    print('Indexing rollouts again')
    gcs.loader.index_rollouts()

    assert gcs.loader.indexed_rollouts != gcs.loader.fetched_rollouts
    gcs.loader.fetch_rollouts()
    assert gcs.loader.fetched_rollouts.issubset(gcs.loader.indexed_rollouts)

    s, a, r = gcs.loader.sample(
        batch_size=2,
        num_time_steps=16,
        from_start=True
    )

    assert s.shape == (2, 16, 6)
    assert a.shape == (2, 16, 8)
    assert r.shape == (2, 16, 1)

    model = Actor(
        input_dim=6,
        output_dim=8,
        bound=1,
        num_layers=2
    )
    gcs.model.upload_model(model)
    gcs.model.remove_old_models()

    for i in range(2):
        rollout = sample(
            model,
            filter,
            client,
            num_steps=100,
            interval=0.001
        )
        gcs.rollout.upload_rollout(
            rollout.to_dict(),
            gcs.model.version
        )

    gcs.loader.index_rollouts()
    print(gcs.loader.fetched_rollouts)
