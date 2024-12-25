import os
import sys
import logging

logger = logging.getLogger(__name__)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.gcs_interface import GCS_Interface


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s'
    )

    gcs = GCS_Interface(
        credentials='world-model-rl-01a513052a8a.json',
        bucket='pogo_wmrl',
        experiment_name='test',
        model_limits=4,
        num_runs=100,
    )

    gcs.loader.init_load()
    print(gcs.loader.indexed_rollouts)
    gcs.loader.fetch_rollouts()

    # action_shape = 8
    # data_loader = DataLoader()
    # for _ in range(4):
    #     data_loader.fetch_rollouts()

    # s, a, r, d = data_loader.sample(batch_size=3, num_time_steps=10)

    # assert s.shape == (3, 10, 3, 64, 64)
    # assert a.shape == (3, 10, action_shape)
    # assert r.shape == (3, 10, 1)
    # assert d.shape == (3, 10, 1)
    # data_loader.close()
