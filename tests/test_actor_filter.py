import os
import sys
import logging
import matplotlib.pyplot as plt
import torch
import time


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
        experiment_name='test'
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
    torch.set_grad_enabled(False)
    action = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    action = filter(action)
    states = []
    actions = []
    unfiltered_actions = [] 
    for _ in range(100):
        state = client.send_data(action)
        state = torch.tensor(state)
        
        action = model(state, deterministic=True).numpy()
        unfiltered_actions.append(action)
        action = filter(action)
        actions.append(action)
        
        state = client.send_data(action)
        states.append(state)
    plt.plot(unfiltered_actions, label="Unfiltered")
    plt.plot(actions, label="Filtered")
    plt.legend()
    plt.show()


test_sample()