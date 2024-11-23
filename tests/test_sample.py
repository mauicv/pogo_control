import os
import sys
import torch
import logging

logger = logging.getLogger(__name__)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.sample import sample
from tests.mock_client import MockClient
from client.model import Actor
from filters.butterworth import ButterworthFilter


def test_sample():
    client = MockClient(
        host="localhost",
        port=8000
    )
    model = Actor(
        input_dim=6,
        output_dim=8,
        bound=1,
        num_layers=2
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

test_sample()