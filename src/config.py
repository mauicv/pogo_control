import torch


PRECOMPUTED_MEANS = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

PRECOMPUTED_STDS = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

INITIAL_ACTION = (0, 0, 0, 0, 0, 0, 0, 0)

INITIAL_POSITION= (-0.4, -0.4, 0.4, 0.4, -0.4, -0.4, 0.4, 0.4)