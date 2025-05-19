import torch


PRECOMPUTED_MEANS = torch.tensor([[
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    -2.78550197e-02, 9.98010031e-03, 8.42685099e-01, -2.13750695e-03, -7.70095060e-04, 8.27231644e-04, 9.81528581e-03, 2.56934800e-03
]])

PRECOMPUTED_STDS = torch.tensor([[
    1, 1, 1, 1, 1, 1, 1, 1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.24452243, 0.13892446, 0.12971678, 0.01274018, 0.03735261, 0.01137343, 0.08246328, 0.04498999
]])

INITIAL_ACTION = (0, 0, 0, 0, 0, 0, 0, 0)


INITIAL_POSITION= (-0.3, 0.4, -0.3, 0.4, -0.3, 0.4, -0.3, 0.4)
# INITIAL_POSITION= (-0.4, 0.3, -0.4, 0.3, -0.4, 0.3, -0.4, 0.3)


# pogo pogo move --front-left-bottom=0.0 --front-right-bottom=0.0 --back-right-bottom=0.4 --back-left-bottom=0.4 --front-left-top=-0.3 --front-right-top=-0.3 --back-right-top=-0.5 --back-left-top=-0.5


ALT_INITIAL_POSITION = (-0.3, 0.0, -0.3, 0.0, -0.5, 0.4, -0.5, 0.4)
