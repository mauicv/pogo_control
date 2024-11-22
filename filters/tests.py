import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.butterworth import ButterworthFilter
from filters.low_pass import LowPassFilter


DATA_STREAM_LENGTH = 100

if __name__ == "__main__":

    # Generate a data stream
    data_stream_x = (
        np.sin(np.linspace(0, 10, DATA_STREAM_LENGTH))
        + 0.1 * np.random.randn(DATA_STREAM_LENGTH)
    )
    data_stream_y = (
        np.cos(np.linspace(0, 10, DATA_STREAM_LENGTH)) +
        0.25 * np.sin(np.linspace(1, 5, DATA_STREAM_LENGTH)) +
        0.1 * np.random.randn(DATA_STREAM_LENGTH)
    )
    data_stream = np.stack([data_stream_x, data_stream_y]).T
    data_stream = data_stream.tolist()
    butterworth_filter = ButterworthFilter(
        order=2,
        cutoff=5.0,
        fs=50.0,
        num_components=2
    )
    low_pass_filter = LowPassFilter(
        alpha=0.85,
        num_components=2
    )
    filtered_output_butterworth = []
    filtered_output_low_pass = []
    for x in data_stream:
        filtered_output_butterworth.append(butterworth_filter.filter(x))
        filtered_output_low_pass.append(low_pass_filter.filter(x))

    data_stream = np.array(data_stream)
    filtered_output_butterworth = np.array(filtered_output_butterworth)
    filtered_output_low_pass = np.array(filtered_output_low_pass)

    plt.subplots(nrows=2, ncols=1)
    plt.subplot(2, 1, 1)
    plt.plot(data_stream[:, 0], label='Original')
    plt.plot(filtered_output_butterworth[:, 0], label='Butterworth')
    plt.plot(filtered_output_low_pass[:, 0], label='Low Pass')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(data_stream[:, 1], label='Original')
    plt.plot(filtered_output_butterworth[:, 1], label='Butterworth')
    plt.plot(filtered_output_low_pass[:, 1], label='Low Pass')
    plt.legend()
    plt.show()