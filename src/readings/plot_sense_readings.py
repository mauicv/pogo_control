import dotenv
dotenv.load_dotenv()
import numpy as np
from client.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from filters.kalman import KalmanMPU6050Filter
from readings.data import SensorDataArray, StateDataArray


def plot_base_sense_readings(client: Client):
    fig, axs = plt.subplots(ncols=3, nrows=2)
    xs = np.arange(100)
    init_ys = np.zeros(100)
    sensor_data = SensorDataArray(
        acc_xs=init_ys.tolist(),
        acc_ys=init_ys.tolist(),
        acc_zs=init_ys.tolist(),
        gyro_xs=init_ys.tolist(),
        gyro_ys=init_ys.tolist(),
        gyro_zs=init_ys.tolist(),
    )

    acc_xs_plot, = axs[0, 0].plot(xs, init_ys)
    acc_ys_plot, = axs[0, 1].plot(xs, init_ys)
    acc_zs_plot, = axs[0, 2].plot(xs, init_ys)
    gyro_xs_plot, = axs[1, 0].plot(xs, init_ys)
    gyro_ys_plot, = axs[1, 1].plot(xs, init_ys)
    gyro_zs_plot, = axs[1, 2].plot(xs, init_ys)
    axs[0, 0].set_title("Side Accelerometer")
    axs[0, 1].set_title("Forward Accelerometer")
    axs[0, 2].set_title("Up Accelerometer")
    axs[1, 0].set_title("Gyroscope X")
    axs[1, 1].set_title("Gyroscope Y")
    axs[1, 2].set_title("Gyroscope Z")

    axs[0, 0].set_ylim(-10, 10)
    axs[0, 1].set_ylim(-10, 10)
    axs[0, 2].set_ylim(-10, 10)
    axs[1, 0].set_ylim(-10, 10)
    axs[1, 1].set_ylim(-10, 10)
    axs[1, 2].set_ylim(-10, 10)

    kalman_filter = KalmanMPU6050Filter(
        init_ax=0,
        init_ay=0,
        init_az=0,
        init_gx=0,
        init_gy=0,
        init_gz=0
    )

    def animate(i, client, sensor_data: SensorDataArray):
        data, _ = client.send_data({})
        ax, ay, az, gx, gy, gz, *_ = data
        data = kalman_filter(ax, ay, az, gx, gy, gz)
        sensor_data.update(data)
        acc_xs, acc_ys, acc_zs, gyro_xs, gyro_ys, gyro_zs = sensor_data.get_data()
        acc_xs_plot.set_ydata(acc_xs)
        acc_ys_plot.set_ydata(acc_ys)
        acc_zs_plot.set_ydata(acc_zs)
        gyro_xs_plot.set_ydata(gyro_xs)
        gyro_ys_plot.set_ydata(gyro_ys)
        gyro_zs_plot.set_ydata(gyro_zs)

    ani = animation.FuncAnimation(fig, animate, fargs=(client, sensor_data, ), interval=25)
    plt.show()


def plot_readings(client: Client):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    xs = np.arange(100)
    init_ys = np.zeros(100)
    state_data_array = StateDataArray(
        pitch=init_ys.tolist(),
        roll=init_ys.tolist(),
        overturned=init_ys.tolist(),
    )  

    pitch_plot, = axs[0].plot(xs, init_ys)
    axs[0].set_title("pitch")
    axs[0].set_ylim(-1, 1)

    roll_plot, = axs[0].plot(xs, init_ys)
    axs[0].set_title("roll")
    axs[0].set_ylim(-1, 1)

    axs[1].set_title("Conditions")
    overturned_plot, = axs[1].plot(xs, init_ys)
    axs[1].set_ylim(-0.1, 1.1)

    def animate(i, client, state_data_array: StateDataArray):
        data = client.send_data({})
        data, [overturned, ts] = data
        roll, pitch = data[-2:]
        state_data_array.update(
            overturned,
            pitch,
            roll,
        )
        overturned, pitch, roll = state_data_array.get_data()
        overturned_plot.set_ydata(overturned)
        pitch_plot.set_ydata(pitch)
        roll_plot.set_ydata(roll)

    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(client, state_data_array),
        interval=25
    )
    plt.show()
