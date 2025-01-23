import dotenv
dotenv.load_dotenv()
import numpy as np
from client.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from calibration.data import SensorDataArray, PitchRollDataArray, VelocityDataArray
from calibration.filter import ComplementaryFilter, VelocityFilter
import random


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

    def animate(i, client, sensor_data: SensorDataArray):
        data = client.send_data({})
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


def plot_pitch_roll_readings(client: Client):
    fig, ax = plt.subplots()
    xs = np.arange(100)
    init_ys = np.zeros(100)
    sensor_data = PitchRollDataArray(
        roll=init_ys.tolist(),
        pitch=init_ys.tolist()
    )

    acc_sensor_data = PitchRollDataArray(
        roll=init_ys.tolist(),
        pitch=init_ys.tolist()
    )
    filter = ComplementaryFilter()

    pitch_plot, = ax.plot(xs, init_ys)
    roll_plot, = ax.plot(xs, init_ys)
    acc_pitch_plot, = ax.plot(xs, init_ys)
    acc_roll_plot, = ax.plot(xs, init_ys)
    ax.set_title("Pitch/Roll")
    ax.set_ylim(-100, 100)

    def animate(i, client, pitch_roll_data: PitchRollDataArray):
        data = client.send_data({})
        acc_data = data[0:3]
        gyro_data = data[3:6]
        filter.update(acc_data, gyro_data)
        
        pitch_roll_data.update(filter.pitch, filter.roll)
        pitch, roll = pitch_roll_data.get_data()

        acc_sensor_data.update(filter.a_pitch, filter.a_roll)
        acc_pitch, acc_roll = acc_sensor_data.get_data()

        pitch_plot.set_ydata(pitch)
        roll_plot.set_ydata(roll)

        acc_pitch_plot.set_ydata(acc_pitch)
        acc_roll_plot.set_ydata(acc_roll)

    ani = animation.FuncAnimation(fig, animate, fargs=(client, sensor_data, ), interval=25)
    plt.show()


def plot_v_readings(client: Client):
    fig, ax = plt.subplots()
    xs = np.arange(100)
    init_ys = np.zeros(100)
    # sensor_data = PitchRollDataArray(
    #     roll=init_ys.tolist(),
    #     pitch=init_ys.tolist()
    # )

    v_data = VelocityDataArray(
        vx=init_ys.tolist(),
        vy=init_ys.tolist()
    )

    a_data = VelocityDataArray(
        vx=init_ys.tolist(),
        vy=init_ys.tolist()
    )

    c_filter = ComplementaryFilter()
    v_filter = VelocityFilter()

    # ax_plot, = ax.plot(xs, init_ys)
    ay_plot, = ax.plot(xs, init_ys)
    # vx_plot, = ax.plot(xs, init_ys)
    vy_plot, = ax.plot(xs, init_ys)
    ax.set_ylim(-10, 10)

    def animate(
            i,
            client,
            v_data: VelocityDataArray,
            a_data: VelocityDataArray
        ):
        data = client.send_data({})
        acc_data = data[0:3]
        gyro_data = data[3:6]
        c_filter.update(acc_data, gyro_data)
        v_filter.update(acc_data, c_filter.g_xy)

        # pitch_roll_data.update(c_filter.pitch, c_filter.roll)
        # pitch, roll = pitch_roll_data.get_data()

        vx, vy = v_filter.v_xy
        v_data.update(vx, vy)
        vx, vy = v_data.get_data()
        # vx_plot.set_ydata(vx)
        vy_plot.set_ydata(vy)

        ax, ay = v_filter.a_xy
        a_data.update(ax, ay)
        ax, ay = a_data.get_data()
        # ax_plot.set_ydata(ax)
        ay_plot.set_ydata(ay)

    ani = animation.FuncAnimation(fig, animate, fargs=(client, v_data, a_data, ), interval=25)
    plt.show()
