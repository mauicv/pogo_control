import dotenv
dotenv.load_dotenv()

from client.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from calibration.data import SensorDataArray


def plot_sense_readings(client: Client):
    fig, axs = plt.subplots(ncols=3, nrows=3)
    sensor_data = SensorDataArray()

    def animate(i, client, sensor_data: SensorDataArray):
        data = client.send_data({})
        sensor_data.update(data)

        xs, acc_xs, acc_ys, acc_zs, gyro_xs, gyro_ys, gyro_zs, x_ints, y_ints, z_ints = sensor_data.get_data()

        for ax in axs.flat:
            ax.clear()
        axs[0, 0].plot(xs, acc_xs)
        axs[0, 1].plot(xs, acc_ys)
        axs[0, 2].plot(xs, acc_zs)
        axs[1, 0].plot(xs, gyro_xs)
        axs[1, 1].plot(xs, gyro_ys)
        axs[1, 2].plot(xs, gyro_zs)
        axs[2, 0].plot(xs, x_ints)
        axs[2, 1].plot(xs, y_ints)
        axs[2, 2].plot(xs, z_ints)

        axs[0, 0].set_title("Side Accelerometer")
        axs[0, 1].set_title("Forward Accelerometer")
        axs[0, 2].set_title("Up Accelerometer")
        axs[1, 0].set_title("Gyroscope X")
        axs[1, 1].set_title("Gyroscope Y")
        axs[1, 2].set_title("Gyroscope Z")
        axs[2, 0].set_title("Side Integral")
        axs[2, 1].set_title("Forward Integral")
        axs[2, 2].set_title("Up Integral")

    ani = animation.FuncAnimation(fig, animate, fargs=(client, sensor_data, ), interval=1000)
    plt.show()
