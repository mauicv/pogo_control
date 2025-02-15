import dotenv
dotenv.load_dotenv()

import numpy as np
from client.client import Client
from tqdm import tqdm
from readings.data import SensorDataArray


def measure_mpu_offsets(client: Client):

    sensor_data = SensorDataArray()
    for _ in tqdm(range(100)):
        data = client.send_data({})
        sensor_data.update(data)

    xs, acc_xs, acc_ys, acc_zs, gyro_xs, gyro_ys, gyro_zs, x_ints, y_ints, z_ints = sensor_data.get_data()
    print(np.mean(acc_xs), np.mean(acc_ys), np.mean(acc_zs))
    print(np.mean(gyro_xs), np.mean(gyro_ys), np.mean(gyro_zs))
