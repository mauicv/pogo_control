"""
In order to use multiple sensors we use the the I2C bus plus n pins on the microcontroller. Each mpu6050 
sensor has the same two addresss 0x68 and 0x69. However, they also have an AD0 pin that toggles between these 
addresses. By using the AD0 pin we can connect multiple sensors to the same I2C bus.
"""
from src.mpu6050 import mpu6050

class MPU6050Interface:
    def __init__(self):
        self.mpu = mpu6050(0x68)

    def get_data(self):
        """Returns a list of the acceleration and gyroscope data.
        
        Order:
        [ax, ay, az, gx, gy, gz]
        """
        return [
            *self.mpu.get_accel_data().values(),
            *self.mpu.get_gyro_data().values()
        ]

    def get_data_sample(self, n_samples: int = 4):
        sum_data = [0] * 6
        for _ in range(n_samples):
            data = self.get_data()
            sum_data = [sum_data[i] + data[i] for i in range(6)]
        return [sum_data[i] / n_samples for i in range(6)]

