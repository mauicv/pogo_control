"""
In order to use multiple sensors we use the the I2C bus plus n pins on the microcontroller. Each mpu6050 
sensor has the same two addresss 0x68 and 0x69. However, they also have an AD0 pin that toggles between these 
addresses. By using the AD0 pin we can connect multiple sensors to the same I2C bus.
"""
from src.mpu6050 import mpu6050
from filters.low_pass import LowPassFilter

class MPU6050Interface:
    def __init__(self, filter=None):
        self.mpu = mpu6050(0x68)
        if filter is None:
            self.filter = LowPassFilter(num_components=6)
        else:
            self.filter = filter

    def get_data(self):
        """Returns a list of the acceleration and gyroscope data.
        
        Order:
        [ax, ay, az, gx, gy, gz]
        """
        return self.filter.filter([
            *self.mpu.get_accel_data().values(),
            *self.mpu.get_gyro_data().values()
        ])