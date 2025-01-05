"""
In order to use multiple sensors we use the the I2C bus plus n pins on the microcontroller. Each mpu6050 
sensor has the same two addresss 0x68 and 0x69. However, they also have an AD0 pin that toggles between these 
addresses. By using the AD0 pin we can connect multiple sensors to the same I2C bus.
"""
from filters.butterworth import ButterworthFilter
from server.loop import Loop

class MPU6050Mixin:
    def __init__(
            self,
            filter=None,
            mpu=None,
            mpu_update_interval=0.01,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.mpu_update_interval = mpu_update_interval
        if mpu is None:
            from server.mpu6050 import mpu6050
            self.mpu = mpu6050(0x68)
        else:
            self.mpu = mpu

        if filter is None:
            self.filter = ButterworthFilter(num_components=6)
        else:
            self.filter = filter

        self._latest_filtered_data = [0] * 6

        self.mpu_update_loop = Loop(
            interval=mpu_update_interval,
            func=self._update_mpu_data
        )
        self.mpu_update_loop.start()

    def _update_mpu_data(self):
        """Background task to continuously update filtered MPU readings"""
        raw_data = [
            *self.mpu.get_accel_data().values(),
            *self.mpu.get_gyro_data().values()
        ]
        self._latest_filtered_data = self.filter(raw_data)

    def get_mpu_data(self):
        """Returns the most recent filtered MPU data"""
        return self._latest_filtered_data

    def deinit_mpu(self):
        """Clean up resources"""
        self.mpu_update_loop.stop()