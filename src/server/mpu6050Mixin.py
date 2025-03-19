"""
In order to use multiple sensors we use the the I2C bus plus n pins on the microcontroller. Each mpu6050 
sensor has the same two addresss 0x68 and 0x69. However, they also have an AD0 pin that toggles between these 
addresses. By using the AD0 pin we can connect multiple sensors to the same I2C bus.
"""
import time
from filters.complementary import ComplementaryFilter
from filters.kalman import KalmanMPU6050Filter
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
            self.filter = KalmanMPU6050Filter()
        else:
            self.filter = filter

        self.c_filter = ComplementaryFilter(alpha=0.95)

        # 3 for acc, 3 for gyro, 2 for comp
        self.latest_filtered_data = [0] * (6 + 2)
        self.last_mpus6050_sample_ts = time.time()

        self.mpu_update_loop = Loop(
            interval=mpu_update_interval,
            func=self._update_mpu_data
        )
        self.mpu_update_loop.start()

    def _update_mpu_data(self):
        """Background task to continuously update filtered MPU readings"""
        ATTEMPTS = 5
        for attempt in range(ATTEMPTS):
            try:
                raw_data = [
                    *self.mpu.get_accel_data().values(),
                    *self.mpu.get_gyro_data().values()
                ]
                break
            except OSError as e:
                if e.errno == 121:
                    if attempt == ATTEMPTS - 2:
                        from server.mpu6050 import mpu6050
                        self.mpu = mpu6050(0x68)
                    if attempt == ATTEMPTS - 1:
                        raise e
                    print(f"Error updating MPU data: {e}")
                    time.sleep(0.05)
                continue

        self.c_filter.update(raw_data[:3], raw_data[3:])
        self.latest_filtered_data = self.filter(raw_data)
        gyro_data = [
            g_data/1000 for g_data in self.latest_filtered_data[3:]
        ]
        acc_data = [
            a_data/10 for a_data in self.latest_filtered_data[:3]
        ]
        self.latest_filtered_data = [
            *acc_data,
            *gyro_data,
        ]
        self.last_mpus6050_sample_ts = time.time()

    def get_mpu_data(self):
        """Returns the most recent filtered MPU data"""
        return [
            *self.latest_filtered_data,
            self.c_filter.roll,
            self.c_filter.pitch,
            self.overturned,
            self.last_mpus6050_sample_ts
        ]

    @property
    def overturned(self):
        _, _, az = self.latest_filtered_data[:3]
        overturned = az * 10 < 1
        return int(overturned)

    def deinit_mpu(self):
        """Clean up resources"""
        self.mpu_update_loop.stop()