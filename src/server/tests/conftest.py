# conftest.py
import pytest
import random
import time
from server.servo_controller import ServoController
from server.mpu6050Mixin import MPU6050Mixin
from server.servo import Servo


class MockValues:
    def __init__(self, values):
        self._values = values

    def values(self):
        return self._values


class Mock_mpu6050:
    def __init__(self):
        self.data = [0] * 6
        self.last_update = time.time()
        self.intervals = []

    def get_accel_data(self):
        time.sleep(0.001)
        self.intervals.append(time.time() - self.last_update)
        self.last_update = time.time()
        self.data = MockValues([(random.uniform(-10, 10)) for _ in range(3)])
        return self.data

    def get_gyro_data(self):
        time.sleep(0.001)
        self.data = MockValues([(random.uniform(-10, 10)) for _ in range(3)])
        return self.data


class Mock_PIGPIO:
    def __init__(self):
        self.results = []
        self.last_update = time.time()
        self.intervals = []

    def set_servo_pulsewidth(self, pin, pulsewidth):
        time.sleep(0.001)
        self.results.append((pin, pulsewidth))
        self.intervals.append(time.time() - self.last_update)
        self.last_update = time.time()

    def stop(self):
        pass

    def connected(self):
        return True

pid_defaults = {
    "kp": 0.1,
    "ki": 0.01,
    "kd": 0.001
}

class Robot(ServoController, MPU6050Mixin):
    servos: list[Servo] = [
        Servo(
            name="A",
            pin_id=0,
            pin=4,
            pin_limits=(-0.2, 0.2),
            init_value=-0.4,
            **pid_defaults
        ),
        Servo(
            name="B",
            pin_id=2,
            pin=27,
            pin_limits=(-0.5, 0.5),
            init_value=-0.4,
            reverse=True,
            **pid_defaults
        ),
    ]

    def __init__(
            self,
            update_interval: float = 0.1,
            gpio=None,
            mpu=None
        ):
        super().__init__(
            servo_update_interval=update_interval,
            mpu_update_interval=update_interval,
            gpio=gpio,
            mpu=mpu
        )

    def get_data(self):
        return self.get_servo_data() + self.get_mpu_data()
    
    def deinit(self):
        self.deinit_servo_controller()
        self.deinit_mpu()


@pytest.fixture
def mock_mpu():
    return Mock_mpu6050()


@pytest.fixture
def mock_gpio():
    return Mock_PIGPIO()


@pytest.fixture()
def robot(mock_mpu, mock_gpio):
    robot = Robot(
        update_interval=0.001,
        gpio=mock_gpio,
        mpu=mock_mpu
    )
    yield robot
    robot.deinit()