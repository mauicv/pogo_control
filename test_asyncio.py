import time
from server.piggpio_async_servo_interface import PIGPIO_AsyncServoInterface, Loop
import matplotlib.pyplot as plt

class MockPIGPIO:
    def __init__(self):
        self.connected = True
        self.values = {18: [], 19: []}

    def get_servo_pulsewidth(self, pin):
        return 0
    
    def set_servo_pulsewidth(self, pin, value):
        self.values[pin].append(value)
        print(f"Setting servo pulsewidth for pin {pin} to {value}")

    def stop(self):
        pass


if __name__ == "__main__":
    pin_map = {
        0: 18,
        1: 19,
    }
    mock_pigpio = MockPIGPIO()
    async_servo_interface = PIGPIO_AsyncServoInterface(
        update_interval=0.01,
        pid_kd=0.01,
        pid_ki=0.001,
        pid_kp=0.001,
        pin_map=pin_map,
        pigpio=mock_pigpio
    )

    async_servo_interface.update_angle_setpoints([0.5, 0.])
    time.sleep(2)
    async_servo_interface.update_angle_setpoints([-0.5, 0.])
    time.sleep(2)
    async_servo_interface.update_angle_setpoints([0.5, 0.])
    time.sleep(2)
    async_servo_interface.update_angle_setpoints([0.6, 0.])
    time.sleep(2)
    async_servo_interface.deinit()

    plt.plot(mock_pigpio.values[18], label="Servo 0")
    plt.plot(mock_pigpio.values[19], label="Servo 1")
    plt.legend()
    plt.show()
