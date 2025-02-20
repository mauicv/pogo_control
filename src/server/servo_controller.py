from server.servo import Servo
from server.loop import Loop
import time


class ServoController:
    servos: list[Servo]

    def __init__(
            self,
            servo_update_interval: float = 0.01,
            gpio = None,
            **kwargs):
        super().__init__(**kwargs)
        self.servo_update_interval = servo_update_interval
        self.gpio = gpio
        if not self.gpio.connected:
            raise Exception("Failed to connect to gpio")

        time.sleep(2)
        self.servo_update_loop = Loop(
            interval=self.servo_update_interval,
            func=self._update_angle
        )
        self.servo_update_loop.start()
        self.last_servo_set_ts = time.time()

    def get_servo_data(self):
        return [servo._value for servo in self.servos]

    def update_angle(self, values: list[float]):
        for servo, value in zip(self.servos, values):
            servo.update_setpoint(value)
        self.last_servo_set_ts = time.time()

    def _update_angle(self):
        for servo in self.servos:
            self.gpio.set_servo_pulsewidth(
                servo.pin,
                servo.get_pwm()
            )

    def deinit_servo_controller(self):
        self.servo_update_loop.stop()
        self.gpio.stop()