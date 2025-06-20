from peripherals.pogo.mpu6050Mixin import MPU6050Mixin
from peripherals.pogo.servo_controller import ServoController
from peripherals.pogo.servo import Servo
import time


generic_values = {
    "kp": 0.08,
    "ki": 0.01,
    "kd": 0.005,
}

class Pogo(ServoController, MPU6050Mixin):
    servos: list[Servo] = [
        Servo(name="front_right_top", pin_id=0, pin=4, pin_limits=(-0.3, 0.9), init_value=-0.4, offset=0.0, **generic_values),
        Servo(name="front_right_bottom", pin_id=1, pin=18, pin_limits=(-0.9, 0.9), init_value=-0.4, offset=0.0, **generic_values),
        Servo(name="front_left_top", pin_id=2, pin=27, pin_limits=(-0.3, 0.9), init_value=-0.4, offset=0.1, reverse=True, **generic_values),
        Servo(name="front_left_bottom", pin_id=3, pin=10, pin_limits=(-0.9, 0.9), init_value=-0.4, offset=0.0, reverse=True, **generic_values),
        Servo(name="back_right_top", pin_id=4, pin=20, pin_limits=(-0.9, 0.2), init_value=-0.4, offset=0.0, **generic_values),
        Servo(name="back_right_bottom", pin_id=5, pin=19, pin_limits=(-0.9, 0.9), init_value=-0.4, offset=0.0, **generic_values),
        Servo(name="back_left_top", pin_id=6, pin=13, pin_limits=(-0.9, 0.2), init_value=-0.4, offset=-0.6, reverse=True, **generic_values),
        Servo(name="back_left_bottom", pin_id=7, pin=6, pin_limits=(-0.9, 0.9), init_value=-0.4, offset=0.0, reverse=True, **generic_values),
    ]

    def __init__(
            self,
            update_interval: float = 0.01,
            gpio=None,
            mpu=None,
        ):

        if not gpio:
            import pigpio
            gpio = pigpio.pi()
        if not mpu:
            from peripherals.pogo.mpu6050 import mpu6050
            mpu = mpu6050(0x68)

        super().__init__(
            servo_update_interval=update_interval,
            mpu_update_interval=update_interval,
            gpio=gpio,
            mpu=mpu,
        )

    def _parse_command(self, message):
        command = message['command']
        args = message['args'] if 'args' in message else {}
        return command, args

    def handle_message(self, message):
        command, args = self._parse_command(message)
        return {
            'act': self.act,
            'set_servo_states': self.set_servo_states,
            'read': self.get_data,
        }[command](**args)
    
    def act(self, values: list[float]):
        self.update_setpoint_delta(values)
        return True
    
    def set_servo_states(self, values: list[float]):
        self.update_setpoint(values)
        return True

    def get_data(self):
        state_data = [
            *self.latest_filtered_data,
            self.c_filter.roll,
            self.c_filter.pitch,
        ]
        conditions_data = [
            self.overturned,
            self.last_mpus6050_sample_ts,
            self.last_servo_set_ts
        ]
        return [
            self.get_servo_data(),
            state_data,
            conditions_data
        ]
    
    def deinit(self):
        self.deinit_servo_controller()
        self.deinit_mpu()


# class SensorPogo(MPU6050Mixin):
#     servos: list[Servo] = []

#     def __init__(self,
#             update_interval: float = 0.01,
#             mpu=None,
#         ):
#         if not mpu:
#             from peripherals.pogo.mpu6050 import mpu6050
#             mpu = mpu6050(0x68)

#         super().__init__(
#             mpu_update_interval=update_interval,
#             mpu=mpu,
#         )

#     def get_data(self):
#         state_data = [
#             *self.latest_filtered_data,
#             self.c_filter.roll,
#             self.c_filter.pitch,
#         ]
#         conditions_data = [
#             self.overturned,
#             self.last_mpus6050_sample_ts,
#         ]
#         return [
#             state_data,
#             conditions_data
#         ]
    
#     def deinit(self):
#         self.deinit_mpu()
