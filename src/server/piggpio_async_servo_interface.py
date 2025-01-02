import threading
from server.pid import MultiPIDController

class Loop:
    def __init__(self, interval, func):
        self.interval = interval
        self.func = func
        self._running = True

    def start(self):
        self._running = True
        threading.Timer(
            self.interval,
            self._loop
        ).start()
        
    def _loop(self):
        if self._running:
            self.func()
            threading.Timer(
                self.interval,
                self._loop
            ).start()

    def stop(self):
        self._running = False


class PIGPIO_AsyncServoInterface:
    SERVO_PWM_THRESHOLD_MIN = 500
    SERVO_PWM_THRESHOLD_MAX = 2500
    HALF_RANGE = (SERVO_PWM_THRESHOLD_MAX - SERVO_PWM_THRESHOLD_MIN) / 2

    def normalized_action_to_servo_pwm(self, action: float) -> int:
        # Action is in the range [-1, 1]
        if action > 1: action = 1
        elif action < -1: action = -1
        return int(self.SERVO_PWM_THRESHOLD_MIN + (1 + action) * self.HALF_RANGE)
    
    def servo_pwm_to_normalized_action(self, pwm_action: float) -> int:
        return ((pwm_action - self.SERVO_PWM_THRESHOLD_MIN)/self.HALF_RANGE) - 1

    def __init__(
            self,
            pin_map: dict,
            init_pos: list[float],
            update_interval: float = 0.01,
            pid_kp: float = 0.1,
            pid_ki: float = 0.01,
            pid_kd: float = 0.001,
            pigpio = None,
        ):
        self.init_pos = init_pos
        self.pin_map = pin_map
        self.update_interval = update_interval
        self.pigpio = pigpio
        if not self.pigpio.connected:
            raise Exception("Failed to connect to pigpio")

        self.servo_pw = [0] * len(self.pin_map)
        for pin_id, _ in self.pin_map.items():
            self.servo_pw[pin_id] = self.init_pos[pin_id]

        self.pid_controller = MultiPIDController(
            num_components=len(pin_map),
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            init_setpoints=self.servo_pw,
        )

        self.servo_update_loop = Loop(
            interval=self.update_interval,
            func=self._update_angle
        )
        self.servo_update_loop.start()

    def get_data(self):
        return [self.servo_pw[pin_id] for pin_id in range(len(self.servo_pw))]

    def update_angle(self, values: list[float]):
        self.pid_controller.set_setpoint(values)

    def _update_angle(self):
        updates = [a+b for a, b in zip(self.servo_pw, self.pid_controller(self.servo_pw))]
        for pin_id, value in enumerate(updates):
            servo_pwm = self.normalized_action_to_servo_pwm(value)
            pin = self.pin_map[pin_id]
            self.pigpio.set_servo_pulsewidth(pin, servo_pwm)
        self.servo_pw = updates

    def deinit(self):
        self.servo_update_loop.stop()
        self.pigpio.stop()