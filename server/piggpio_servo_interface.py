import pigpio


class PIGPIO_ServoInterface:
    SERVO_PWM_THRESHOLD_MIN = 500
    SERVO_PWM_THRESHOLD_MAX = 2500
    HALF_RANGE = (SERVO_PWM_THRESHOLD_MAX - SERVO_PWM_THRESHOLD_MIN) / 2

    def normalized_action_to_servo_pwm(self, action: float) -> int:
        # Action is in the range [-1, 1]
        return int(self.SERVO_PWM_THRESHOLD_MIN + (1 + action) * self.HALF_RANGE)

    def __init__(self, pin_map: dict):
        self.pin_map = pin_map
        self.pigpio = pigpio.pi()
        if not self.pigpio.connected:
            raise Exception("Failed to connect to pigpio")
        self.servo_pw = {}
        for pin_id, pin in self.pin_map.items():
            try:
                self.servo_pw[pin_id] = self.pigpio.get_servo_pulsewidth(pin)
            except Exception as err:
                print(err)
                self.servo_pw[pin_id] = 0

    def update_angle(self, values: list[float]):
        for pin, value in zip(self.pin_map.values(), values):
            servo_pwm = self.normalized_action_to_servo_pwm(value)
            self.pigpio.set_servo_pulsewidth(pin, servo_pwm)

    def deinit(self):
        self.pigpio.stop()
