from dataclasses import dataclass
from simple_pid import PID

SERVO_PWM_THRESHOLD_MIN: int = 500
SERVO_PWM_THRESHOLD_MAX: int = 2500
HALF_RANGE = (SERVO_PWM_THRESHOLD_MAX - SERVO_PWM_THRESHOLD_MIN) / 2 # 1000

@dataclass
class Servo:
    name: str
    pin_id: int
    pin: int
    pin_limits: tuple[float, float]
    init_value: float
    reverse: bool = False
    kp: float = 0.1
    ki: float = 0.01
    kd: float = 0.001
    _value: float = 0.0
    pid_controller: PID = None
    offset: float = 0.0

    def __post_init__(self):
        self.pid_controller = PID(
            self.kp, self.ki, self.kd,
            starting_output=0,
            setpoint=self.init_value,
        )

    def update_setpoint(self, setpoint: float):
        self.pid_controller.setpoint = setpoint

    @property
    def value(self):
        value = self._value
        if value > self.pin_limits[1]: value = self.pin_limits[1]
        elif value < self.pin_limits[0]: value = self.pin_limits[0]
        value = -value if self.reverse else value
        value += self.offset
        return value

    def _value_to_pwm(self) -> int:
        pwm_val = int(SERVO_PWM_THRESHOLD_MIN + (1 + self.value) * HALF_RANGE)
        if pwm_val > SERVO_PWM_THRESHOLD_MAX: pwm_val = SERVO_PWM_THRESHOLD_MAX
        elif pwm_val < SERVO_PWM_THRESHOLD_MIN: pwm_val = SERVO_PWM_THRESHOLD_MIN
        return pwm_val

    def get_pwm(self):
        self._value += self.pid_controller(self._value)
        return self._value_to_pwm()
