from dataclasses import dataclass
from simple_pid import PID

SERVO_PWM_THRESHOLD_MIN: int = 500
SERVO_PWM_THRESHOLD_MAX: int = 2500
HALF_RANGE = (SERVO_PWM_THRESHOLD_MAX - SERVO_PWM_THRESHOLD_MIN) / 2

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
        return value

    def _value_to_pwm(self) -> int:
        return int(SERVO_PWM_THRESHOLD_MIN + (1 + self.value) * HALF_RANGE)

    def get_pwm(self):
        self._value += self.pid_controller(self._value)
        return self._value_to_pwm()
