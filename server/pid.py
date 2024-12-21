import numpy as np
from simple_pid import PID


class MultiPIDController:
    def __init__(
            self,
            num_components: int = 8,
            init_setpoint: float = 0.0,
            kp: float = 1.0,
            ki: float = 0.1,
            kd: float = 0.01,
        ):
        self.controllers = [PID(kp, ki, kd, setpoint=init_setpoint) for _ in range(num_components)]

    def set_setpoint(self, new_setpoints: list[float]):
        assert len(new_setpoints) == len(self.controllers), \
            "Number of new setpoints must match number of controllers"
        for value, controller in zip(new_setpoints, self.controllers):
            controller.setpoint = value

    def __call__(self, new_values: list[float]):
        assert len(new_values) == len(self.controllers), \
            "Number of new values must match number of controllers"
        return [controller(value) for controller, value in zip(self.controllers, new_values)]
