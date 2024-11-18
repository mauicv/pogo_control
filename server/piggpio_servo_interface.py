import pigpio


class PIGPIO_ServoInterface:
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

    def update_angle(self, angles):
        for pin, angle in zip(self.pin_map.values(), angles):
            self.pigpio.set_servo_pulsewidth(pin, angle)

    def deinit(self):
        self.pigpio.stop()
