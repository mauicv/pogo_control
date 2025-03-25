import threading


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
