import threading
import time
import sys

class LoadingIndicator:
    def __init__(self, message="Processing"):
        self.message = message
        self.is_running = False
        self.thread = None

    def _animate(self):
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self.is_running:
            sys.stdout.write(f'\r{self.message} {chars[i % len(chars)]}')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write('\r\033[K')  # Limpiar la línea
        sys.stdout.flush()
