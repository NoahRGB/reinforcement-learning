
class LinearScheduler:
    def __init__(self, start_value, end_value, decay_steps):
        self.start_value = start_value
        self.current_value = start_value
        self.current_step = 0
        self.end_value = end_value
        self.decay_steps = decay_steps

    def step(self, n=1):
        self.current_step = min(self.current_step + n, self.decay_steps)
        progress = self.current_step / self.decay_steps
        self.current_value = self.start_value + (self.end_value - self.start_value) * progress

        return self.current_value

    def get_value(self):
        return self.current_value
    