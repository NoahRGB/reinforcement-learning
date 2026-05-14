
class LinearScheduler:
    def __init__(self, start_value, end_value, decay_steps):
        self.start_value = start_value
        self.current_value = start_value
        self.current_step = 0
        self.end_value = end_value
        self.decay_steps = decay_steps

    def step(self):
        self.current_step += 1
        progress = min(self.current_step / self.decay_steps, 1.0)
        self.current_value = self.start_value + (self.end_value - self.start_value) * progress

        return self.current_value

    def get_value(self):
        return self.current_value
    

class StepScheduler:
    def __init__(self, values: list, step_thresholds: list):
        self.values = values
        self.step_thresholds = step_thresholds
        self.current_idx = 0
        self.current_value = values[0]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_idx < len(self.step_thresholds) - 1:
            if self.current_step >= self.step_thresholds[self.current_idx+1]:
                self.current_idx += 1
                self.current_value = self.values[self.current_idx]
        return self.current_value

    def get_value(self):
        return self.current_value