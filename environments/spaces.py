
class DiscreteSpace:
    def __init__(self, dimensions):
        self.dimensions = dimensions

class ContinuousSpace:
    def __init__(self, dimensions, min_bound, max_bound):
        self.dimensions = dimensions
        self.min_bound = min_bound
        self.max_bound = max_bound
