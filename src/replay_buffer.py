import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = None
        self.position = 0
        self.size = 0
        
    def add(self, data):
        num_samples = data['actual_rewards'].shape[0]
        
        if self.buffer is None:
            def init_array(x):
                shape = (self.capacity,) + x.shape[1:]
                return np.zeros(shape, dtype=x.dtype)
            self.buffer = {k: init_array(v) for k, v in data.items()}
            
        indices = (np.arange(num_samples) + self.position) % self.capacity
        
        for key in self.buffer:
            self.buffer[key][indices] = data[key]

        self.position = (self.position + num_samples) % self.capacity
        self.size = min(self.size + num_samples, self.capacity) 