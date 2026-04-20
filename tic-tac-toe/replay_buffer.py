import jax
from jax import numpy as jnp

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = None
        self.position = 0
        self.size = 0
        
    def add(self, data):
        # We can dynamically get the number of incoming samples from any key
        num_samples = data['actual_rewards'].shape[0]
        
        # Lazy initialization based on the actual shape of the first incoming data
        if self.buffer is None:
            def init_array(x):
                shape = (self.capacity,) + x.shape[1:]
                return jnp.zeros(shape, dtype=x.dtype)
            self.buffer = jax.tree_util.tree_map(init_array, data)
            
        indices = (jnp.arange(num_samples) + self.position) % self.capacity

        def update_array(b, d):
            return b.at[indices].set(d)
            
        self.buffer = jax.tree_util.tree_map(update_array, self.buffer, data)
        self.position = (self.position + num_samples) % self.capacity
        self.size = min(self.size + num_samples, self.capacity) 