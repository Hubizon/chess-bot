from flax import linen as nn
from jax import numpy as jnp


class DenseNet(nn.Module):
    """A simple neural network outputting policy and value based on the current state and legal actions"""
    
    @nn.compact
    def __call__(self, x, legal_actions, train: bool = True):
        batch_dims = x.shape[:-3] # everything except the last 3 dims (3, 3, 2)
        x = x.reshape((*batch_dims, -1))
        x = nn.Dense(128)(x) # FC 18 -> 128
        x = nn.relu(x) # ReLU
        x = nn.Dense(64)(x) # FC 128 -> 64
        x = nn.relu(x) # ReLU
        
        value = nn.Dense(1)(x) # FC 64 -> 1 (value) 
        value = jnp.tanh(value) # tanh(value) for output in [-1, 1]
        value = jnp.squeeze(value, axis=-1)
        
        logits = nn.Dense(9)(x) # FC 64 -> 9 (policy)
        
        # Mask out logits of illegal moves so the agent isn't allowed to pick them
        masked_logits = jnp.where(legal_actions, logits, -1e9)
        
        return masked_logits, value
