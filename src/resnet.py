from flax import linen as nn
from jax import numpy as jnp


class ResidualBlock(nn.Module):
    filters: int
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        
        x = x + residual
        x = nn.relu(x)
        
        return x

class ResNet(nn.Module):
    num_blocks: int = 10
    filters: int = 256
    num_actions: int = 4672
    
    @nn.compact
    def __call__(self, x, legal_actions, train: bool = True):
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.filters)(x, train)
            
        # Policy head: outputs logits masking illegal actions out
        p = nn.Conv(features=73, kernel_size=(1, 1), padding='SAME', use_bias=False)(x)
        p = nn.BatchNorm(use_running_average=not train)(p)
        p = nn.relu(p)
        p = p.reshape((p.shape[0], -1)) 
        policy_logits = nn.Dense(self.num_actions)(p)
        masked_logits = jnp.where(legal_actions, policy_logits, -1e9)
        
        # Value head: computes single scalar score between -1 and 1
        v = nn.Conv(features=1, kernel_size=(1, 1), padding='SAME', use_bias=False)(x)
        v = nn.BatchNorm(use_running_average=not train)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1))
        v = nn.Dense(256)(v)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        value = jnp.tanh(v)
        
        return masked_logits, jnp.squeeze(value, axis=-1)
