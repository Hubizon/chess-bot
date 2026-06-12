from flax import linen as nn
from jax import numpy as jnp

class ResidualBlock(nn.Module):
    filters: int
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        
        x = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            padding='SAME',
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(x)
        x = nn.relu(x)
        x = x.astype(self.compute_dtype)
        
        x = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            padding='SAME',
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(x)
        x = x.astype(self.compute_dtype)
        
        x = x + residual
        x = nn.relu(x)
        
        return x

class ResNet(nn.Module):
    num_blocks: int = 10
    filters: int = 256
    num_actions: int = 4672
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x, legal_actions, train: bool = True):
        x = x.astype(self.compute_dtype)
        x = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            padding='SAME',
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(x)
        x = nn.relu(x)
        x = x.astype(self.compute_dtype)
        
        for _ in range(self.num_blocks):
            x = ResidualBlock(
                self.filters,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )(x, train)
            
        # Policy head
        p = nn.Conv(
            features=73,
            kernel_size=(1, 1),
            padding='SAME',
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)
        p = nn.BatchNorm(
            use_running_average=not train,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(p)
        policy_logits = p.reshape((p.shape[0], -1))
        policy_logits = policy_logits.astype(jnp.float32)
        legal_actions = legal_actions.astype(jnp.bool_)
        masked_logits = jnp.where(legal_actions, policy_logits, jnp.float32(-1e9))
        
        # Value head
        v = nn.Conv(
            features=32,
            kernel_size=(1, 1),
            padding='SAME',
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)
        v = nn.BatchNorm(
            use_running_average=not train,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(v)
        v = nn.relu(v)
        v = v.astype(self.compute_dtype)
        v = v.reshape((v.shape[0], -1))
        v = nn.Dense(256, dtype=self.compute_dtype, param_dtype=self.param_dtype)(v)
        v = nn.relu(v)
        v = nn.Dense(1, dtype=self.compute_dtype, param_dtype=self.param_dtype)(v)
        value = jnp.tanh(v.astype(jnp.float32))
        
        return masked_logits, jnp.squeeze(value, axis=-1)
