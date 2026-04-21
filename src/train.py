import functools
import os
import time

import jax
import numpy as np
import optax
import pgx
from jax import numpy as jnp
from pgx.experimental import auto_reset

from .densenet import DenseNet
from .mcts import run_mcts
from .replay_buffer import ReplayBuffer
from .resnet import ResNet

CPU_DEVICE = jax.devices("cpu")[0]

@functools.partial(jax.jit, static_argnames=("env", "network", "batch_size", "max_steps", "num_simulations"))
def generate_self_play_data(env, network, params, batch_stats, rng, batch_size, max_steps, zero_temp_move, num_simulations):
    """Generates self-play data by running games with the current policy."""
    # Generate initial states for a batch of games
    rng, env_rng = jax.random.split(rng, 2)
    env_rngs = jax.random.split(env_rng, batch_size)
    initial_states = jax.vmap(env.init)(env_rngs)
    
    def make_move(states, step_rng):
        """Makes a move for each game in the batch using MCTS and returns the next states and the game history.
         - states: the current states of the batch of games; shape: [B, ...]
         - step_rng: a random key for this step; shape: [2]"""
        step_rng, init_rng = jax.random.split(step_rng)
        init_rngs = jax.random.split(init_rng, batch_size)
        temperatures = jnp.where(states._step_count < zero_temp_move, 1.0, 0.0)[:, None]
        policy_outputs = run_mcts(step_rng, params, batch_stats, states, network, env.step, num_simulations, temperatures)
        next_states = jax.vmap(auto_reset(env.step, env.init))(states, policy_outputs.action, init_rngs)
        
        # Keep only fields needed for training targets to reduce temporary memory.
        current_games = {
            'obs': states.observation.astype(jnp.bool_),
            'masks': states.legal_action_mask.astype(jnp.bool_),
            'current_player': states.current_player.astype(jnp.int8),
            'mcts_probs': policy_outputs.action_weights.astype(jnp.float16),
            'rewards': next_states.rewards[:, 0].astype(jnp.int8), # reward for player 0
            'is_terminated': next_states.terminated.astype(jnp.bool_)
        }
        
        return next_states, current_games
    
    step_rngs = jax.random.split(rng, max_steps)
    final_states, game_history = jax.lax.scan(make_move, initial_states, step_rngs)
    # game_history is a dictionary, where each element contains an array of shape [max_steps, B, ...] with data for each step of the games
    return game_history


@jax.jit
def generate_training_data(game_histories):
    """
    Generates training data from the game history. The training data is a dictionary of flattened arrays, each containing:
    - states: the states of the environment for each training example; shape: [B * max_steps, 3, 3, 2]
    - masks: the legal actions for each state; shape: [B * max_steps, 9]
    - mcts_probs: the target policy probabilities for each state (the output of MCTS); shape: [B * max_steps, 9]
    - actual_rewards: the target value for each state
    game_histories is a dictionary with arrays of shape [max_steps, B, ...] 
    """
    # We calculate the backward trace to assign rewards. Because we reset terminated states, games overlap inside a batch.
    # Backward propagation maps the future terminal step's reward back to the respective game steps.
    # Additionally, we discard games that have not ended yet.
    def scan_fn(carry, x):
        future_reward, future_is_tail = carry
        reward, terminated = x
        
        current_reward = jnp.where(terminated, reward, future_reward)
        current_is_tail = jnp.where(terminated, False, future_is_tail)
        
        next_carry = (current_reward, current_is_tail)
        return next_carry, next_carry
    
    rewards = game_histories['rewards']
    is_terminated = game_histories['is_terminated']
    init_carry = (jnp.zeros_like(rewards[0]), jnp.ones_like(is_terminated[0], dtype=jnp.bool_))
    _, (player_0_rewards, is_tail) = jax.lax.scan(scan_fn, init_carry, (rewards, is_terminated), reverse=True)

    actual_rewards = jnp.where(game_histories['current_player'] == 0, player_0_rewards, -player_0_rewards)
    
    # Flatten max_steps and B dimension 
    flatten = lambda x: x.reshape(-1, *x.shape[2:])
    training_data = {
        'states': flatten(game_histories['obs']).astype(jnp.bool_),
        'masks': flatten(game_histories['masks']).astype(jnp.bool_), 
        'mcts_probs': flatten(game_histories['mcts_probs']).astype(jnp.float16),
        'actual_rewards': actual_rewards.flatten().astype(jnp.int8), 
        'valid': (~is_tail).flatten()
    }
    
    # training_data is a dictionary containing arrays of shape [B * max_steps, ...] that we can use for training the network.
    return training_data

@functools.partial(jax.jit, static_argnames=("network", "optimizer"))
def train_step(network, optimizer, params, batch_stats, opt_state, batch):
    """
    Performs a single training step for the neural network for the given batch of data.
    The batch is a dictionary containing:
    - states: the states of the environment for each training example; shape: [B, 3, 3, 2]
    - masks: the legal actions for each state; shape: [B, 9]
    - mcts_probs: the target policy probabilities for each state (the output of MCTS); shape: [B, 9]
    - actual_rewards: the target value for each state (the actual reward received at the end of the game from the perspective of the current player); shape: [B]
    We calculate the loss for the batch and backpropagate it to update the parameters of the network using the optimizer.
    """
    def compute_loss(params):
        """
        Compute the loss for the given batch of data. The loss consists of two parts:
        - policy loss: the cross-entropy loss between the predicted policy logits and the target policy probabilities from MCTS.
          The aim is to train the network to predict the same policy as MCTS, which although uses the same network for calculating the logits,
          uses the search tree to calculate a better policy that takes into account the future rewards of the actions.
        - value loss: the mean squared error between the predicted values and the actual rewards received at the end of the game.
          In contrast to the policy loss, the value loss is not trying to match the value predicted by MCTS, 
          but rather trying to predict the actual reward received at the end of the game.
        """
        (logits, values), mutated_vars = network.apply(
            { 'params': params, 'batch_stats': batch_stats },
            batch['states'].astype(jnp.float32),
            batch['masks'],
            mutable=['batch_stats']
        )
        
        policy_loss = jnp.mean(optax.softmax_cross_entropy(logits, batch['mcts_probs'].astype(jnp.float32)))
        value_loss = jnp.mean(optax.l2_loss(values.flatten(), batch['actual_rewards'].astype(jnp.float32)))

        return policy_loss + value_loss, mutated_vars

    (loss_value, mutated_vars), grads = jax.value_and_grad(compute_loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    new_batch_stats = mutated_vars.get('batch_stats', batch_stats)
    return new_params, new_batch_stats, opt_state, loss_value
    
def train_loop(network, optimizer, params, batch_stats, opt_state, replay_buffer, num_batches, batch_size):
    losses = []
    for _ in range(num_batches):
        batch_indices = np.random.randint(0, replay_buffer.size, size=batch_size)
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x[batch_indices]), replay_buffer.buffer)
        params, batch_stats, opt_state, loss = train_step(
            network, optimizer, params, batch_stats, opt_state, batch
        )
        losses.append(loss)
        
    train_loss = jnp.mean(jnp.array(losses)).item()
    return params, batch_stats, opt_state, train_loss

def train(network, params, batch_stats, optimizer, opt_state, env, rng, 
          num_generations, self_plays_per_generation, batch_size, max_steps, num_simulations,
          replay_buffer_size, zero_temp_move, save_path=None, max_saved=5, load_checkpoint=True):
    checkpoint_files = []
    
    if load_checkpoint and save_path is not None and os.path.exists(save_path):
        checkpoint_files = [f for f in os.listdir(save_path) if f.startswith("bot_gen_") and f.endswith(".npy")]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[2].split(".")[0]))
            checkpoint = jnp.load(os.path.join(save_path, latest_checkpoint), allow_pickle=True).item()
            params = checkpoint['params']
            batch_stats = checkpoint.get('batch_stats', batch_stats)
            print(f"Loaded checkpoint: {latest_checkpoint}")
    
    replay_buffer = ReplayBuffer(replay_buffer_size)
    for generation in range(num_generations):
        start_time = time.time()
        
        rng, sp_rng = jax.random.split(rng) 
        game_histories = generate_self_play_data(env, network, params, batch_stats, sp_rng, batch_size=self_plays_per_generation, max_steps=max_steps, zero_temp_move=zero_temp_move, num_simulations=num_simulations)
        
        # Move game data to CPU right away to free up memory
        game_histories = jax.tree_util.tree_map(lambda x: jax.device_put(x, CPU_DEVICE), game_histories)
        training_data = generate_training_data(game_histories)
        training_data = jax.tree_util.tree_map(np.asarray, training_data)
        
        # Drop in-progress games and push to replay buffer
        valid_mask = training_data.pop('valid') != 0
        valid_data = {k: v[valid_mask] for k, v in training_data.items()}
        replay_buffer.add(valid_data)
        
        new_samples = valid_data["actual_rewards"].shape[0]
        num_batches = max(1, new_samples // batch_size)
        params, batch_stats, opt_state, train_loss = train_loop(network, optimizer, params, batch_stats, opt_state, replay_buffer, num_batches, batch_size)
        print(f"Generation {generation}, Loss: {train_loss}, Time: {time.time() - start_time:.2f}s")
        print(f"New samples this generation: {new_samples}, Replay buffer size: {replay_buffer.size}")
        
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            filename = f"bot_gen_{generation}.npy"
            jnp.save(os.path.join(save_path, filename), {'params': params, 'batch_stats': batch_stats})
            checkpoint_files.append(filename)
            if len(checkpoint_files) > max_saved:
                old_file = checkpoint_files.pop(0)
                os.remove(os.path.join(save_path, old_file))
        
    return params, batch_stats

def train_tic_tac_toe(num_generations=25, self_plays_per_generation=8192, batch_size=32, max_steps=9, num_simulations=32,
                      replay_buffer_size=200_000, zero_temp_move=5, save_path=None, max_saved=-1, load_checkpoint=True):
    rng = jax.random.PRNGKey(42)
    
    network = DenseNet()
    variables = network.init(rng, jnp.zeros((3, 3, 2)), jnp.ones(9))
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    env = pgx.make("tic_tac_toe")
    
    return train(network, params, batch_stats, optimizer, opt_state, env, rng, 
                 num_generations=num_generations, self_plays_per_generation=self_plays_per_generation, batch_size=batch_size, max_steps=max_steps, num_simulations=num_simulations,
                 replay_buffer_size=replay_buffer_size, zero_temp_move=zero_temp_move, save_path=save_path, max_saved=max_saved, load_checkpoint=load_checkpoint)

def train_chess(num_generations=10, self_plays_per_generation=1024, batch_size=4096, max_steps=512, num_simulations=400,
                replay_buffer_size=3_000_000, zero_temp_move=30, save_path=None, max_saved=10, load_checkpoint=True):
    rng = jax.random.PRNGKey(42)
    
    network = ResNet()
    variables = network.init(rng, jnp.zeros((1, 8, 8, 119)), jnp.ones((1, 4672)))
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    theoretical_max_batches = (self_plays_per_generation * num_generations * max_steps) // batch_size
    total_batches = theoretical_max_batches * 0.9
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=2e-3,
        warmup_steps=0.05*total_batches,
        decay_steps=0.95*total_batches,
        end_value=1e-6
    )
    
    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(params)
        
    env = pgx.make("chess")
    
    return train(network, params, batch_stats, optimizer, opt_state, env, rng, 
                 num_generations=num_generations, self_plays_per_generation=self_plays_per_generation, batch_size=batch_size, max_steps=max_steps, num_simulations=num_simulations,
                 replay_buffer_size=replay_buffer_size, zero_temp_move=zero_temp_move, save_path=save_path, max_saved=max_saved, load_checkpoint=load_checkpoint)

if __name__ == "__main__":
    train_chess(save_path='./chess_checkpoints')
