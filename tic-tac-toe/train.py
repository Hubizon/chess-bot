import functools
import time

import jax
import optax
import pgx
from jax import numpy as jnp
from pgx.experimental import auto_reset

from mcts import run_mcts
from network import AgentNet
from replay_buffer import ReplayBuffer

@functools.partial(jax.jit, static_argnames=("env", "network", "batch_size", "max_steps", "zero_temp_move", "num_simulations"))
def generate_self_play_data(env, network, params, batch_stats, rng, batch_size=32, max_steps=9, zero_temp_move=5, num_simulations=32):
    """Generates self-play data by running games with the current policy."""
    # Generate initial states for a batch of games
    rng, env_rng = jax.random.split(rng, 2)
    env_rngs = jax.random.split(env_rng, batch_size)
    initial_states = jax.vmap(env.init)(env_rngs)
    
    def make_move(states, step_rng):
        """Makes a move for each game in the batch using MCTS and returns the next states and the game history.
         - states: the current states of the batch of games; shape: [B, ...]
         - step_rng: a random key for this step; shape: [2]"""
        temperatures = jnp.where(states._step_count < zero_temp_move, 1.0, 0.0)[:, None]
        policy_outputs = run_mcts(
            step_rng, params, states, network, env.step, num_simulations=num_simulations, temperature=temperatures
        )
        batch_step_rngs = jax.random.split(step_rng, states.observation.shape[0])
        next_states = jax.vmap(auto_reset(env.step, env.init))(states, policy_outputs.action, batch_step_rngs)
        
        # current_games is a dictionary that gets stored in the game history
        current_games = {
            'states': states,
            'mcts_probs': policy_outputs.action_weights,
            'rewards': next_states.rewards[:, 0], # reward for player 0
            'is_terminated': next_states.terminated
        }
        
        return next_states, current_games
    
    step_rngs = jax.random.split(rng, max_steps)
    final_states, game_history = jax.lax.scan(make_move, initial_states, step_rngs)
    # game_history is a dictionary, where each element contains an array of shape [max_steps, B, ...] with data for each step of the games
    return game_history


@jax.jit
def generate_training_data(game_histories):
    """
    Generates training data from the game history. The training data is a dictionary of arrays, each of first dimension (B * max_steps):
    - states: the states of the environment for each training example; shape: [B * max_steps, 3, 3, 2]
    - masks: the legal actions for each state; shape: [B * max_steps, 9]
    - mcts_probs: the target policy probabilities for each state (the output of MCTS); shape: [B * max_steps, 9]
    - actual_rewards: the target value for each state
    game_histories is a dictionary with arrays of shape [max_steps, B, ...] 
    """
    rewards = game_histories['rewards']
    is_terminated = game_histories['is_terminated']
    
    # We calculate the backward trace to assign terminal rewards.
    # Because we reset terminated states, variable-length episodes overlap inside a batch.
    # Backward propagation exactly maps the future terminal step's reward back to the respective game steps.
    # Additionally, we want to discard the game that have not ended yet
    def scan_fn(carry, x):
        future_reward, future_is_tail = carry
        reward, terminated = x
        
        current_reward = jnp.where(terminated, reward, future_reward)
        current_is_tail = jnp.where(terminated, False, future_is_tail)
        
        next_carry = (current_reward, current_is_tail)
        return next_carry, next_carry
        
    init_carry = (jnp.zeros_like(rewards[0]), jnp.ones_like(is_terminated[0], dtype=jnp.bool_))
    _, (player_0_rewards, is_tail) = jax.lax.scan(scan_fn, init_carry, (rewards, is_terminated), reverse=True)

    actual_rewards = jnp.where(game_histories['states'].current_player == 0, player_0_rewards, -player_0_rewards)
    
    # Flatten max_steps and B dimension 
    flatten = lambda x: x.reshape(-1, *x.shape[2:])
    training_data = {
        'states': flatten(game_histories['states'].observation), # MUST extract `.observation`, otherwise network fails
        'masks': flatten(game_histories['states'].legal_action_mask), # Provide masks for the network explicitly
        'mcts_probs': flatten(game_histories['mcts_probs']),
        'actual_rewards': actual_rewards.flatten(),
        'valid': (~is_tail).flatten().astype(jnp.float32) # Weight 0 for tail games, 1 for valid games
    }
    
    # training_data is a dictionary containing arrays of shape [B * max_steps, ...] that we can use for training the network.
    return training_data

@functools.partial(jax.jit, static_argnames=("batch_size", "num_batches"))
def create_batches(buffer_size, batch_size, num_batches, rng):
    """Creates batches of training data by uniformly sampling from the replay buffer."""
    return jax.random.randint(rng, (num_batches, batch_size), 0, buffer_size)

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
            batch['states'],
            batch['masks'],
            mutable=['batch_stats']
        )
        
        policy_loss = jnp.mean(optax.softmax_cross_entropy(logits, batch['mcts_probs']))
        value_loss = jnp.mean(optax.l2_loss(values.flatten(), batch['actual_rewards']))

        return policy_loss + value_loss, mutated_vars

    (loss_value, mutated_vars), grads = jax.value_and_grad(compute_loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    new_batch_stats = mutated_vars.get('batch_stats', batch_stats)
    return new_params, new_batch_stats, opt_state, loss_value


@functools.partial(jax.jit, static_argnames=("network", "optimizer"))
def train_loop(network, optimizer, params, batch_stats, opt_state, buffer, batch_indices):
    def scan_train_step(carry, batch_indices):
        batch = jax.tree_util.tree_map(lambda x: jnp.take(x, batch_indices, axis=0), buffer)
        curr_params, curr_batch_stats, curr_opt_state = carry
        new_params, new_batch_stats, new_opt_state, loss = train_step(network, optimizer, curr_params, curr_batch_stats, curr_opt_state, batch)
        return (new_params, new_batch_stats, new_opt_state), loss
    
    # We need to use scan to loop sequentially instead of vmap that would try to parallelize the training steps
    (params, batch_stats, opt_state), losses = jax.lax.scan(scan_train_step, (params, batch_stats, opt_state), batch_indices)
    return params, batch_stats, opt_state, jnp.mean(losses)


def train(network, params, batch_stats, optimizer, opt_state, env, rng, 
          num_generations=10, self_plays_per_generation=1024, batch_size=32, max_steps=9,
          num_simulations=32,replay_buffer_size=50_000, zero_temp_move=5):    
    replay_buffer = ReplayBuffer(replay_buffer_size)
    for generation in range(num_generations):
        start_time = time.time()
        
        rng, sp_rng = jax.random.split(rng) 
        game_histories = generate_self_play_data(
            env, network, params, batch_stats, sp_rng, batch_size=self_plays_per_generation, 
            num_simulations=num_simulations, max_steps=max_steps, zero_temp_move=zero_temp_move
        )
        training_data = generate_training_data(game_histories)
        
        valid_mask = training_data.pop('valid') != 0
        valid_data = jax.tree_util.tree_map(lambda x: x[valid_mask], training_data)
        replay_buffer.add(valid_data)
        
        rng, batch_rng = jax.random.split(rng)
        num_batches = (self_plays_per_generation * max_steps) // batch_size
        batch_indices = create_batches(replay_buffer.size, batch_size, num_batches, batch_rng)
        
        params, batch_stats, opt_state, train_loss = train_loop(network, optimizer, params, batch_stats, opt_state, replay_buffer.buffer, batch_indices)
        print(f"Generation {generation}, Loss: {train_loss}, Time: {time.time() - start_time:.2f}s")
        
    return params, batch_stats

def train_tic_tac_toe(num_generations=10, self_plays_per_generation=1024, batch_size=32, max_steps=9,
                      num_simulations=32, replay_buffer_size=50_000, zero_temp_move=5):
    rng = jax.random.PRNGKey(42)
    
    network = AgentNet()
    variables = network.init(rng, jnp.zeros((3, 3, 2)), jnp.ones(9))
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    env = pgx.make("tic_tac_toe")
    
    return train(network, params, batch_stats, optimizer, opt_state, env, rng, 
                 num_generations=num_generations, self_plays_per_generation=self_plays_per_generation, batch_size=batch_size, max_steps=max_steps,
                 num_simulations=num_simulations, replay_buffer_size=replay_buffer_size, zero_temp_move=zero_temp_move)[0]
