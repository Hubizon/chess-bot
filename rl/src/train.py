import os
import functools
import time

import pgx
import optax
import jax
import numpy as np
from jax import numpy as jnp
from pgx.experimental import auto_reset
import orbax.checkpoint as ocp

from .mcts import run_mcts
from .densenet import DenseNet
from .resnet import ResNet
from .replay_buffer import ReplayBuffer

CPU_DEVICE = jax.devices("cpu")[0]

@functools.partial(jax.jit, static_argnames=("env", "network", "batch_size", "max_steps", "num_simulations", "max_num_considered_actions"))
def generate_self_play_data(env, network, params, batch_stats, rng, batch_size, max_steps, zero_temp_move, num_simulations, max_num_considered_actions):
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
        policy_outputs = run_mcts(step_rng, params, batch_stats, states, network, env.step, num_simulations, max_num_considered_actions)
        best_actions = jnp.where(states._step_count < zero_temp_move, policy_outputs.action, jnp.argmax(policy_outputs.action_weights, axis=-1))
        next_states = jax.vmap(auto_reset(env.step, env.init))(states, best_actions, init_rngs)
        
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
    # game_history is a dictionary, where each element contains the data for one step of the game for the whole batch of games of shape [max_steps, B, ...]
    return game_history


@jax.jit
def generate_training_data(game_histories):
    """
    Generates training data from the game history. The training data is a dictionary of flattened arrays, each containing:
    - states: the states of the environment for each training example; shape: [B * max_steps, ...]
    - mcts_probs: the target policy probabilities for each state (the output of MCTS); shape: [B * max_steps, num_actions]
    - rewards: the target value for each state
    game_histories is a dictionary with arrays of shape [max_steps, B, ...] 
    """
    # We calculate the backward trace to assign terminal rewards.
    # Because we reset terminated states, variable-length episodes overlap inside a batch.
    # Backward propagation exactly maps the future terminal step's reward back to the respective game steps.
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
    
    # training_data is a dictionary containing lists of shape [B*max_steps, ...] that we can use for training the network.
    return training_data

@functools.partial(jax.jit, static_argnames=("network", "optimizer"))
def train_step(network, optimizer, params, batch_stats, opt_state, batch):
    """
    Performs a single training step for the neural network for the given batch of data.
    The batch is a dictionary containing:
    - states: the states of the environment for each training example; shape: [B, ...]
    - masks: the legal actions for each state; shape: [B, num_actions]
    - mcts_probs: the target policy probabilities for each state (the output of MCTS); shape: [B, num_actions]
    - actual_rewards: the target value for each state (the actual reward received at the end of the game from the perspective of the current player); shape: [B]
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

        total_loss = policy_loss + value_loss
        return total_loss, (policy_loss, value_loss, mutated_vars)

    (loss_value, (policy_loss, value_loss, mutated_vars)), grads = jax.value_and_grad(compute_loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    new_batch_stats = mutated_vars.get('batch_stats', batch_stats)
    return new_params, new_batch_stats, opt_state, loss_value, policy_loss, value_loss
    
def train_loop(network, optimizer, params, batch_stats, opt_state, replay_buffer, num_batches, batch_size):
    losses = []
    policy_losses = []
    value_losses = []
    for _ in range(num_batches):
        batch_indices = np.random.randint(0, replay_buffer.size, size=batch_size)
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x[batch_indices]), replay_buffer.buffer)
        params, batch_stats, opt_state, loss, policy_loss, value_loss = train_step(
            network, optimizer, params, batch_stats, opt_state, batch
        )
        losses.append(loss)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        
    train_loss = jnp.mean(jnp.array(losses)).item()
    train_policy_loss = jnp.mean(jnp.array(policy_losses)).item()
    train_value_loss = jnp.mean(jnp.array(value_losses)).item()

    return params, batch_stats, opt_state, train_loss, train_policy_loss, train_value_loss

def create_checkpoint_dict(start_generation, params, batch_stats, opt_state):
    return {
        'step': start_generation,
        'params': params,
        'batch_stats': batch_stats,
        'opt_state': opt_state
    }

def load_checkpoint(manager, item=None):
    latest_step = manager.latest_step()
    if latest_step is not None:
        if item is not None:
            return manager.restore(latest_step, args=ocp.args.StandardRestore(item=item))
        return manager.restore(latest_step, args=ocp.args.StandardRestore())
    return item

def train(network, params, batch_stats, optimizer, opt_state, env, rng, 
          num_generations, self_plays_per_generation, batch_size, max_steps, num_simulations, max_num_considered_actions,
          replay_buffer_size, zero_temp_move, checkpoint_path=None, max_saved=5):
    start_generation = 0
    replay_buffer = ReplayBuffer(replay_buffer_size)
    if checkpoint_path is not None:
        checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=max_saved, create=True)
        checkpoint_dir = os.path.abspath(checkpoint_path)
        checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=checkpoint_options)
        has_checkpoint = checkpoint_manager.latest_step() is not None
        state_dict = load_checkpoint(checkpoint_manager, create_checkpoint_dict(0, params, batch_stats, opt_state))
        
        rb_dir = os.path.abspath(f"{checkpoint_path}_rb")
        rb_manager = ocp.CheckpointManager(rb_dir, options=ocp.CheckpointManagerOptions(max_to_keep=2, create=True))
        rb_state_dict = load_checkpoint(rb_manager)
        if rb_state_dict is not None:
            replay_buffer.load_state_dict(rb_state_dict)
        
        start_generation = state_dict['step'] + 1 if has_checkpoint else 0
        params = state_dict['params']
        batch_stats = state_dict['batch_stats']
        opt_state = state_dict['opt_state']
        if start_generation > 0:
            print(f"Loaded checkpoint from generation {start_generation-1}.")
    
    for generation in range(start_generation, num_generations):
        start_time = time.time()
        
        print(f"Generation {generation}/{num_generations}:", flush=True)
        rng, sp_rng = jax.random.split(rng) 
        game_histories = generate_self_play_data(
            env, network, params, batch_stats, sp_rng, batch_size=self_plays_per_generation, max_steps=max_steps, 
            zero_temp_move=zero_temp_move, num_simulations=num_simulations, max_num_considered_actions=max_num_considered_actions)

        terminal_mask = game_histories['is_terminated']
        terminal_rewards = game_histories['rewards']
        games_finished = int(jnp.sum(terminal_mask))
        player0_wins = int(jnp.sum(jnp.logical_and(terminal_mask, terminal_rewards > 0)))
        player1_wins = int(jnp.sum(jnp.logical_and(terminal_mask, terminal_rewards < 0)))
        draws = int(jnp.sum(jnp.logical_and(terminal_mask, terminal_rewards == 0)))
        print(f"  Self-play data generation completed. Time: {time.time() - start_time:.2f}s", flush=True)
        
        game_histories = jax.tree_util.tree_map(lambda x: jax.device_put(x, CPU_DEVICE), game_histories)
        training_data = generate_training_data(game_histories)
        training_data = jax.tree_util.tree_map(np.asarray, training_data)
        
        valid_mask = training_data.pop('valid') != 0
        valid_data = {k: v[valid_mask] for k, v in training_data.items()}
        replay_buffer.add(valid_data)

        new_samples = valid_data["actual_rewards"].shape[0]
        print(f"  Generated {new_samples} new samples. Time: {time.time() - start_time:.2f}s", flush=True)
        num_batches = max(1, new_samples // batch_size)
        params, batch_stats, opt_state, train_loss, policy_loss, value_loss = train_loop(network, optimizer, params, batch_stats, opt_state, replay_buffer, num_batches, batch_size)
        jax.clear_caches()
        print(f"  Total Loss: {train_loss}, Value Loss: {value_loss}, Policy Loss: {policy_loss}")
        print(f"  Finished games: {games_finished}, Player0 wins: {player0_wins}, Player1 wins: {player1_wins}, Draws: {draws}")
        print(f"  New samples this generation: {new_samples}. Replay buffer size: {replay_buffer.size}")
        
        if checkpoint_path is not None:
            state_dict = create_checkpoint_dict(generation, params, batch_stats, opt_state)
            checkpoint_manager.save(generation, args=ocp.args.StandardSave(item=state_dict))
            rb_manager.save(generation, args=ocp.args.StandardSave(item=replay_buffer.state_dict()))
        print(f"  Total time: {time.time() - start_time:.2f}s")
        
    if checkpoint_path is not None:
        checkpoint_manager.wait_until_finished()
        rb_manager.wait_until_finished()
        
    return params, batch_stats

def train_tic_tac_toe(num_generations=25, self_plays_per_generation=8192, batch_size=32, max_steps=16, num_simulations=16, max_num_considered_actions=4,
                      replay_buffer_size=250_000, zero_temp_move=5, checkpoint_path=None, max_saved=5):
    rng = jax.random.PRNGKey(42)
    
    network = DenseNet()
    variables = network.init(rng, jnp.zeros((3, 3, 2)), jnp.ones(9))
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    env = pgx.make("tic_tac_toe")
    
    return train(network, params, batch_stats, optimizer, opt_state, env, rng, 
                 num_generations=num_generations, self_plays_per_generation=self_plays_per_generation, batch_size=batch_size, max_steps=max_steps, num_simulations=num_simulations, max_num_considered_actions=max_num_considered_actions,
                 replay_buffer_size=replay_buffer_size, zero_temp_move=zero_temp_move, checkpoint_path=checkpoint_path, max_saved=max_saved)

def train_chess(num_generations=200, self_plays_per_generation=2048, batch_size=4096, max_steps=512, num_simulations=50, max_num_considered_actions=16,
                replay_buffer_size=4_500_000, zero_temp_move=30, checkpoint_path=None, max_saved=25):
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
                 num_generations=num_generations, self_plays_per_generation=self_plays_per_generation, batch_size=batch_size, max_steps=max_steps, num_simulations=num_simulations, max_num_considered_actions=max_num_considered_actions,
                 replay_buffer_size=replay_buffer_size, zero_temp_move=zero_temp_move, checkpoint_path=checkpoint_path, max_saved=max_saved)

if __name__ == "__main__":
    train_chess(checkpoint_path='./checkpoints')
