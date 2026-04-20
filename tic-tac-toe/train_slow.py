import random
from tqdm import tqdm

import pgx
import optax
import jax
from jax import numpy as jnp

from mcts import run_mcts
from network import AgentNet
from train import train_step

def assign_rewards_slow(game_history, player_0_reward):
    """Assigns rewards to each state in the game history from the perspective of the current player."""
    training_data = []
    
    for state, mcts_probs in game_history:
        # If current_player is 0, they get player_0_reward. If 1, they get the opposite.
        reward = player_0_reward if state.current_player == 0 else -player_0_reward
        training_data.append((state, mcts_probs, reward))
    
    return training_data

def train_slow(max_iters=1, self_plays_per_generation=4, batch_size=2):
    """
    The training loop for the network. It performs the following steps for a given number of iterations:
    1. Self-play: it plays games against itself using the current network and MCTS to generate training data.
       For each game, it makes moves using run_mcts until the game is terminated and then assigns rewards 
       to each state in the game history based on the winner of the game.
    2. Training: Trains the network on the generated training data.
    """
    rng = jax.random.PRNGKey(42)
    
    # Initialize the network and optimizer
    network = AgentNet()
    params = network.init(rng, jnp.zeros((3, 3, 2)), jnp.ones(9))['params']
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    # Create the environment
    env = pgx.make("tic_tac_toe")
    
    for generation in range(max_iters):
        experience_batch = []
        
        # 1. Self-play to generate training data
        for game_num in tqdm(range(self_plays_per_generation)):
            rng, init_rng = jax.random.split(rng)
            state = env.init(init_rng) # generate a random initial state of the environment
            game_history = []
            while not state.terminated:
                rng, mcts_rng = jax.random.split(rng, 2) # generate a new rng for MCTS to use in this turn
                batched_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), state) # add a batch dimension to the state
                policy_output = run_mcts(mcts_rng, params, batched_state, network, env.step) # run MCTS to get the policy output
                game_history.append((state, policy_output.action_weights[0])) # store the state and its corresponding MCTS policy
                action = policy_output.action[0] # shape: [B] -> scalar
                state = env.step(state, action)# take the action in the environment to get the next state
            player_0_reward = state.rewards[0] # reward for player 0: 1 (win), -1 (loss), 0 (draw)
            game_data = assign_rewards_slow(game_history, player_0_reward) 
            experience_batch.extend(game_data) # (state, mcts_probs, reward) for each state in the game history
        
        # 2. Training loop
        
        # Shuffle the training data and create batches
        random.shuffle(experience_batch)
        batches = [experience_batch[i:i+batch_size] for i in range(0, len(experience_batch), batch_size)]
        
        # For each batch, perform a training step and update the parameters of the network. 
        total_loss = 0.0
        for batch in batches:
            # Create a simple batch dict
            batch_data = {
                'states': jnp.stack([b[0].observation for b in batch]),
                'masks': jnp.stack([b[0].legal_action_mask for b in batch]),
                'mcts_probs': jnp.stack([b[1] for b in batch]),
                'actual_rewards': jnp.array([b[2] for b in batch])
            }
            
            params, opt_state, loss = train_step(network, optimizer, params, opt_state, batch_data)
            total_loss += loss
            
        print(f"Generation {generation}, Loss: {total_loss / len(batches)}")
        
    return params
