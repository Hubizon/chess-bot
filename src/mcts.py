import functools

import jax
import mctx
from jax import numpy as jnp


@functools.partial(jax.jit, static_argnames=("network", "env_step", "num_simulations", "dirichlet_fraction", "dirichlet_alpha"))
def run_mcts(rng_key, params, batch_stats, env_state, network, env_step, 
             num_simulations, temperature=1.0, dirichlet_fraction=0.25, dirichlet_alpha=0.3):
    """
    Runs the Monte Carlo Tree Search (MCTS) algorithm to select the best action for the
    current state of the environment. It is called at each step of the game to determine the next move.
    Theoretically, we could remember the tree from the previous step and reuse part of it,
    but it's faster to just run the MCTS from scratch at each step of the game.
    Arguments:
    - rng_key: a random key for JAX's random number generator, used for any stochasticity in the MCTS (e.g., adding noise to the policy).
    - params: the parameters of the neural network used to evaluate the states and actions in the MCTS
    - env_state: the current state of the environment (batched), which includes the board state, legal actions, etc.
    - network: the neural network used to evaluate the states and actions in the MCTS
    - env_step: a function that takes the current state and an action, and returns the next state of the environment after the action
    - num_simulations: the number of simulations to run in the MCTS (the passes through the tree starting from the root)
    - temperature: the temperature for the softmax used in the final action selection (should be 0 during evaluation as argmax)
    - dirichlet_fraction: P_final = (1 - dirichlet_fraction) * P_network + dirichlet_fraction * Dirichlet_noise
    - dirichlet_alpha: the alpha parameter for the Dirichlet distribution used to add noise to the root node
    Returns:
    - PolicyOutput dataclass which contains:
      action: the chosen action to take; shape: [B]
      action_weights: the targets used to train a policy network (probabilities of the next actions from which the action is sampled); shape: [B, num_actions] 
      search_tree: the search tree of the finished search; shape: [B, ...]
    """
    def root_fn(params, state):
        """
        A function used in mctx to evaluate the current state before expanding the tree.
        mctx.RootFnOutput is a simple dataclass that contains:
        - policy logits of the state (so that it knows where to move next); shape: [B, 3 * 3] (the number of actions)
        - the value of the state (the initial quality of the state); shape: [B, 1] (the value of the state)
        state is the state of the environment (from pgx) and it contains among other things:
        - observation: the current state of the board; shape: [B, 3, 3, 2] 
        - legal_action_mask: a mask of legal actions (1 for legal, 0 for illegal); shape: [B, 3 * 3]
        """
        logits, value = network.apply(
            {'params': params, 'batch_stats': batch_stats}, state.observation, state.legal_action_mask, train=False
        )

        return mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state
        )

    def recurrent_fn(params, rng, action, state):
        """
        A function used in mctx to evaluate the current state after taking an action and expanding the tree.
        It calculates the policy logits and value of the new state to use in future simulations, 
        and the reward received to backpropagate it up the tree. For non-terminal states, the reward is 0 and discount is 1,
        while for terminal states, the reward is the final reward of the game and the discount is 0 (to ignore the predicted value of the terminal state).
        mctx.RecurrentFnOutput is a simple dataclass that contains:
        - reward: the reward received after taking the action, from the perspective of the player who just took the action; shape: [B]
        - discount: the discount to apply to the value of the next state (since this is a finite game, it's 1 for non-terminal states)
        - prior_logits: the policy logits of the next state (so that it knows where to move next in future simulations); shape: [B, num_actions]
        - value: the value of the next state (the expected score of the next state from the perspective of the player who just took the action); shape: [B]
        next_state: the next state of the environment after taking the action; shape: [B, ...]
        """
        # Calculate the next state of the environment after taking the action
        next_state = jax.vmap(env_step)(state, action) # shape: [B]
        
        logits, value = network.apply(
            {'params': params, 'batch_stats': batch_stats}, next_state.observation, next_state.legal_action_mask, train=False
        )
        
        # The reward is from the perspective of the player who just took the action, and is needed to calculate the Q value of this state-action pair.
        # Q(state, action) is the expected future reward starting from the state and taking the action:
        # Q(state, action) = reward + discount * value(next_state)
        
        # state.rewards in pgx is of shape [B, num_players], and holds the reward for each player. 
        # state.current_player is the player who took the action to reach next_state
        batch_size = next_state.rewards.shape[0]
        reward = next_state.rewards[jnp.arange(batch_size), state.current_player]
        discount = -jnp.where(next_state.terminated, 0.0, 1.0)
        
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value
        ), next_state
    
    # We create the root of the tree, which evaluates the current state of the environment
    # Note: the muzero_policy adds dirichlet noise to the root node (and only the root node) to encourage exploration.
    root = root_fn(params, env_state)
    
    """
    For each move, we create a new tree and run num_simulations simulations to calculate the best action to take.
    The initial tree is built using the root, which contains the initial state and its logits and value. 
    The logits are used to calculate the policy at the root, which is used to select the next action (P_network).
    Additionally, solely for the root, we add Dirichlet Noise to the policy to encourage exploration from the root:
    P_final = (1 - dirichlet_fraction) * P_network + dirichlet_fraction * Dirichlet_noise
    The value isn't that important since it's only used to calculate the Q value of the root, which isn't used for action selection. 
    
    For each simulation, Monte Carlo Tree Search performs the following steps:
    1. Selection: Starting from the current state, it travels down the tree by selecting actions according to the PUCT formula 
       until it reaches a leaf node (a node that hasn't been expanded yet).
    2. Expansion & Evaluation: If we are in state `s` and about to take an action `a` that leads to a leaf node, we call recurrent_fn(s, a)
       to calculate the potential reward received after taking the action.
       After that, the new state is considered expanded and its children are added to the tree as new leaf nodes.
       If the leaf node is not a terminal state, it expands the node by calling the recurrent_fn to calculate
       the policy logits and value of the new state, and adds the new node to the tree.
    3. Backpropagation: After the expansion and evaluation step, we don't go further down the tree, but backpropagate the
       reward received up the tree. If our trajectory in the tree was s_0 -> a_0 -> s_1 -> a_1 -> ... -> s_n
       then G_n = Value(s_n) and G_i = Reward(s_i, a_i) + discount * G_{i+1}. Additionally, we update the tree statistics:
       N(s_i, a_i) += 1 and W(s_i, a_i) += G_i and Q(s_i, a_i) = W(s_i, a_i) / N(s_i, a_i)
       Note: Reward(s_i, a_i) is always 0 except for the last step, when s_i -> a_i -> a terminal state
       
    MCTX's muzero_policy uses PUCT formula to select actions during the tree search:
     PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
     where Q(s, a) is the value of the state-action pair (calculated as the average of the rewards received in the simulations that went through this state-action pair),
     P(s, a) is the prior probability of taking action a in state s (given by the neural network),
     N(s) is the number of times the state s has been visited in the simulations,
     N(s, a) is the number of times the action a has been taken from state s in the simulations,
     c_puct is a hyperparameter that controls the level of exploration (higher c_puct encourages more exploration).
    """
    policy_output = mctx.muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        temperature=temperature,
        dirichlet_fraction=dirichlet_fraction,
        dirichlet_alpha=dirichlet_alpha,
        invalid_actions=~env_state.legal_action_mask,
        qtransform=mctx.qtransform_by_parent_and_siblings
    )
    
    return policy_output
