import tkinter as tk

import jax
import jax.numpy as jnp
import pgx

from .mcts import run_mcts


class TicTacToeGUI:
    def __init__(self, network, params, batch_stats, player_x=True, num_simulations=32):
        self.network = network
        self.params = params
        self.batch_stats = batch_stats
        self.board = [0] * 9  # 0 for empty, 1 for X, -1 for O
        self.player_x = player_x
        self.num_simulations = num_simulations
        self.env = pgx.make("tic_tac_toe")
        self.state = self.env.init(jax.random.PRNGKey(0))
        self.rng = jax.random.PRNGKey(0)
        self.window = tk.Tk()
        self.buttons = []
        self._create_board()
        if not self.player_x:
            self._bot_move()
        
    def _create_board(self):
        for i in range(9):
            btn = tk.Button(self.window, text="", font=("Arial", 24), width=5, height=2,
                                command=lambda idx=i: self._on_click(idx))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)
    
    def _end_game(self):
        winner = "X" if self.state.rewards[0] == 1 else "O" if self.state.rewards[0] == -1 else "Draw"
        result_text = f"Game Over! Winner: {winner}"
        result_label = tk.Label(self.window, text=result_text, font=("Arial", 16))
        result_label.grid(row=3, column=0, columnspan=3)
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)
    
    def _make_move(self, action, player):
        self.board[action] = 1 if player else -1
        self.buttons[action].config(text="X" if player else "O")
        self.state = self.env.step(self.state, action)
        if self.state.terminated:
            self._end_game()
            return True
        return False
    
    def _on_click(self, idx):
        if self.board[idx] != 0:
            return
        
        if not self._make_move(idx, self.player_x):
            self._bot_move()
        
    def _bot_move(self):
        rng, mcts_rng = jax.random.split(self.rng)
        self.rng = rng
        
        batched_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), self.state)
        if self.num_simulations == -1:
            masked_logits, value = self.network.apply(
                { 'params': self.params, 'batch_stats': self.batch_stats },
                batched_state.observation,
                batched_state.legal_action_mask,
                train=False
            )
            action = jnp.argmax(masked_logits[0])
            weights_str = "[" + ", ".join(f"{float(x):.4f}" if x != -1e9 else "illegal" for x in masked_logits[0]) + "]"
            print(f"action weights: {weights_str}")
            print(f"value: {float(value[0]):.4f}")
            self._make_move(int(action), not self.player_x)
        else:
            policy_output = run_mcts(mcts_rng, self.params, self.batch_stats, batched_state, self.network, self.env.step, 
                                    num_simulations=self.num_simulations, temperature=0.0, dirichlet_fraction=0.0)
            weights_str = "[" + ", ".join(f"{float(x):.4f}" for x in policy_output.action_weights[0]) + "]"
            print(f"action weights: {weights_str}")
            self._make_move(int(policy_output.action[0]), not self.player_x)

    def run(self):
        self.window.mainloop()