# PictureBot

A machine learning project for exploring the capabilities of reinforcement learning and convolutional neural networks in chess.
This project does not include a GUI, however the CNN-based model can communicate with most GUIs via [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface).
The final result of the CNN-based architecture training resulted in a bot rated around 1900 on Lichess. See its performance on [Lichess](https://lichess.org/@/PictureBot).

## Files

* assets - the icon for the bot
* rl - files for training and running a bot based on reinforcement learning
* cnn - files for training a bot based on CNNs. This also includes:
  * data - PGNs containing master chess games as training data. The unfiltered games ought to be put into the /raw folder.
  * adapters - Adapters for UCI to communicate with the bot. They may be used by a GUI after training the model and compiling run_bot.
