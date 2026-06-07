import torch
from cnn import ChessCNN

# Test for the correct model input/output shapes.

BATCH_SIZE = 23

test_input = torch.randn(BATCH_SIZE, 12, 8, 8) # BATCH_SIZE x 12(features) x 8 x 8

model = ChessCNN()
from_squares, to_squares = model(test_input)

print("Output shapes:", from_squares.shape, to_squares.shape)