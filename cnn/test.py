import torch
from cnn import ChessCNN, ExtendedChessCNN

# Test for the correct model input/output shapes.

BATCH_SIZE = 23

test_input = torch.randn(BATCH_SIZE, 12, 8, 8) # BATCH_SIZE x 12(features) x 8 x 8

model = ChessCNN()
from_squares, to_squares = model(test_input)

print("Non-extended output shapes:", from_squares.shape, to_squares.shape)

model = ExtendedChessCNN()
square, prom = model(test_input)

print("Extended output shapes:", square.shape, prom.shape)
print("Extracted moves: {}")