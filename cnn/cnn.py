import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# 3-layer convolutional neural network
# [conv-relu] - [affine]x2 with 32 and 128 features
class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # [conv-relu]
        # 12 input channels (6 pieces for 2 colors), output 32 features
        # Kernel size 3x3 requires padding=1 to scan the entire board
        self.conv = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        
        # First [affine]
        # Input from [conv-relu] as 8x8 board of 32 features, output 128 features
        self.affine1 = nn.Linear(in_features=(32*8*8), out_features=128)
        
        # Second [affine]
        # Input from first [affine] as 128 features, output the final 128 features
        # The first 64 outputs are for the from square and the second 64 for the to square.
        self.affine2 = nn.Linear(in_features=128, out_features=128)

    def forward(self, x):
        # Input tensor shape: [batch_size x 12 x 8 x 8]
        
        # Apply convolution layer with relu
        x = F.relu(self.conv(x))

        # Flatten the tensor for the affine layers 
        x = x.view(x.size(0), 32*8*8)
        
        # Apply first [affine] with relu
        x = F.relu(self.affine1(x))
        
        # Apply second [affine]
        output = self.affine2(x)
        
        from_features = output[:, :64]
        to_features = output[:, 64:]
        
        return from_features, to_features

# Dataset requires __init__, __len__ and __getitem__    
class ChessDataset(Dataset):
    def __init__(self, X_data, y_data):
        # X_data: [data_size x 12 x 8 x 8] - board state
        # y_data: [data_size x 2] - true move label (from, to)
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y_from = torch.tensor(y_data[:,0], dtype=torch.long)
        self.y_to = torch.tensor(y_data[:,1], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_from[idx], self.y_to[idx]