import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import chess

def extract_move(y, prom):
    from_square = chess.Square(chess.square_file(y//64), chess.square_rank(y//64))
    to_square = chess.Square(chess.square_file(y%64), chess.square_rank(y%64))
    return chess.Move(from_square, to_square)

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

# Extended convolutional neural network
# Quadruple [conv-relu] - [affine]x2 with 64/128 and 64**2+5 features
class ExtendedChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # [conv-relu]
        # 12 input channels (6 pieces for 2 colors)
        # Kernel size 3x3 requires padding=1 to scan the entire board
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        # [affine]
        # Input from the convolution layers, output in 64**2+5 format
        self.affine1 = nn.Linear(in_features=(128*8*8), out_features=64**2+5)
        self.affine2 = nn.Linear(in_features=64**2+5, out_features=64**2+5)

    def forward(self, x):
        # Input tensor shape: [batch_size x 12 x 8 x 8]
        
        # Apply convolution layers with relu
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten the tensor for the affine layers 
        x = x.view(x.size(0), 128*8*8)
        
        # Apply first [affine] with relu
        x = F.relu(self.affine1(x))
        
        # Apply second [affine]
        output = self.affine2(x)

        move = output[:, :64**2]
        promotion = output[:, 64**2:]
        
        return move, promotion


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

class ExtendedChessDataset(Dataset):
    def __init__(self, X_data, y_data, prom_data):
        # X_data: [data_size x 12 x 8 x 8] - board state
        # y_data: [data_size] - true move label (in 64**2 format)
        # prom_data: [data_size x 5] - promotion indicator (no promotion,K,B,R,Q)
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)
        self.prom = torch.tensor(prom_data, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.prom[idx], 