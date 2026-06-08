import torch
from cnn import ChessCNN, ChessDataset
from datetime import datetime
from torch.utils.data import DataLoader
import chess
import chess.pgn
import numpy as np
import argparse

# Transforms a board state into a datapoint.
def board_to_datapoint(board):
    datapoint = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    
    for square, piece in piece_map.items():
        file = chess.square_file(square) # a-h -> 0-7
        rank = chess.square_rank(square) # 1-8 -> 0-7
        piece_type = piece.piece_type - 1 # 1-6 to 0-5 (6-11)
        datapoint[piece_type if (piece.color == chess.WHITE) else (piece_type+6), file, rank] = 1.0

    return datapoint

def file_to_data(filename):
    pgn_file = open(filename)
    X_data = []
    y_data = []

    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break

        result = game.headers.get("Result")

        winner_color = chess.WHITE if (result == "1-0") else chess.BLACK

        board = game.board()

        for move in game.mainline_moves():
            if board.turn == winner_color:
                if winner_color == chess.WHITE:
                    board_perspective = board
                    from_square = move.from_square
                    to_square = move.to_square
                else:
                    board_perspective = board.mirror() # Board is mirrored when predicting black's moves.
                    # print(f"Original square: {chess.square_name(move.from_square)} Mirrored: {chess.square_name(chess.square_mirror(move.from_square))}")
                    from_square = chess.square_mirror(move.from_square)
                    to_square = chess.square_mirror(move.to_square)
                X_data.append(board_to_datapoint(board_perspective))
                y_data.append((from_square, to_square))
            board.push(move)

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    print(f"Input from {filename} completed.")
    return X_data, y_data

def train_one_epoch(model, data_loader, optimizer, loss_fn, epoch_index, progcheck=64):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(data_loader):
        boards, target_from, target_to = data

        optimizer.zero_grad()

        predicted_from, predicted_to = model(boards)

        loss_from = loss_fn(predicted_from, target_from)
        loss_to = loss_fn(predicted_to, target_to)
        loss = loss_from + loss_to

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % progcheck == progcheck-1:
            last_loss = running_loss / progcheck
            #print(f'Batch {i // progcheck + 1} loss: {last_loss}')
            running_loss = 0.

    print(f"Epoch {epoch_index} completed.")
    return last_loss

def train(train_file, valid_file, epochs=25, batchsize=64, progcheck=64):

    model = ChessCNN()
    loss_fn = torch.nn.CrossEntropyLoss()

    train_X, train_y = file_to_data(train_file)
    valid_X, valid_y = file_to_data(valid_file)
    train_dataset = ChessDataset(train_X, train_y)
    valid_dataset = ChessDataset(valid_X, valid_y)
    train_data_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 1_000_000

    print("Starting training loop.")
    print('-'*20)

    for epoch in range(epochs):

        all_guesses = 0
        correct_guesses = 0
        correct_piece_match = 0
        piece_correct_guesses = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        piece_all_guesses = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

        print(f"Starting epoch: {epoch}")
        model.train(True)
        avg_loss = train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch, progcheck)

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(valid_data_loader):
                vinputs, v_from, v_to = vdata
                vprediction_from, vprediction_to = model(vinputs)
                vloss_from = loss_fn(vprediction_from, v_from)
                vloss_to = loss_fn(vprediction_to, v_to)
                running_vloss += (vloss_from + vloss_to)

                for id in range(batchsize):
                    all_guesses += 1
                    curr_pred_from = int(torch.argmax(vprediction_from[id]))
                    curr_pred_to = int(torch.argmax(vprediction_to[id]))
                    curr_actual_from = int(v_from[id])
                    curr_actual_to = int(v_to[id])

                    piece_type = None
                    for pc in range(6):
                        if vinputs[id][pc][chess.square_file(curr_actual_from)][chess.square_rank(curr_actual_from)] != 0:
                            piece_type = pc+1

                    piece_all_guesses[piece_type] += 1
                    if vinputs[id][piece_type-1][chess.square_file(curr_pred_from)][chess.square_rank(curr_pred_from)] != 0:
                        correct_piece_match += 1
                        
                    if(curr_actual_from == curr_pred_from and curr_actual_to == curr_pred_to):
                        correct_guesses += 1
                        piece_correct_guesses[piece_type] += 1

            avg_vloss = running_vloss / (i+1)
            print('-'*20)
            print(f'Epoch: {epoch}')
            print(f'\tTraining loss: {avg_loss:.6f}')
            print(f'\tValidation loss: {avg_vloss:.6f}')
            print(f'\tAccuracy: {correct_guesses}/{all_guesses} {100*correct_guesses/all_guesses:.3f}%')
            print(f'\tPiece type matched: {correct_piece_match}/{all_guesses} {100*correct_piece_match/all_guesses:.3f}%')
            for pc in range(1,7):
                print(f'\t{chess.piece_name(pc)} move accuracy: {piece_correct_guesses[pc]}/{piece_all_guesses[pc]} {100*piece_correct_guesses[pc]/piece_all_guesses[pc]:.3f}%')
            print('-'*20)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_{timestamp}_{epoch}.model'
            save_training_checkpoint(epoch, model, optimizer, batchsize, progcheck)
            torch.save(model.state_dict(), model_path)

def save_training_checkpoint(epoch, model, optimizer, batchsize, progcheck):
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_size': batchsize,
        'progress_check': progcheck
    }
    torch.save(checkpoint_data, "training_checkpoint_epoch"+str(epoch)+".tar")

def load_training_checkpoint(epoch, model, optimizer):
    checkpoint_data = torch.load("chess_checkpoint_epoch"+str(epoch)+".tar")
    model.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    return checkpoint_data['epoch'], checkpoint_data['batch_size'], checkpoint_data['progress_check']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="../data/parsed/train.pgn")
    parser.add_argument("--valid", type=str, default="../data/parsed/valid.pgn")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--progcheck", type=int, default=64)
    args = parser.parse_args()
    train(args.train, args.valid, args.epochs, args.batchsize, args.progcheck)