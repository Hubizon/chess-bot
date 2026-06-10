import torch
from cnn import ChessCNN, ChessDataset, ExtendedChessCNN, ChessResNet, ExtendedChessDataset, extract_move
from datetime import datetime
from torch.utils.data import DataLoader
import chess
import chess.pgn
import numpy as np
import argparse
import time

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

def file_to_data(filename, extended, max_games=None, progress_check=5000):
    time_start = time.time()
    pgn_file = open(filename)
    X_data = []
    y_data = []
    prom_data = []
    games_read = 0

    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        games_read += 1
        if max_games is not None and games_read > max_games:
            break
        if games_read % progress_check == 0:
            print(f"  {filename}: {games_read:,} games, {len(X_data):,} positions")

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
                if extended:
                    y_data.append(from_square*64+to_square)
                    prom_node = np.zeros(5, dtype=np.float32)
                    if move.promotion is not None:
                        prom_node[move.promotion-1] = 1.0
                    else:
                        prom_node[0] = 1.0
                    prom_data.append(prom_node)
                else:
                    y_data.append((from_square, to_square))
            board.push(move)

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    print(f"Input from {filename} completed: {len(X_data):,} positions in {time.time()-time_start:.1f}s.")
    if extended:
        prom_data = np.array(prom_data)
        return X_data, y_data, prom_data
    else:
        return X_data, y_data

def train_one_epoch(model, data_loader, optimizer, loss_fn, epoch_index, device, prog_check=10000, extended=False, scheduler=None):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(data_loader):
        if extended:
            boards, target_move, target_prom = data
            boards = boards.to(device)
            target_move = target_move.to(device)
            target_prom = target_prom.to(device)
        else:
            boards, target_from, target_to = data
            boards = boards.to(device)
            target_from = target_from.to(device)
            target_to = target_to.to(device)

        optimizer.zero_grad()

        if extended:
            predicted_move, predicted_prom = model(boards)
        else:
            predicted_from, predicted_to = model(boards)

        if extended:
            loss = loss_fn(predicted_move, target_move)
            loss += loss_fn(predicted_prom, target_prom)
        else:
            loss = loss_fn(predicted_from, target_from)
            loss += loss_fn(predicted_to, target_to)

        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        if i % prog_check == prog_check-1:
            last_loss = running_loss / prog_check
            print(f"  batch {i+1:,} (epoch {epoch_index}) train loss: {last_loss:.6f}")
            running_loss = 0.

    print(f"Epoch {epoch_index} completed.")
    return last_loss

def train(train_file, valid_file, epochs=10, batch_size=64, prog_check=10000, extended=False, resnet=False, max_games=None):
    if resnet:
        extended = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    if resnet:
        model = ChessResNet()
    elif extended:
        model = ExtendedChessCNN()
    else:
        model = ChessCNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    data_time_start = time.time()
    if extended:
        train_X, train_y, train_prom = file_to_data(train_file, extended=True, max_games=max_games)
        valid_X, valid_y, valid_prom = file_to_data(valid_file, extended=True, max_games=max_games)
        train_dataset = ExtendedChessDataset(train_X, train_y, train_prom)
        valid_dataset = ExtendedChessDataset(valid_X, valid_y, valid_prom)
        del train_X, train_y, train_prom, valid_X, valid_y, valid_prom
    else:
        train_X, train_y = file_to_data(train_file, extended=False, max_games=max_games)
        valid_X, valid_y = file_to_data(valid_file, extended=False, max_games=max_games)
        train_dataset = ChessDataset(train_X, train_y)
        valid_dataset = ChessDataset(valid_X, valid_y)
        del train_X, train_y, valid_X, valid_y
    print(f"Total data loading: {time.time()-data_time_start:.1f}s.")

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001) if resnet else torch.optim.Adam(model.parameters(), lr=0.001)
    total_steps = len(train_data_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6) if resnet else None

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
        epoch_time_start = time.time()
        model.train(True)
        train_time_start = time.time()
        avg_loss = train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch, device, prog_check, extended, scheduler)
        train_secs = time.time() - train_time_start

        running_vloss = 0.0
        model.eval()

        valid_time_start = time.time()
        with torch.no_grad():
            for i, vdata in enumerate(valid_data_loader):
                if extended:
                    vinputs, v_move, v_prom = vdata
                    vinputs = vinputs.to(device)
                    v_move = v_move.to(device)
                    v_prom = v_prom.to(device)
                    vprediction_move, vprediction_prom = model(vinputs)
                    vloss_move = loss_fn(vprediction_move, v_move)
                    vloss_prom = loss_fn(vprediction_prom, v_prom)
                    running_vloss += (vloss_move + vloss_prom).item()
                else:
                    vinputs, v_from, v_to = vdata
                    vinputs = vinputs.to(device)
                    v_from = v_from.to(device)
                    v_to = v_to.to(device)
                    vprediction_from, vprediction_to = model(vinputs)
                    vloss_from = loss_fn(vprediction_from, v_from)
                    vloss_to = loss_fn(vprediction_to, v_to)
                    running_vloss += (vloss_from + vloss_to).item()

                for id in range(vinputs.size(0)):
                    all_guesses += 1
                    if extended:
                        curr_pred_move = int(torch.argmax(vprediction_move[id]))
                        curr_pred_from = curr_pred_move//64
                        curr_pred_to = curr_pred_move%64
                        curr_actual_from = int(v_move[id]//64)
                        curr_actual_to = int(v_move[id]%64)
                        vprediction_prom_piece = int(torch.argmax(vprediction_prom[id][1:]))+1
                    else:
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
                        if extended and v_prom[id][0] == 0.0 and v_prom[id][vprediction_prom_piece] == 0.0:
                            # print("Incorrect promotion guess.")
                            # print(f"Valid promotion table: {v_prom[id]}")
                            # print(f"Vprediction: {vprediction_prom}")
                            # print(f"Predicted piece (if any): {vprediction_prom_piece}")
                            piece_correct_guesses[piece_type] -= 1

            avg_vloss = running_vloss / (i+1)
            print('-'*20)
            print(f'Epoch: {epoch}')
            print(f'\tTraining loss: {avg_loss:.6f}')
            print(f'\tValidation loss: {avg_vloss:.6f}')
            print(f'\tAccuracy: {correct_guesses}/{all_guesses} {100*correct_guesses/all_guesses:.3f}%')
            print(f'\tPiece type matched: {correct_piece_match}/{all_guesses} {100*correct_piece_match/all_guesses:.3f}%')
            for pc in range(1,7):
                print(f'\t{chess.piece_name(pc)} move accuracy: {piece_correct_guesses[pc]}/{piece_all_guesses[pc]} {100*piece_correct_guesses[pc]/piece_all_guesses[pc]:.3f}%')
            valid_secs = time.time() - valid_time_start
            print(f'\tTime: train {train_secs:.1f}s, valid {valid_secs:.1f}s, epoch {time.time()-epoch_time_start:.1f}s')
            print('-'*20)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"model_{timestamp}{'_extended_' if extended else '_'}{epoch}.model"
            save_training_checkpoint(epoch, model, optimizer, batch_size, prog_check, extended)
            torch.save(model.state_dict(), model_path)

def save_training_checkpoint(epoch, model, optimizer, batch_size, prog_check, extended):
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_size': batch_size,
        'progress_check': prog_check,
        'extended': extended
    }
    if extended:
        torch.save(checkpoint_data, "extended_training_checkpoint_epoch"+str(epoch)+".tar")
    else:
        torch.save(checkpoint_data, "training_checkpoint_epoch"+str(epoch)+".tar")

def load_training_checkpoint(epoch, model, optimizer, extended=False):
    prefix = "extended_training_checkpoint_epoch" if extended else "training_checkpoint_epoch"
    checkpoint_data = torch.load(prefix+str(epoch)+".tar")
    model.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    return checkpoint_data['epoch'], checkpoint_data['batch_size'], checkpoint_data['progress_check'], checkpoint_data['extended']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="../data/parsed/train.pgn")
    parser.add_argument("--valid", type=str, default="../data/parsed/valid.pgn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--prog_check", type=int, default=10000)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--resnet", action="store_true")
    parser.add_argument("--max_games", type=int, default=None)
    args = parser.parse_args()
    if args.extended:
        print("Extended model chosen.")
    if args.resnet:
        print("ResNet model chosen.")
    train(args.train, args.valid, args.epochs, args.batch_size, args.prog_check, args.extended, args.resnet, args.max_games)