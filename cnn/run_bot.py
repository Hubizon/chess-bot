import torch
import chess
import numpy as np
import sys
import argparse
from cnn import ChessCNN, ExtendedChessCNN, ChessResNet 

def board_to_datapoint(board):
    datapoint = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    
    for square, piece in piece_map.items():
        file = chess.square_file(square) # a-h -> 0-7
        rank = chess.square_rank(square) # 1-8 -> 0-7
        piece_type = piece.piece_type - 1 # 1-6 -> 0-5/6-11
        datapoint[piece_type if (piece.color == chess.WHITE) else (piece_type+6), file, rank] = 1.0

    return datapoint

def get_best_move(model, board, extended):
    if board.turn == chess.WHITE:
        board_view = board
    else:
        board_view = board.mirror()
    board_datapoint = board_to_datapoint(board_view)

    best_move = None
    best_value = float("-inf")

    with torch.no_grad():
        if extended:
            prediction_move, prediction_prom = model(torch.as_tensor(board_datapoint).view(1, 12, 8, 8))
            prediction_move = prediction_move.view(64**2)
            prediction_prom = prediction_prom.view(5)
        else:
            prediction_from, prediction_to = model(torch.as_tensor(board_datapoint).view(1, 12, 8, 8))
            prediction_from = prediction_from.view(64)
            prediction_to = prediction_to.view(64)

    legal_moves = board.legal_moves
    for move in legal_moves:
        if board.turn == chess.WHITE:
            if extended:
                value = prediction_move[move.from_square*64+move.to_square]
            else:
                value = prediction_from[move.from_square]+prediction_to[move.to_square]
        else:
            mirrored_from = chess.square_mirror(move.from_square)
            mirrored_to = chess.square_mirror(move.to_square)
            if extended:
                value = prediction_move[mirrored_from*64+mirrored_to]
            else:
                value = prediction_from[mirrored_from]+prediction_to[mirrored_to]
        if value > best_value:
            best_value = value
            best_move = move

    if best_move.promotion is not None:
        if extended:
            best_move.promotion = torch.argmax(prediction_prom[1:])+1 # Chooses the best promotion piece
        else:
            best_move.promotion = chess.QUEEN # No underpromotion is recognized in this model

    return best_move

# Function for running the chess bot
# Compliant with the UCI protocol
def run_bot(model_path, extended, resnet):
    board = chess.Board()

    if resnet:
        model = ChessResNet()
    elif extended:
        model = ExtendedChessCNN()
    else:
        model = ChessCNN()

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval() 

    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue
        
        args = line.split()
        uci_command = args[0]

        if uci_command == "uci":
            print("uciok")
            sys.stdout.flush()

        if uci_command == "isready":
            print("readyok")
            sys.stdout.flush()

        if uci_command == "position":
            if "fen" in args:
                fen_id = args.index("fen")+1
                fen = " ".join(args[fen_id : fen_id+6])
                board = chess.Board(fen)
            else:
                board = chess.Board()

            if "moves" in args:
                moves_id = args.index("moves")+1
                for move in args[moves_id:]:
                    try:
                        board.push_uci(move)
                    except ValueError:
                        continue

        if uci_command == "go":
            print(f"bestmove {get_best_move(model, board, extended)}")
            sys.stdout.flush()
        if uci_command == "quit":
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--extended", type=bool, default=False)
    parser.add_argument("--resnet", type=bool, default=False)
    args = parser.parse_args()
    run_bot(args.model, args.extended, args.resnet)