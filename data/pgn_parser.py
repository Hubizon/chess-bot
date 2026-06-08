import chess.pgn
import argparse
import os

# Filter for games with a high enough winner's rating that ended normally.
# Passing a negative sample_size (as default) causes the entire file to be read.
def filter_pgn(fin, fout, min_rating = 2300, progress_check = 1000, sample_size = -1, games_saved = 0):
    games_seen = 0

    while True:
        if games_saved >= sample_size > 0:
            break

        game = chess.pgn.read_game(fin)

        if game is None:
            break # EOF

        headers = game.headers
        result = headers.get("Result", None)
        if result is None:
            break # Result will likely never be missing

        if result == "1/2-1/2":
            continue # Discard drawn games

        games_seen += 1

        winner_rating = None
        if result == "1-0": # Gets the winner's ELO. Also will likely never be missing.
            winner_rating = headers.get("WhiteElo")
        else:
            winner_rating = headers.get("BlackElo")
        if winner_rating is None:
            continue

        event_type = headers.get("Event", "Bullet") # Excludes the bullet games 
        game_end_reason = headers.get("Termination", None) # Excludes the TimeForfeit games

        if game_end_reason == "Normal" and "Bullet" not in event_type:
            if int(winner_rating) >= min_rating:
                fout.write(str(game)+"\n\n")
                games_saved += 1

        if games_seen % progress_check == 0:
            print(f"Seen {games_seen} - saved {games_saved} ({100*games_saved/games_seen}%)")

    print(f"Games seen in this file: {games_seen}")
    print(f"Total games saved so far: {games_saved}")

    return games_saved

# Cuts a number of games from the front, spliting into validation and training sets.
def split_data(parsed_file, train_file, valid_file, valid_size = 500):
    fin = open(parsed_file, "r", encoding="utf-8")
    fout_train = open(train_file, "w", encoding="utf-8")
    fout_valid = open(valid_file, "w", encoding="utf-8")

    valid_counter = 0
    while True:
        game = chess.pgn.read_game(fin)
        if game is None:
            break # EOF

        valid_counter += 1
        if valid_counter <= valid_size:
            fout_valid.write(str(game)+"\n\n")
        else:
            fout_train.write(str(game)+"\n\n")
    
    print("Finished splitting the data.\n")

def view_folder(input_directory_path, output_file, min_rating, progress_check, sample_size):
    # Memory-safe alternative loop for massive multi-gigabyte files
    infiles = os.listdir(input_directory_path)

    games_saved = 0

    fout = open(output_file, "w", encoding="utf-8")

    for infile in infiles:
        if not infile.endswith(".pgn"):
            continue
        fin = open(input_directory_path+infile, "r", encoding="utf-8")
        games_saved = filter_pgn(fin, fout, min_rating, progress_check, sample_size, games_saved)
        print(f"Finished reading file {input_directory_path+infile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./raw/")
    parser.add_argument("--parsed", type=str, default="./parsed/parsed.pgn")
    parser.add_argument("--train", type=str, default="./parsed/train.pgn")
    parser.add_argument("--valid", type=str, default="./parsed/valid.pgn")
    parser.add_argument("--min_rating", type=int, default=2500)
    parser.add_argument("--progress_check", type=int, default=1000)
    parser.add_argument("--sample_size", type=int, default=-1) # Sample_size of -1 causes the entire file to be read
    parser.add_argument("--valid_size", type=int, default=5000)
    args = parser.parse_args()

    view_folder(args.input, args.parsed, args.min_rating, args.progress_check, args.sample_size)
    split_data(args.parsed, args.train, args.valid, args.valid_size)

    # Empirically with ELO 2200-2300 around 0.2-1% of games satisfy all conditions
    # With ELO 2000 this value increases to around 5%
    # In the expert-only database it was found that around 75% of the games are accepted (non-draws).