import chess.pgn

# Filter for games with a high enough winner's rating that ended normally.
# Passing a negative sample_size (as default) causes the entire file to be read.
def filter_pgn(input_file, output_file, min_rating = 2300, progress_check = 1000, sample_size = -1):
    games_seen = 0
    games_saved = 0

    fin = open(input_file, "r", encoding="utf-8")
    fout = open(output_file, "w", encoding="utf-8")

    while True:
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

        if games_saved >= sample_size > 0:
            break

    print("PGN reading completed.")
    print(f"Total games seen: {games_seen}")
    print(f"Total games saved: {games_saved}")

# Cuts a number of games from the front, spliting into validation and training sets.
def split_data(output_file, train_file, valid_file, valid_size = 500):
    fin = open(output_file, "r", encoding="utf-8")
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

if __name__ == "__main__":
    INPUT_FILE = "lichess_elite_2025-11.pgn" # Input the name of the PGN file from the database
    PARSED_FILE = "filtered_games.pgn"
    TRAIN_FILE = "train_games.pgn"
    VALID_FILE = "valid_games.pgn"
    MIN_RATING = 2000 # The minimum rating of the winner

    # Empirically with ELO 2200-2300 around 0.2-1% of games satisfy all conditions
    # With ELO 2000 this value increases to around 5%
    # In the expert-only database it was found that around 75% of the games are accepted (non-draws).

    # remove sample_size to parse through the entire file
    filter_pgn(INPUT_FILE, PARSED_FILE, MIN_RATING, progress_check=1000, sample_size=21000)
    split_data(PARSED_FILE, TRAIN_FILE, VALID_FILE, valid_size=1000)