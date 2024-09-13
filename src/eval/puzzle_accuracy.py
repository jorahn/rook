# Lichess Puzzle Database data format:
#PuzzleId                                                       00008
#FEN                r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - ...
#Moves                                  f2g3 e6e7 b2b1 b3c1 b1c1 h6c1
#Rating                                                          1853
#RatingDeviation                                                   76
#Popularity                                                        94
#NbPlays                                                         6405
#Themes                         crushing hangingPiece long middlegame
#GameUrl                        https://lichess.org/787zsVup/black#48
#OpeningTags                                                      NaN

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import chess
from tqdm import tqdm
import pandas as pd
import requests

from src.data.common import process_fen


URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
DEBUG = False

def make_move(pipe, fen):
    prompt = process_fen(fen) + "[CLS]"
    print(prompt)
    generation = pipe(prompt)
    try:
        move = generation[0]["generated_text"].split("B: ")[-1].strip()
    except IndexError:
        move = "0000"
    return move

def iterate_puzzles(num_puzzles):
    if not os.path.exists("src/eval/lichess_db_puzzle.csv.zst"):
        print("Downloading Lichess Puzzle Database")
        with open("src/eval/lichess_db_puzzle.csv.zst", "wb") as f:
            f.write(requests.get(URL).content)
    else:
        print("Lichess Puzzle Database already exists, skipping download")
    data = pd.read_csv("src/eval/lichess_db_puzzle.csv.zst")
    print(f"Sampling {num_puzzles} puzzles")
    data = data.sample(num_puzzles, random_state=42)

    print("Lichess Puzzle Rating Distribution:")
    print(pd.cut(data["Rating"], bins=12).value_counts().sort_index())

    for _, row in tqdm(data.iterrows(), total=len(data)):
        targets = row["Moves"].split()
        yield row["FEN"], targets

def evaluate_puzzle(pipe, fen, targets):
    stats = {
        "solved_puzzles": 0, 
        "total_moves": 0,
        "correct_moves": 0, 
    }
    for target in targets:
        stats["total_moves"] += len(targets)

        board = chess.Board(fen)
        for i, target in enumerate(targets):
            move = make_move(pipe, board.fen())
            if DEBUG: print(f"Move: {move}, Target: {target}, FEN: {board.fen()}, i: {i}")
            if move == target:
                stats["correct_moves"] += 1
                if i == len(targets) - 1:
                    stats["solved_puzzles"] += 1
                board.push_uci(target)
            if move != target:
                break

    return stats

