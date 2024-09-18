# Evaluate checkpoint on actions, puzzles, checkmate-in-one

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from argparse import ArgumentParser
import json
import io

from datasets import load_dataset, load_from_disk, DatasetDict
import evaluate
from evaluate import evaluator
import torch
import chess
import pandas as pd
from tqdm import tqdm

from src.policy import BCChessPolicy
from src.model import RookTokenizer


parser = ArgumentParser()
parser.add_argument("checkpoint", type=str, help="Path to model & tokenizer checkpoint")
parser.add_argument("--filter_illegal", action="store_true", help="Filter illegal moves during evaluation")
parser.add_argument("--eval_type", type=str, choices=["action", "puzzle", "checkmate"], default="action", help="Type of evaluation to perform")
parser.add_argument("--eval_dataset", type=str, help="Path to evaluation dataset")
parser.add_argument("--eval_split", type=str, default="test", help="Dataset split to evaluate")
parser.add_argument("-n", "--num_samples", type=int, default=1_000, help="Number of samples to evaluate")
parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size for evaluation")
args = parser.parse_args()

if args.eval_type == "action":
    # expect preprocessed (`process_fen()`) dataset with "text" and "label" columns
    if os.path.exists(args.eval_dataset):
        data = load_from_disk(args.eval_dataset)
        if isinstance(data, DatasetDict):
            data = data["train"]
    else:
        data = load_dataset(args.eval_dataset, split=args.eval_split)
    if args.num_samples > 0:
        data = data.select(range(args.num_samples))

    tokenizer = RookTokenizer.from_pretrained(args.checkpoint)
    policy = BCChessPolicy(args.checkpoint, tokenizer, batch_size=args.batch_size, filter_illegal=args.filter_illegal)
    eval_results = {"total": 0, "correct": 0}
    for example in tqdm(data):
        prediction = policy.play(example["text"])
        if prediction[0] == example["label"]:
            eval_results["correct"] += 1
        eval_results["total"] += 1
    print("-"*50)
    print("Action Evaluation results:")
    print(f"Accuracy: {eval_results['correct'] / eval_results['total']:.2%}")
    print(eval_results)

elif args.eval_type == "puzzle":
    # expect csv file with columns "FEN" and "Moves" (lichess puzzle format)
    data = pd.read_csv(args.eval_dataset)
    tokenizer = RookTokenizer.from_pretrained(args.checkpoint)
    policy = BCChessPolicy(args.checkpoint, tokenizer, batch_size=args.batch_size)

    eval_results = {
        "correct_moves": 0,
        "total_moves": 0,
        "correct_puzzles": 0,
        "total_puzzles": 0,
    }
    for _, row in tqdm(data.iterrows()):
        eval_results["total_puzzles"] += 1
        fen = row["FEN"]
        board = chess.Board(fen)
        moves = row["Moves"].split()
        for idx, move in enumerate(moves):
            if idx % 2 == 0:
                eval_results["total_moves"] += 1
                prediction = policy.play(board.fen())
                if prediction["label"] == move:
                    eval_results["correct_moves"] += 1
                    if idx == len(moves) - 1:
                        eval_results["correct_puzzles"] += 1
                else:
                    break
            board.push(chess.Move.from_uci(move))
    print("-"*50)
    print("Puzzle Evaluation results:")
    print(f"Accuracy: {eval_results['correct_puzzles']/eval_results['correct_puzzles']:.2%}")
    print(eval_results)

elif args.eval_type == "checkmate":
    with open("../data/checkmate.json", "r") as f:
        data = json.load(f)
    data = data["examples"]
    if args.num_samples:
        data = data[:args.num_samples]

    tokenizer = RookTokenizer.from_pretrained(args.checkpoint)
    policy = BCChessPolicy(args.checkpoint, tokenizer, batch_size=args.batch_size)
    
    eval_results = {
        "correct_moves": 0,
        "total_moves": 0,
    }
    for example in tqdm(data):
        eval_results["total_moves"] += 1
        game = chess.pgn.read_game(io.StringIO(example["input"]))
        board = game.board()
        for move in game.mainline_moves(): board.push(move)
        fens = board.fen()
        move = board.parse_san(example["target"])
        predicted_move = policy.play(board.fen())
        if predicted_move["label"] == move:
            eval_results["correct_moves"] += 1

    print("-"*50)
    print("Checkmate Evaluation results:")
    print(f"Accuracy: {eval_results['correct_moves']/eval_results['total_moves']:.2%}")
    print(eval_results)

