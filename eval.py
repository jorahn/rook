# Evaluate checkpoint on actions, puzzles, checkmate-in-one

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from argparse import ArgumentParser
import json
import io

from datasets import load_dataset, load_from_disk, DatasetDict
import chess
import chess.pgn
import pandas as pd
from tqdm import tqdm

from src.policy import BCChessPolicy
from src.model import RookTokenizer


parser = ArgumentParser()
parser.add_argument("checkpoint", type=str, help="Path to model & tokenizer checkpoint")
parser.add_argument("--task", type=str, default="clf", help="HF Decoder Transformer trained on clf|lm|lm-cot task")
parser.add_argument("--filter_illegal", action="store_true", help="Filter illegal moves during evaluation")
parser.add_argument("--eval_type", type=str, choices=["action", "puzzle", "checkmate"], default="action", help="Type of evaluation to perform")
parser.add_argument("--eval_dataset", type=str, help="Path to evaluation dataset (not required for checkmate eval)")
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

    policy = BCChessPolicy(args.checkpoint, args.checkpoint, train_task=args.task, batch_size=args.batch_size, filter_illegal=args.filter_illegal)
    eval_results = {"total": 0, "correct": 0}
    for example in tqdm(data):
        if args.task == "clf":
            prediction = policy.play(example["text"])
            if prediction[0] == example["label"]:
                eval_results["correct"] += 1
        else:
            fen, move = example["text"].split("[ACTION]")
            prediction = policy.play(fen)
            print(fen, move, prediction)
            input()
            if prediction[0] == move:
                eval_results["correct"] += 1
        eval_results["total"] += 1
    print("-"*50)
    print("Action Evaluation results:")
    print(f"Accuracy: {eval_results['correct'] / eval_results['total']:.2%}")
    print(eval_results)

elif args.eval_type == "puzzle":
    # expect csv file with columns "FEN" and "Moves" (lichess puzzle format)
    data = pd.read_csv(args.eval_dataset)
    if args.num_samples > 0:
        data = data[:args.num_samples]
    policy = BCChessPolicy(args.checkpoint, args.checkpoint, batch_size=args.batch_size, train_task=args.task)

    eval_results = {
        "correct_moves": 0,
        "total_moves": 0,
        "correct_puzzles": 0,
        "total_puzzles": 0,
    }
    pbar = tqdm(total=len(data), desc="Puzzle Evaluation")
    for _, row in data.iterrows():
        eval_results["total_puzzles"] += 1
        pbar.update(1)
        fen = row["FEN"]
        board = chess.Board(fen)
        moves = row["Moves"].split()
        for i, move in enumerate(moves):
            if i % 2 == 1:
                eval_results["total_moves"] += 1
                prediction = policy.play(board.fen())
                if prediction[0] == move:
                    eval_results["correct_moves"] += 1
                    if i == len(moves) - 1:
                        eval_results["correct_puzzles"] += 1
                else:
                    # all checkmates are correct solutions
                    # https://github.com/google-deepmind/searchless_chess/blob/9bcd9918bdbcd25b1b3b77401fd630cd9ce874b0/src/puzzles.py#L91
                    board.push(chess.Move.from_uci(prediction[0]))
                    if board.is_checkmate():
                        eval_results["correct_puzzles"] += 1
                    break
                
            board.push(chess.Move.from_uci(move))

    pbar.close()

    print("-"*50)
    print("Puzzle Evaluation results:")
    print(f"Accuracy: {eval_results['correct_puzzles']/eval_results['total_puzzles']:.2%}")
    print(eval_results)

elif args.eval_type == "checkmate":
    with open("data/checkmate.json", "r") as f:
        data = json.load(f)
    data = data["examples"]
    if args.num_samples > 0:
        data = data[:args.num_samples]

    policy = BCChessPolicy(args.checkpoint, args.checkpoint, batch_size=args.batch_size, train_task=args.task)
    
    eval_results = {
        "correct_moves": 0,
        "total_moves": 0,
    }
    for example in tqdm(data):
        eval_results["total_moves"] += 1
        game = chess.pgn.read_game(io.StringIO(example["input"]))
        board = game.board()
        for m in game.mainline_moves(): board.push(m)
        move = board.parse_san(example["target"])
        prediction = policy.play(board.fen())
        if prediction[0] == move.uci():
            eval_results["correct_moves"] += 1

    print("-"*50)
    print("Checkmate Evaluation results:")
    print(f"Accuracy: {eval_results['correct_moves']/eval_results['total_moves']:.2%}")
    print(eval_results)

