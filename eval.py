import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from argparse import ArgumentParser

from transformers import pipeline
import torch
import chess

from src.eval import puzzle_accuracy


parser = ArgumentParser()
parser.add_argument("-m", "--model", type=str)
parser.add_argument("-e", "--eval_type", type=str, choices=["puzzle", "checkmate"], default="puzzle")
parser.add_argument("-n", "--num_samples", type=int, default=10_000)
args = parser.parse_args()

pipe = pipeline(
    "text-classification", 
    model=args.model,
    tokenizer=args.model,
    torch_dtype=torch.bfloat16,
    device=0 if torch.cuda.is_available() else -1,
)

if args.eval_type == "puzzle":
    stats = {}
    for position, targets in puzzle_accuracy.iterate_puzzles(args.num_samples):
        result = puzzle_accuracy.evaluate_puzzle(pipe, position, targets)
        stats.update(result)


print("-" * 50)
print("Evaluation Complete")
print(f"Total Puzzles: {args.num_samples}")
print(f"Total Moves: {stats['total_moves']}")
print(f"Correct Moves: {stats['correct_moves']} ({stats['correct_moves'] / stats['total_moves']:.2%})")
print(f"Solved Puzzles: {stats['solved_puzzles']}")
print(f"Puzzle Accuracy: {stats['solved_puzzles'] / args.num_samplese:.2%}")