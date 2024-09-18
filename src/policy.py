import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import pipeline
import torch
import chess
import numpy as np

from src.utils.common import process_fen, unprocess_fen


class BCChessPolicy():
    def __init__(self, model, tokenizer, batch_size=1, filter_illegal=False):
        print(f"Loading ROOK Chess Policy (Behavior Cloning)")
        self._pipeline = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            batch_size=batch_size,
            top_k=None if filter_illegal else 1,
        )
        self._filter_illegal = filter_illegal
        print(f"Total parameters: {self._pipeline.model.num_parameters():,}")

    def _convert_fen(self, fen):
        if fen[-5:] == "[CLS]":
            return fen
        return process_fen(fen) + "[CLS]"
    
    def _normalize_fen(self, fen):
        try:
            board = chess.Board(fen)
            return fen
        except ValueError:
            if fen[-5:] == "[CLS]":
                fen = fen[:-5]
            return unprocess_fen(fen)

    def play(self, fen):
        if isinstance(fen, str):
            fen = [fen]
        inputs = [self._convert_fen(f) for f in fen]
        if not self._filter_illegal:
            predictions = self._pipeline(inputs)
            return [p[0]["label"] for p in predictions]
        else:
            # TODO vectorize
            predictions = self._pipeline(inputs)

            boards = [chess.Board(self._normalize_fen(f)) for f in fen]
            legal_moves = [[m.uci() for m in board.legal_moves] for board in boards]
            scores = [[p["score"] for p in pred] for pred in predictions]
            labels = [[p["label"] for p in pred] for pred in predictions]

            dropped_labels = []
            for i, label in enumerate(labels):
                for j, l in enumerate(label):
                    if l not in legal_moves[i]:
                        dropped_labels.append((i, j))
                labels[i] = [l for j, l in enumerate(label) if (i, j) not in dropped_labels]
                scores[i] = [s for j, s in enumerate(scores[i]) if (i, j) not in dropped_labels]
            best_moves = [max(zip(label, score), key=lambda x: x[1])[0] for label, score in zip(labels, scores)]

            return best_moves