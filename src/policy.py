import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import pipeline
import torch
import chess

from src.utils.common import process_fen, unprocess_fen
from src.model import RookTokenizer

class BCChessPolicy():
    def __init__(self, model, tokenizer, batch_size=1, train_task="clf", filter_illegal=False):
        print(f"Loading ROOK Chess Policy (Behavior Cloning)")

        pipeline_type = "text-classification" if train_task == "clf" else "text-generation"
        self.pipeline_type = pipeline_type

        if isinstance(tokenizer, str) and os.path.exists(tokenizer):
            tokenizer = RookTokenizer.from_pretrained(tokenizer)
        
        if train_task == "clf":
            top_k = None if filter_illegal else 1
            max_new_tokens = None
        else:
            top_k = None
            max_new_tokens = 2 # increase for lm-cot
            
        self._pipeline = pipeline(
            pipeline_type, 
            model=model, 
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            batch_size=batch_size,
            top_k=top_k,
        )
        
        self._filter_illegal = filter_illegal
        print(f"Total parameters: {self._pipeline.model.num_parameters():,}")

    def _convert_fen(self, fen):
        fen = process_fen(fen) if "/" in fen else fen
        end_token = "[CLS]" if self.pipeline_type == "text-classification" else "[ACTION]"
        fen += end_token if not fen.endswith(end_token) else ""
        return fen

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
            if self.pipeline_type == "text-classification":
                predictions = self._pipeline(inputs)
                return [p[0]["label"] for p in predictions]
            else:
                predictions = self._pipeline(inputs, max_new_tokens=2)
                actions = []
                for p in predictions:
                    try:
                        actions.append(p[0]["generated_text"].split("[ACTION]")[-1])
                    except:
                        print("failed extracting [ACTION] from", p[0]["generated_text"])
                        actions.append("0000")
                return actions
        else:
            # TODO vectorize
            # TODO add pipeline_type text-generation
            if self.pipeline_type == "text-generation":
                raise NotImplementedError("text-generation pipeline is not yet implemented for filter_illegal")
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
