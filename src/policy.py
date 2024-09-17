import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import pipeline
import torch

from src.utils.common import process_fen


class BCChessPolicy():
    def __init__(self, model, tokenizer, batch_size=1):
        print(f"Loading ROOK Chess Policy (Behavior Cloning)")
        self._pipeline = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            batch_size=batch_size,
        )
        print(f"Total parameters: {self._pipeline.model.num_parameters():,}")

    def _convert_fen(self, fen):
        return process_fen(fen) + "[CLS]"

    def play(self, fen):
        if isinstance(fen, str):
            fen = [fen]
        inputs = [self._convert_fen(f) for f in fen]
        predictions = self._pipeline(inputs)
        return [p["label"] for p in predictions]
