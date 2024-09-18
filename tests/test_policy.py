import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.policy import BCChessPolicy
from src.model import make_model, make_tokenizer
from src.const import START_POSITION, ACTION_SPACE

def test_policy():
    cfg = {
        "vocab_size": 2048,
        "pad_token_id": 0,
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "max_position_embeddings": 78,
        "finetuning_task": "text-classification",
    }
    model = make_model(cfg)
    tokenizer = make_tokenizer()
    p = BCChessPolicy(model=model, tokenizer=tokenizer)
    moves = p.play(START_POSITION)
    assert len(moves) == 1
    assert moves[0] in ACTION_SPACE

def test_policy_bs2():
    cfg = {
        "vocab_size": 2048,
        "pad_token_id": 0,
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "max_position_embeddings": 78,
        "finetuning_task": "text-classification",
    }
    model = make_model(cfg)
    tokenizer = make_tokenizer()
    p = BCChessPolicy(model=model, tokenizer=tokenizer, batch_size=2)
    moves = p.play([START_POSITION]*2)
    assert len(moves) == 2
    for move in moves:
        assert move in ACTION_SPACE
