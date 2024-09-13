from src.model.model import make_config, make_model

def test_make_config():
    config = make_config(num_labels=2)
    assert config.num_labels == 2

def test_make_model():
    config = make_config(
        vocab_size=2048,
        hidden_size=512,
        intermediate_size=512,
        pad_token_id=2047,
        num_hidden_layers=8,
        num_attention_heads=8,
        max_position_embeddings=256,
    )
    model = make_model(config)
    assert model.config.vocab_size == 2048
    assert model.num_parameters() == 15_738_368

    # try to match the number of 9M parameters from https://arxiv.org/pdf/2402.04494 (ablation from Figure A6)
    config = make_config(
        vocab_size=2048,             # 32 should suffice for FEN encoding + 1968 actions for av predictor
                                     # padding to the next power of 2 -> 2048
        pad_token_id=0,
        hidden_size=256,             # embedding dimension from the paper
        intermediate_size=1024,      # not specified
        num_hidden_layers=8,         # as in the paper
        num_attention_heads=8,       # as in the paper
        max_position_embeddings=78,  # as in the paper (for bc and sv predictors, +1 for av)
    )
    model = make_model(config)
    assert model.config.vocab_size == 2048
    assert model.num_parameters() == 8_917_760
