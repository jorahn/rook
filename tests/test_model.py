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
