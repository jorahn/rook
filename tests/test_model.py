from src.model import make_model, make_tokenizer

def test_make_model():
    model = make_model({
        "vocab_size": 2048,
        "pad_token_id": 0,
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "max_position_embeddings": 78,
    })
    assert model.config.vocab_size == 2048
    assert model.config.num_labels == 1968 # data/action_space.json
    assert model.num_parameters() == 9_421_056

def test_make_tokenizer():
    tokenizer = make_tokenizer()
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.vocab_size == 32
    assert tokenizer.encode("1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33") == [3, 30, 8, 28, 7, 26, 3, 7, 28, 4, 15, 15, 28, 5, 28, 3, 4, 30, 5, 15, 3, 6, 12, 5, 6, 13, 15, 4, 17, 9, 31, 0, 0, 3, 5, 5]
    
    # https://arxiv.org/pdf/2402.04494 paper seems to use the same tokenizer for BC, SV, AV predictors
    # otherwise model size would vary, although BC and SV dont need action space tokenization

    # pad vocab/embedding-size to next power of 2 --> 2048 for efficient mat-muls in model embedding layer
    # this can also fit plain and COT data representation 

    # => one tokenizer for all tasks