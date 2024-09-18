from src.model import make_model, make_tokenizer

def test_make_model():
    model = make_model({
        "pad_token_id": 0,
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "max_position_embeddings": 78,
        "finetuning_task": "text-classification",
    })
    assert model.config.vocab_size == 128
    assert model.config.num_labels == 1968 # data/action_space.json
    assert model.num_parameters() == 8_929_536

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


def test_make_tokenizer_lm():
    tokenizer = make_tokenizer(task="lm")
    assert tokenizer.encode("..rqkbQ..b...p..p..p...pP.pPp.p..p..P...P....N....P..PPPR.BNKB.RwKQ..c60..13.[OPTIONS]c1d2 f1e2 c1b2 f3d2 f1c4[VALUES]-80.78 -81.09 -81.12 -81.18 -81.15[ACTION]f3d2") == [1, 1, 30, 29, 26, 19, 16, 1, 1, 19, 1, 1, 1, 28, 1, 1, 28, 1, 1, 28, 1, 1, 1, 28, 15, 1, 28, 15, 28, 1, 28, 1, 1, 28, 1, 1, 15, 1, 1, 1, 15, 1, 1, 1, 1, 14, 1, 1, 1, 1, 15, 1, 1, 15, 15, 15, 17, 1, 12, 14, 13, 12, 1, 17, 31, 13, 16, 1, 1, 20, 8, 2, 1, 1, 3, 5, 1, 2000, 498, 1294, 488, 1357, 1289, 2001, 0, 10, 2, 1, 9, 10, 0, 10, 3, 1, 2, 11, 0, 10, 3, 1, 3, 4, 0, 10, 3, 1, 3, 10, 0, 10, 3, 1, 3, 7, 2002, 1357]
