from src.tokenizer.tokenizer import make_tokenizer

def test_make_tokenizer():
    tokenizer = make_tokenizer("src/tokenizer/rookworld_vocab.json")
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.vocab_size == 2002
    assert tokenizer.encode("1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33") == [5, 32, 10, 30, 9, 28, 5, 9, 30, 6, 17, 17, 30, 7, 30, 5, 6, 32, 7, 17, 5, 8, 14, 7, 8, 15, 17, 6, 19, 11, 0, 33, 0, 2, 0, 2, 0, 5, 0, 7, 7]
    
    # https://arxiv.org/pdf/2402.04494 paper seems to use the same tokenizer for BC, SV, AV predictors
    # otherwise model size would vary, although BC and SV dont need action space tokenization

    # pad vocab/embedding-size to next power of 2 --> 2048 for efficient mat-muls in model embedding layer
    # this can also fit plain and COT data representation 

    # => one tokenizer for all tasks
    