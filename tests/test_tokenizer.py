from src.tokenizer.tokenizer import make_tokenizer

def test_make_tokenizer():
    tokenizer = make_tokenizer("src/tokenizer/rookworld_vocab.json")
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.vocab_size == 2003
    assert tokenizer.encode("1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33") == [6, 33, 11, 31, 10, 29, 6, 10, 31, 7, 18, 18, 31, 8, 31, 6, 7, 33, 8, 18, 6, 9, 15, 8, 9, 16, 18, 7, 20, 12, 0, 34, 0, 2, 0, 2, 0, 6, 0, 8, 8]
    
    # different tokenizers for BC, SV, AV predictors
    # different tokenizers for plain and COT data representation
    # test padded vocab (to next power of 2)
    