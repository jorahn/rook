import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
with open("dev/data/rookworld_vocab.json", "r") as f:
    vocab = json.load(f)

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(vocab=dict(zip(vocab, range(len(vocab)))), merges=[]))

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)
print(fast_tokenizer)
print(fast_tokenizer.encode("rrbbkK"))