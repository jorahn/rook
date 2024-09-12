import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import LlamaForSequenceClassification, LlamaTokenizerFast, LlamaConfig
from transformers import trainer, TrainingArguments

config = LlamaConfig(
    vocab_size=2048,
    hidden_size=512,
    intermediate_size=512,
    pad_token_id=2047,
    num_hidden_layers=8,
    num_attention_heads=8,
    max_position_embeddings=256,
)
model = LlamaForSequenceClassification(config=config)
#tokenizer = LlamaTokenizerFast(tokenizer_file="tokenizer.json")

print(model)
print(f"{model.num_parameters():,}")


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
