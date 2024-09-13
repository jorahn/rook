# WORK IN PROGRESS

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE

import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--input", help="Input file")
parser.add_argument("-o", "--output", help="Output file")
args = parser.parse_args()

SPECIAL_TOKENS = {
    "POLICY_TASK": "[POLICY]",
    "ENVIRONMENT_TASK": "[ENV]",
}


def make_tokenizer(vocab_path):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    tokenizer = Tokenizer(BPE(
        vocab=dict(zip(vocab, range(len(vocab)))), 
        merges=[])
    )
    tokenizer.add_special_tokens(list(SPECIAL_TOKENS.values()))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        clean_up_tokenization_spaces=False
    )
    return fast_tokenizer

if __name__ == "__main__":
    tokenizer = make_tokenizer(args.input)
    tokenizer.save_pretrained(args.output)
    print(f"Tokenizer saved to {args.output}")