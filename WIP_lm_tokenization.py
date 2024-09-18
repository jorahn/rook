from datasets import load_dataset
from src.utils.common import process_cot
from src.utils.convert_rook import extract_rook
from src.model import make_tokenizer
ds = load_dataset("text", data_files="/home/jrahn/dev/RookWorld/dev/data/rook/rook_train_260k.txt")
tokenizer = make_tokenizer(task="lm")
print(tokenizer.vocab_size)

n = 0
example = ds["train"][n]
print(example)
step = extract_rook(example)
print(step)
step = process_cot(step)
print(step)
step = tokenizer(step["text"])["input_ids"]
print(step)
step = tokenizer.decode(step)
print(step)

for ex in ds["train"]:
    if len(ex["text"]) > 10:
        ids = tokenizer(process_cot(extract_rook(ex))["text"])["input_ids"]
        if len(ids) != 116:
            print(ex)
            print(extract_rook(ex))
            print(process_cot(extract_rook(ex))["text"])
            print(ids)
    
