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
print(extract_rook(example))
print(process_cot(extract_rook(example)))
print(tokenizer(process_cot(extract_rook(ds["train"][n]))["text"])["input_ids"])

print(tokenizer.decode(tokenizer(process_cot(extract_rook(ds["train"][n]))["text"])["input_ids"]))