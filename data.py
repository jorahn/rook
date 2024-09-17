# Load/prepare dataset for training and evaluation

from argparse import ArgumentParser

from datasets import load_dataset

from src.utils.common import process_fen


parser = ArgumentParser()
parser.add_argument("dataset", type=str, help="Path to dataset")
parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
parser.add_argument("--push_to_hub", type=str, help="Push dataset to Hugging Face Hub")
args = parser.parse_args()


def load_csv_dataset(fen_column="FEN", move_column="Move"):
    data = load_dataset("csv", data_files=args.dataset.split(","))
    data = data.map(
        lambda x: {"text": process_fen(x[fen_column])+"[CLS]", "label": x[move_column]},
        remove_columns=[fen_column, move_column],
    )
    return data

def load_hf_dataset(fen_column="FEN", move_column="Move"):
    data = load_dataset(args.dataset, split=args.split)
    data = data.map(
        lambda x: {"text": process_fen(x[fen_column])+"[CLS]", "label": x[move_column]},
        remove_columns=[fen_column, move_column],
    )
    return data

if ".csv" in args.dataset:
    data = load_csv_dataset()
else:
    data = load_hf_dataset()

print(data)

if args.push_to_hub:
    data.push_to_hub(args.push_to_hub)
else:
    data.save_to_disk(f"data/dataset_{args.dataset.replace('/', '__').replace('.', '_')}")