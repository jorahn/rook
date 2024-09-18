# Load/prepare dataset for training and evaluation

from argparse import ArgumentParser

from datasets import load_dataset

from src.utils.common import process_fen, process_cot
from src.utils.convert_rook import extract_rook


parser = ArgumentParser()
parser.add_argument("dataset", type=str, help="Path to dataset")
parser.add_argument("--task", type=str, default="clf", help="Task type: clf or lm or lm-cot")
parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
parser.add_argument("--fen_column", type=str, default="FEN", help="Column name for FEN")
parser.add_argument("--move_column", type=str, default="Move", help="Column name for Move")
parser.add_argument("--options_column", type=str, help="Column name for Options")
parser.add_argument("--values_column", type=str, help="Column name for Values")
parser.add_argument("--rook", action="store_true", help="Convert from ROOK format")
parser.add_argument("--push_to_hub", type=str, help="Push dataset to Hugging Face Hub")
args = parser.parse_args()


def process_clf(data, fen_column, move_column):
    data = data.map(
        lambda x: {"text": process_fen(x[fen_column])+"[CLS]", "label": x[move_column]},
        remove_columns=[fen_column, move_column],
    )
    return data

def process_lm(data, fen_column, move_column, options_column=None, values_column=None, cot=False):
    if cot:
        # scale values from (-999.99, 999.99) independent of player -> to (0, 100) from the perspective of the active player
        data = data.map(
            process_cot, 
            fn_kwargs={
                "fen_column": fen_column, "options_column": options_column, 
                "values_column": values_column, "move_column": move_column
            },
            remove_columns=[fen_column, move_column, options_column, values_column],
        )
    else:
        data = data.map(
            lambda x: {"text": process_fen(x[fen_column])+"[ACTION]"+x[move_column]},
            remove_columns=[fen_column, move_column],
        )
    return data

## Load Dataset
print("Loading Dataset ...")
if ".csv" in args.dataset:
    data = load_dataset("csv", data_files=args.dataset.split(","))
elif ".txt" in args.dataset:
    data = load_dataset("text", data_files=args.dataset)
else:
    data = load_dataset(args.dataset, split=args.split)

## Process Dataset
print("Processing Dataset ...")
if args.rook:
    print("Converting from ROOK format")
    data = data.filter(lambda x: isinstance(x["text"], str) and len(x["text"]) > 10)
    data = data.map(extract_rook)
    args.fen_column = "fen"
    args.move_column = "action"
    args.options_column = "options"
    args.values_column = "values"

if args.task == "clf":
    print("Processing for text-classification task")
    data = process_clf(
        data, 
        fen_column=args.fen_column, 
        move_column=args.move_column
    )
elif args.task == "lm":
    print("Processing for language-modeling task")
    data = process_lm(
        data, 
        fen_column=args.fen_column, 
        move_column=args.move_column, 
        cot=False
    )
elif args.task == "lm-cot":
    print("Processing for language-modeling task with COT")
    data = process_lm(
        data, 
        fen_column=args.fen_column, 
        move_column=args.move_column,
        options_column=args.options_column,
        values_column=args.values_column,
        cot=True
    )

print()
print("Processed Dataset overview:")
print(data)

## Save Dataset
print("Saving Dataset ...")
if args.push_to_hub:
    data.push_to_hub(args.push_to_hub)
else:
    data.save_to_disk(f"data/dataset_{args.task}_{args.dataset.replace('/', '__').replace('.', '_')}")