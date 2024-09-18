import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser

import wandb
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import numpy as np

from src.model import make_model, make_tokenizer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}

def run_training(args):
    tokenizer = make_tokenizer()
    def encode(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    model = make_model({
        "pad_token_id": tokenizer.pad_token_id,
        "hidden_size": 256,             # embedding dimension from the paper
        "intermediate_size": 1024,      # not specified
        "num_hidden_layers": 8,         # as in the paper
        "num_attention_heads": 8,       # as in the paper
        "max_position_embeddings": 78,  # as in the paper (for bc and sv predictors, +1 for av)
        "torch_dtype": torch.bfloat16,
        #"attn_implementation": "flash_attention_2",
        #device_map="auto",
        "device": "cuda",
        "finetuning_task": "text-classification",
    })
    
    if os.path.exists(args.dataset):
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(args.dataset)
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})
    if args.val:
        if os.path.exists(args.val):
            val_dataset = load_from_disk(args.val)
        else:
            val_dataset = load_dataset(args.val)
        
        if isinstance(val_dataset, DatasetDict):
            split = "test" if "test" in val_dataset else "train"
            dataset["test"] = val_dataset[split]
        else:
            dataset["test"] = val_dataset
    
    if args.max_samples:
        dataset["train"] = dataset["train"].select(range(args.max_samples))
    
    dataset["train"] = dataset["train"].class_encode_column("label")
    dataset["train"] = dataset["train"].align_labels_with_mapping(
            label2id=model.config.label2id, label_column="label")
    class_label_feature = dataset["train"].features["label"]
    dataset["test"] = dataset["test"].cast_column("label", class_label_feature)

    dataset = dataset.map(encode, batched=True)
    print(dataset)

    training_args = TrainingArguments(
        # 2 devices
        per_device_train_batch_size=1024, # bs 1024 in the paper on 4x 95G tpu, try to fit as much as possible ...
        gradient_accumulation_steps=1,    # ... else increase this
        gradient_checkpointing=False,     # save memory if needed, reduces speed
        bf16=True,
        learning_rate=4e-4,               # as in the paper
        torch_compile=True,
        output_dir="checkpoints/save_"+args.run,
        per_device_eval_batch_size=512,
        eval_strategy="steps",
        eval_steps=500,
        eval_on_start=True,
        num_train_epochs=5.0,             # 2.7-3.2 in the paper for ablations, 5.4 for full training
        #max_steps=args.max_steps,        # 5e6 in the paper, 40m samples, bs 1024 -> 128 Epochs !?!
        lr_scheduler_type="cosine",
        warmup_steps=500,
        save_strategy="epoch",
        log_level="error",
        report_to="wandb" if args.run else "none",
        run_name=args.run,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        optimizers=(None, None), # TODO: non-standard optimizer and scheduler
        )
    
    print(f"training {model.num_parameters():,} parameters")
    trainer.train()
    trainer.save_model("checkpoints/save_"+args.run)
    tokenizer.save_pretrained("checkpoints/save_"+args.run)


if __name__ == "__main__":
    parser = ArgumentParser("Run training")
    parser.add_argument("dataset", help="Local or remote HF Dataset name")
    parser.add_argument("-max_samples", default=40_000_000, help="Max Samples")
    parser.add_argument("-val", help="Local or remote HF Dataset name for validation")
    parser.add_argument("-max_steps", help="Max Steps")
    parser.add_argument("-run", default="rook_policy_bc", help="W&B run name, None for no logging")
    args = parser.parse_args()

    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"]="ROOK"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"

    run_training(args)

    wandb.finish()
