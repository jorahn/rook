import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser

import wandb
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_dataset
import numpy as np

from src.tokenizer.tokenizer import make_tokenizer
from src.model.model import make_model, make_config

parser = ArgumentParser("Run training")
parser.add_argument("-ds", default="jrahn/rw_policy_bc_709k", help="HF Dataset name")
parser.add_argument("-tk", default="src/tokenizer/rookworld_vocab.json", help="Tokenizer vocab file")
args = parser.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}


def run_training(args):
    tokenizer = make_tokenizer(args.tk)
    tokenizer.model_max_length = 78
    def encode(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    dataset = load_dataset(args.ds)
    dataset = dataset.map(encode, batched=True)
    label_list = list(set(dataset["train"].unique("label") + dataset["test"].unique("label")))
    label_list.sort()
    num_labels = len(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}

    config = make_config(
        vocab_size=2048,             # 32 should suffice for FEN encoding + 1968 actions for av predictor
                                    # padding to the next power of 2 -> 2048
        pad_token_id=0,
        hidden_size=256,             # embedding dimension from the paper
        intermediate_size=1024,      # not specified
        num_hidden_layers=8,         # as in the paper
        num_attention_heads=8,       # as in the paper
        max_position_embeddings=78,  # as in the paper (for bc and sv predictors, +1 for av)
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2",
        #device_map="auto",
        device="cuda",
        num_labels=num_labels,
        finetuning_task="text-classification",
    )

    model = make_model(config)
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in label_to_id.items()}

    training_args = TrainingArguments(
        # 2 devices
        per_device_train_batch_size=1024,  # bs 1024 in the paper on 4x 95G tpu, try to fit as much as possible ...
        gradient_accumulation_steps=1,    # ... else increase this
        gradient_checkpointing=False,     # save memory if needed, reduces speed
        bf16=True,
        learning_rate=4e-4,               # as in the paper
        #optim="adamw_torch_fused",
        torch_compile=True,
        output_dir="tmp",
        per_device_eval_batch_size=256,
        eval_strategy="steps",
        eval_steps=250,
        num_train_epochs=3.0,            # 2.7-3.2 in the paper for ablations, 5.4 for full training
        #max_steps=5e6,                    # 5e6 in the paper, 40m samples, bs 1024 -> 128 Epochs !?!
        lr_scheduler_type="cosine",
        warmup_steps=500,
        save_strategy="epoch",
        log_level="error",
        #report_to="none",
        report_to="wandb",
        run_name=args.ds.split("/")[-1],
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

    trainer.train()
    trainer.save_model("tmp")
    tokenizer.save_pretrained("tmp")

if __name__ == "__main__":
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"]="RookWorld"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"

    run_training(args)

    wandb.finish()