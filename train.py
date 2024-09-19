import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser

import wandb
from transformers import Trainer, TrainingArguments, get_wsd_schedule
from transformers import DataCollatorForLanguageModeling, default_data_collator
import evaluate
import torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import numpy as np

from src.model import make_model, make_tokenizer


def make_optimizer(model, learning_rate, weight_decay):
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer

def compute_accuracy_clf(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}

def compute_accuracy_lm(eval_pred):
    print(eval_pred)
    scores = []
    for reference, generated_text in zip(eval_pred.label_ids, eval_pred.predictions):
        # only compute accuracy for the last token ([ACTION] token)
        scores.append(reference[-1] == generated_text[-1])

    return {"action_accuracy": np.mean(scores)}

def compute_accuracy_lm(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    scores = (labels[-1] == preds[-1])
    return {"action_accuracy": np.mean(scores)}

def preprocess_logits_for_metrics_lm(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def run_training(args):
    if args.task == "clf": task = "text-classification" 
    elif args.task in ("lm", "lm-cot"): task = "text-generation"
    else: raise ValueError(f"Unknown task: {args.task}")

    learning_rate = 4e-4
    weight_decay = 0.01
    scheduler = "warmup_stable_decay" # linear, cosine or warmup_stable_decay
    num_warmup_steps = 500
    stable_pct = 0.9 # pct of steps in stable lr for wsd scheduler
    num_epochs = 5
    batch_size = 1024 if args.task == "clf" else 512
    grad_acc = 1 if args.task == "clf" else 2
    num_devices = 2
    min_lr_ratio = 0.1
    max_position_embeddings = 78 if args.task == "clf" else 116

    tokenizer = make_tokenizer(task=args.task)
    def encode(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    compute_metrics = compute_accuracy_clf if args.task == "clf" else compute_accuracy_lm
    preprocess_logits_for_metrics = None if args.task == "clf" else preprocess_logits_for_metrics_lm
    data_collator = default_data_collator if args.task == "clf" else DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer)
    # TODO: maybe pad seq lengths to multiple of 128
    # https://huggingface.co/docs/transformers/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling

    model = make_model({
        "pad_token_id": tokenizer.pad_token_id,
        "hidden_size": 256,             # embedding dimension from the paper
        "intermediate_size": 1024,      # not specified
        "num_hidden_layers": 8,         # as in the paper
        "num_attention_heads": 8,       # as in the paper
        "max_position_embeddings": max_position_embeddings,  # as in the paper (for bc and sv predictors, +1 for av)
        "torch_dtype": torch.bfloat16,
        #"attn_implementation": "flash_attention_2",
        #"device_map": "auto",
        "device": "cuda",
        "finetuning_task": task,
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
    else:
        dataset = dataset["train"].train_test_split(test_size=0.01)
    
    if args.max_samples and len(dataset["train"]) > args.max_samples:
        dataset["train"] = dataset["train"].select(range(args.max_samples))
    
    if task == "text-classification":
        dataset["train"] = dataset["train"].class_encode_column("label")
        dataset["train"] = dataset["train"].align_labels_with_mapping(
                label2id=model.config.label2id, label_column="label")
        class_label_feature = dataset["train"].features["label"]
        dataset["test"] = dataset["test"].cast_column("label", class_label_feature)

    dataset = dataset.map(encode, batched=True)
    print(dataset)

    total_steps = (len(dataset["train"]) // (batch_size * num_devices)) * num_epochs
    num_stable_steps = int(total_steps * stable_pct)
    num_decay_steps = total_steps - num_stable_steps

    if scheduler == "warmup_stable_decay":
        optimizer = make_optimizer(model, learning_rate, weight_decay)
        schedule = get_wsd_schedule(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_stable_steps=num_stable_steps,
            num_decay_steps=num_decay_steps,
            min_lr_ratio=min_lr_ratio
        )
    else:
        optimizer, schedule = None, None

    training_args = TrainingArguments(
        # 2 devices
        per_device_train_batch_size=batch_size, # bs 1024 in the paper on 4x 95G tpu, try to fit as much as possible ...
        gradient_accumulation_steps=grad_acc,    # ... else increase this
        gradient_checkpointing=False,     # save memory if needed, reduces speed
        bf16=True,
        learning_rate=learning_rate,      # as in the paper
        torch_compile=True,
        output_dir="checkpoints/save_"+args.run,
        per_device_eval_batch_size=int(batch_size/2),
        eval_strategy="steps",
        eval_steps=500,
        eval_on_start=True,
        num_train_epochs=num_epochs,      # 2.7-3.2 in the paper for ablations, 5.4 for full training
        #max_steps=5e6,        # 5e6 in the paper, 40m samples, bs 1024 -> 128 Epochs !?!
        lr_scheduler_type=scheduler,
        warmup_steps=num_warmup_steps,
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
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, schedule),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    model_class = str(type(model)).split("'")[1].split(".")[-1]
    print(f"training {model_class} with {model.num_parameters():,} parameters")
    trainer.train()

    trainer.save_model("checkpoints/save_"+args.run)
    tokenizer.save_pretrained("checkpoints/save_"+args.run)


if __name__ == "__main__":
    parser = ArgumentParser("Run training")
    parser.add_argument("dataset", help="Local or remote HF Dataset name")
    parser.add_argument("-task", default="clf", help="Training task (clf|lm|lm-cot)")
    parser.add_argument("-max_samples", type=int, default=40_000_000, help="Max Samples")
    parser.add_argument("-val", help="Local or remote HF Dataset name for validation")
    parser.add_argument("-max_steps", type=int, help="Max Steps")
    parser.add_argument("-run", help="W&B run name, None for no logging")
    args = parser.parse_args()

    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"]="ROOK"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"

    run_training(args)

    wandb.finish()
